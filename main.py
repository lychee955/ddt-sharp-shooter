import math
import re
import threading
import time
import tkinter
from dataclasses import replace
from tkinter import ttk
from tkinter import font as tkfont
from queue import Empty, Queue

from PIL import Image, ImageTk

from pyautogui import Point
from analysis import AnalysisWorker, TurnAnalysisWorker, MINIMAP_REGION, SnapshotAnalyzer, analyze_frame, crop_region
from vision import recognize_players
from turn import OwnTurnDetector
from config import dump_config, load_config
from window_region import bind_region, bind_window_at, resolve_region, shot_target, focus_shot_target, verify_shot_target
from sidecar_window import WindowsPanel, SidecarController, sidecar_settings
from shot import ShotController, ShotStateChanged, parse_force
from logger import logger, setup_logger
from runtime_diagnostics import setup_diagnostics, report_callback_exception
from force import calc_force
from km import (
    get_curr_mouse_pos,
    space_press,
    space_release,
    direction_press,
    direction_release,
    direction_tap,
    setup as setup_km,
    stop_listen as km_stop_listen,
)
from ocr import _capture_region, recognize, recognize_force, recognize_ten_units, recognize_wind


_GAME_CONFIG_PATH = "game_config.json"
_PRESS_DURATION_PER_FORCE = 4 / 100
_REF_GAME_REGION_WIDTH = 1500
_WIND_REGION = (709, 22, 84, 54)  # (x, y, w, h)
_DEG_REGION = (43, 835, 64, 36)  # (x, y, w, h)
_MINIMAP_REGION = MINIMAP_REGION  # Includes both edges and the full viewport.
_FORCE_REGION = (225, 860, 745, 28)  # (x, y, w, h)
_cmd_flag = 0
_cmd_typing = ""
_km_queue = Queue()
_stop_signal = False
_game_config = {
    "region": (0, 0, 0, 0),  # (x, y, w, h)
}
_ten_units_pixels = 0
_enemy_pos: tuple[int, int, int] | None = None  # dx, dy, enemy_left_side
_tmp_pos: Point = None
_tk: tkinter.Tk
AUTO_MODE = "自动分析"
MANUAL_MODE = "原有发射模式"
PREVIEW_WIDTH, PREVIEW_HEIGHT = 720, 380
_mode = AUTO_MODE
_analysis_worker: AnalysisWorker | None = None
_ui_actions = Queue()
_ui = {}
_preview_photo = None
_parameter_photos = []
_region_overlay = None
_region_canvas = None
_region_corner = None
_window_drag = None
_last_valid_result = None
_last_valid_at = None
_pending_failure = None
_displayed_result = None
_fire_cancel = threading.Event()
_shot_controller = None
_force_dialog = None
_force_dialog_pending = False
_last_s_press = None
_dialog_escape_until = 0
_shot_selection_ready = False
_sidecar = None


def km_listen_queue():
    global _stop_signal
    while not _stop_signal:
        inputs = _km_queue.get()
        if _stop_signal:
            break
        handle_inputs(inputs)


def resolve_force():
    """Valid cmds will be like:
    - directly give one force: `l30`
    - calculate force from wind, distance: `x12,y1,w-2,d65`
        means: dx=12, dy=1, wind=-2, degree=65
    """
    var_val = {"w": 0, "x": 0, "y": 0, "d": 0, "l": 0}
    try:
        for var, val in re.findall(r"([lwxyd])(-?\d+(?:\.\d+)?)", _cmd_typing):
            var_val[var] = float(val)
        if var_val["l"]:
            logger.info(f"Direct force:\n {var_val['l']}")
            return var_val["l"]
        logger.info(
            f"Wind: {var_val['w']}, Delta X: {var_val['x']=}, "
            f"Delta Y: {var_val['y']=}, Degree: {var_val['d']}"
        )
        return calc_force(var_val["d"], var_val["w"], var_val["x"], var_val["y"])
    except ValueError:
        logger.info("输入无效: 请检查输入格式.")


def _screen_region(reference_region: tuple[int, int, int, int]):
    """Scale offsets within the game, then add its screen position."""
    x, y, width, height = resolve_region(_game_config)
    if width <= 0 or height <= 0:
        raise ValueError("游戏区域无效，请拖动瞄准图标绑定游戏窗口")
    ratio = width / _REF_GAME_REGION_WIDTH
    rx, ry, rw, rh = reference_region
    region = (
        int(x + rx * ratio),
        int(y + ry * ratio),
        int(rw * ratio),
        int(rh * ratio),
    )
    if region[2] <= 0 or region[3] <= 0:
        raise ValueError("游戏区域过小，请重新标记完整游戏画面")
    return region


def recognize_and_fire():
    global _ten_units_pixels
    if _enemy_pos is None:
        raise ValueError("未标记敌我位置，请用 y 先标记自己、再标记敌人")
    dx, dy, enemy_left_side = _enemy_pos

    if not _ten_units_pixels:
        logger.info("十屏距离未标记，尝试自动识别...")
        rect_res = recognize_ten_units(_screen_region(_MINIMAP_REGION))
        if rect_res <= 0:
            raise ValueError("小地图标尺识别失败，请检查游戏区域、小地图是否被遮挡")
        _ten_units_pixels = rect_res

    if _ten_units_pixels <= 0:
        _ten_units_pixels = 0
        raise ValueError("小地图标尺无效，请按 Esc 后重新标记")

    dx = dx / _ten_units_pixels * 10
    dy = dy / _ten_units_pixels * 10
    logger.info(f"十屏距离: {_ten_units_pixels}, dx: {dx}, dy: {dy}")

    wind, left_more_dark = recognize_wind(_screen_region(_WIND_REGION))
    wind = wind * (
        -1
        if left_more_dark
        and not enemy_left_side
        or not left_more_dark
        and enemy_left_side
        else 1
    )
    logger.info(f"风速: {wind}")
    deg = recognize(_screen_region(_DEG_REGION)).strip()
    if not deg.isdigit() or not 0 <= int(deg) <= 180:
        raise ValueError(f"角度识别失败（结果：{deg!r}），请检查角度显示区域")
    logger.info(f"角度: {deg}")
    force = calc_force(int(deg), wind, dx, dy)
    if not math.isfinite(force) or not 0 < force <= 100:
        raise ValueError(f"发射力度无效：{force}，应在 0 到 100 之间（不含 0）")
    fire(force)


def reset_inputs(new_game=False):
    global _cmd_flag, _cmd_typing, _tmp_pos, _ten_units_pixels, _enemy_pos
    if new_game:
        _cmd_flag = 0
        _ten_units_pixels = 0
    _enemy_pos = None
    _cmd_typing = ""
    _tmp_pos = None
    logger.info("指令输入关闭." if new_game else "就绪.")


def handle_inputs(inputs: str):
    """To handle inputs"""
    global _cmd_flag, _cmd_typing, _tmp_pos, _ten_units_pixels, _enemy_pos
    global _last_s_press, _force_dialog_pending, _dialog_escape_until

    if not inputs:
        return

    drag = _window_drag
    if drag is not None:
        if inputs == "esc":
            drag["cancelled"] = True
            _ui_actions.put(("cancel_binding", None))
        return

    if _force_dialog is not None or _force_dialog_pending:
        if inputs == "esc":
            _force_dialog_pending = False
            _dialog_escape_until = time.monotonic()+.25
            _ui_actions.put(("cancel_force",None))
        return

    now = time.monotonic()
    if inputs == "esc" and _dialog_escape_until and now < _dialog_escape_until:
        return
    previous_s = _last_s_press
    _last_s_press = None
    if inputs.lower() == "s" and _region_overlay is None:
        if _shot_controller is not None and _shot_controller.busy:
            return
        if previous_s is not None and now-previous_s <= .5:
            _force_dialog_pending = True
            _ui_actions.put(("force_dialog",None))
        else:
            _last_s_press = now
        return

    if inputs == "r":
        _fire_cancel.set()
        if _shot_controller is not None:
            _shot_controller.cancel()
        _ui_actions.put(("mark_region", None))
        return

    if _region_overlay is not None and inputs != "esc":
        return

    if _shot_controller is not None and _shot_controller.busy:
        if inputs != "esc":
            return

    # Analysis never fires; ss opens a separate user-confirmed shot dialog.
    if _mode == AUTO_MODE:
        if inputs == "t":
            if _analysis_worker:
                _ui_actions.put(("refresh", None))
            return
        if inputs != "esc":
            return

    # press ESC to cancel
    if inputs == "esc":
        _fire_cancel.set()
        if _shot_controller is not None:
            _shot_controller.cancel()
        _ui_actions.put(("paused", None))
    # press the key 't' twice to enable command mode
    elif inputs == "t":
        if _cmd_flag == 2:
            if _enemy_pos or _cmd_typing:
                if _enemy_pos:
                    try:
                        recognize_and_fire()
                    except Exception as exc:
                        logger.exception("自动识别或发射失败")
                        logger.info(f"失败原因：{type(exc).__name__}: {exc}")
                        time.sleep(1)
                elif _cmd_typing:
                    try:
                        direct_force = resolve_force()
                        if direct_force and direct_force > 0:
                            fire(force=direct_force)
                        else:
                            logger.info("输入无效: 请检查输入格式.")
                    except Exception as exc:
                        logger.exception("手动发射失败")
                        logger.info(f"失败原因：{type(exc).__name__}: {exc}")
                reset_inputs()
                return
        _cmd_flag += 1
        _cmd_flag %= 3
        if _cmd_flag == 2:
            logger.info("指令输入开启..")
        elif _cmd_flag == 0:
            reset_inputs()
    elif _cmd_flag == 2:
        if inputs == "delete":
            _cmd_typing = _cmd_typing[:-1]
        elif inputs == "e":
            if _tmp_pos:
                pos = get_curr_mouse_pos()
                _ten_units_pixels = abs(pos.y - _tmp_pos.y)
                logger.info(f"十屏距离: {_ten_units_pixels}")
                _tmp_pos = None
                return
            logger.info("标记十屏距离.")
            _tmp_pos = get_curr_mouse_pos()
        elif inputs == "y":
            if _tmp_pos:
                pos = get_curr_mouse_pos()
                _enemy_pos = (
                    abs(pos.x - _tmp_pos.x),
                    (_tmp_pos.y - pos.y),
                    _tmp_pos.x > pos.x,
                )
                _tmp_pos = None
                logger.info("标记完成.")
                return
            logger.info("标记敌我.")
            _tmp_pos = get_curr_mouse_pos()
        else:
            _cmd_typing += inputs

    # when not in command mode
    # any key except 't' will reset mode flag
    # which means only consecutive 't' input can enable command mode
    elif _cmd_flag == 1:
        reset_inputs()


def calc_duration(force):
    return _PRESS_DURATION_PER_FORCE * force


def _get_curr_force():
    return recognize_force(_screen_region(_FORCE_REGION))


def fire(force: int):
    """Steps to fire:
    - Calculate force
    - Press space to store force,
    and then release to fire
    """
    if _mode != MANUAL_MODE:
        raise ValueError("该键盘发射流程仅支持原有发射模式；自动分析模式请选中行点击发射")
    if not math.isfinite(force) or not 0 < force <= 100:
        raise ValueError("发射力度应大于 0 且不超过 100")
    _fire_cancel.clear()
    if _fire_cancel.wait(1.5) or _mode != MANUAL_MODE:
        return
    logger.info(f"发射力度: {force}")
    logger.info("发射!")
    space_press()
    try:
        _fire_cancel.wait(_PRESS_DURATION_PER_FORCE * force)
    finally:
        space_release()


def on_destroy(_=None):
    global _stop_signal
    if _stop_signal:
        return
    logger.info("用户关闭辅助窗口")
    _stop_signal = True
    _fire_cancel.set()
    cancel_window_binding(resume=False)
    if _shot_controller is not None:
        _shot_controller.close()
    close_force_dialog()
    if _analysis_worker:
        _analysis_worker.close()
    # put something to break the km_queue blocking
    _km_queue.put("stop")
    km_stop_listen()
    if _sidecar is not None:
        _sidecar.close()
    _tk.destroy()


def select_mode():
    global _mode
    _fire_cancel.set()
    if _shot_controller is not None:
        _shot_controller.cancel()
    close_force_dialog()
    cancel_window_binding(resume=False)
    if _region_overlay is not None:
        cancel_region_selection()
    _mode = _ui["mode"].get()
    reset_inputs(True)
    _analysis_worker.configure(_game_config["region"], _mode == AUTO_MODE)
    clear_results("选中目标行后点击下方发射；请先调整角度和道具" if _mode == AUTO_MODE else "发射模式：tt 开启输入，y 标记或输入 l30，t 发射")
    update_controls()


def toggle_analysis():
    if _mode != AUTO_MODE:
        return
    if _analysis_worker.running:
        _analysis_worker.pause()
        _ui["status"].set("已暂停监听，保留上次结果；t 可手动计算一次")
    else:
        clear_shot_selection()
        _analysis_worker.configure(_game_config["region"], True)
        _ui["status"].set("等待轮到你出手；t 可手动计算一次" if _game_config["region"][2] > 0 else "请将瞄准图标拖到游戏画面内绑定窗口")
    update_controls()


def refresh_analysis():
    if _mode == AUTO_MODE:
        clear_shot_selection()
        _analysis_worker.refresh()


def update_controls():
    automatic = _mode == AUTO_MODE
    selecting = _region_overlay is not None or _window_drag is not None
    _ui["start"].configure(text="暂停监听" if _analysis_worker.running else "开始监听",
                           state="normal" if automatic and not selecting else "disabled")
    _ui["refresh"].configure(state="normal" if automatic and not selecting else "disabled")
    _ui["mark"].configure(state="disabled" if selecting else "normal")
    if "shoot" in _ui:
        update_shot_controls()


def prepare_window_binding(_=None):
    global _window_drag, _last_s_press
    if _window_drag is not None or _region_overlay is not None:
        return False
    _window_drag = {"running": _analysis_worker.running}
    _fire_cancel.set()
    if _shot_controller is not None:
        _shot_controller.cancel()
    close_force_dialog()
    _last_s_press = None
    _analysis_worker.pause()
    reset_inputs(True)
    _ui["status"].set("拖到游戏画面内松开即可绑定；Esc 取消")
    region_prompt("正在绑定窗口：避开工具栏和下方力度表")
    update_controls()
    return True


def start_window_drag(_=None):
    if not prepare_window_binding():
        return "break"
    try:
        _ui["finder"].configure(cursor="crosshair")
        _ui["finder"].grab_set_global()
        _tk.configure(cursor="crosshair")
    except tkinter.TclError:
        cancel_window_binding()
    return "break"


def release_window_drag():
    global _window_drag
    previous = _window_drag
    _window_drag = None
    for widget, cursor in ((_ui.get("finder"), "hand2"), (_tk, "")):
        if widget is not None:
            try:
                widget.grab_release()
                widget.configure(cursor=cursor)
            except tkinter.TclError:
                pass
    return previous


def cancel_window_binding(_=None, *, resume=True):
    global _dialog_escape_until
    if _window_drag is not None:
        if getattr(_, "keysym", None) == "Escape":
            _dialog_escape_until = time.monotonic()+.25
        previous = release_window_drag()
        if resume:
            _analysis_worker.configure(_game_config["region"], previous["running"])
        region_prompt("已取消绑定，原窗口保留；将瞄准图标拖到游戏内可重新绑定")
        _ui["status"].set("等待轮到你出手" if resume and previous["running"] else "已暂停")
        update_controls()
    return "break"


def finish_window_binding(point):
    global _game_config, _ten_units_pixels, _enemy_pos, _tmp_pos
    if _window_drag is None:
        return "break"
    if _window_drag.get("cancelled"):
        return cancel_window_binding()
    previous = release_window_drag()
    try:
        target = bind_window_at(point)
        config = dict(_game_config, **target)
        # Validate again before saving in case the game was closed during lookup.
        config["region"] = resolve_region(config)
        dump_config(config, _GAME_CONFIG_PATH)
    except Exception as exc:
        _analysis_worker.configure(_game_config["region"], previous["running"])
        region_prompt(f"绑定失败：{exc}；原窗口保留")
        _ui["status"].set("绑定未更改，请将图标拖到完整游戏画面内重试")
        update_controls()
        return "break"
    _game_config = config
    _ten_units_pixels, _enemy_pos, _tmp_pos = 0, None, None
    clear_results("等待轮到你出手；t 可手动计算一次" if _mode == AUTO_MODE else "游戏窗口已绑定")
    _analysis_worker.configure(config["region"], _mode == AUTO_MODE)
    width, height = config["region"][2:]
    region_prompt(f"已绑定游戏窗口（{width} × {height}），移动和缩放后自动跟随")
    logger.info(f"窗口绑定成功，游戏区域：{config['region']}")
    update_controls()
    return "break"


def finish_window_drag(event):
    return finish_window_binding((event.x_root, event.y_root))


def bind_game_under_mouse():
    if prepare_window_binding():
        finish_window_binding(tuple(get_curr_mouse_pos()))


def region_prompt(message):
    _ui["calibration"].set(message)
    if _region_canvas is not None:
        _region_canvas.itemconfigure("prompt", text=message+"\n只标游戏画面，不包含标题栏或下方参考表。Esc 取消")
    logger.info(message)


def close_region_overlay():
    global _region_overlay, _region_canvas, _region_corner
    overlay = _region_overlay
    # Clear state first: a repeated local/global Escape must not destroy or
    # release the same overlay twice, even if Tk reports it already closed.
    _region_overlay = _region_canvas = _region_corner = None
    if overlay is not None:
        try:
            overlay.grab_release()
        except tkinter.TclError:
            pass
        try:
            overlay.destroy()
        except tkinter.TclError:
            pass


def cancel_region_selection(_=None):
    if _region_overlay is None:
        return "break"
    close_region_overlay()
    region_prompt("已取消校准，原游戏区域保留；拖动图标绑定或点击精细校准重新开始")
    _ui["status"].set("已暂停")
    update_controls()
    return "break"


def record_region_corner(pos):
    global _region_corner, _ten_units_pixels, _enemy_pos, _tmp_pos, _game_config
    if _region_corner is None:
        _region_corner = pos
        region_prompt("左上角已标记，请点击游戏画面右下角")
        return
    if pos.x <= _region_corner.x or pos.y <= _region_corner.y:
        region_prompt("右下角位置无效，请在已标记左上角的右下方重新点击")
        return
    region = (_region_corner.x, _region_corner.y,
              pos.x-_region_corner.x, pos.y-_region_corner.y)
    config = dict(_game_config, region=region)
    config.pop("window", None)
    binding = bind_region(region)
    if binding is not None:
        config["window"] = binding
    try:
        dump_config(config, _GAME_CONFIG_PATH)
    except Exception as exc:
        region_prompt(f"游戏区域保存失败：{exc}；请重试点击右下角")
        return
    _game_config = config
    _ten_units_pixels, _enemy_pos, _tmp_pos = 0, None, None
    close_region_overlay()
    clear_results("等待轮到你出手；t 可手动计算一次" if _mode == AUTO_MODE else "游戏区域已更新")
    follow = "；已绑定窗口，移动后自动跟随" if binding else "；使用固定区域，移动后请重新标记"
    region_prompt("游戏区域标记成功，已保存"+follow+ ("；自动分析已恢复" if _mode == AUTO_MODE else ""))
    logger.info(f"游戏区域：{region}")
    _analysis_worker.configure(region, _mode == AUTO_MODE)
    update_controls()


def start_region_selection():
    global _region_overlay, _region_canvas, _region_corner
    if _window_drag is not None:
        return
    if _region_overlay is not None:
        _region_overlay.lift()
        return
    _fire_cancel.set()
    if _shot_controller is not None:
        _shot_controller.cancel()
    close_force_dialog()
    _analysis_worker.pause()
    reset_inputs(True)
    clear_results("正在设置游戏区域，自动分析已暂停")
    _region_corner = None
    overlay = tkinter.Toplevel(_tk)
    _region_overlay = overlay
    overlay.overrideredirect(True)
    x, y = _tk.winfo_vrootx(), _tk.winfo_vrooty()
    width, height = _tk.winfo_vrootwidth(), _tk.winfo_vrootheight()
    overlay.geometry(f"{width}x{height}{x:+d}{y:+d}")
    overlay.wm_attributes("-topmost", True)
    overlay.wm_attributes("-alpha", .4)
    _region_canvas = tkinter.Canvas(overlay, bg="#101820", highlightthickness=0, cursor="crosshair")
    _region_canvas.pack(fill="both", expand=True)
    _region_canvas.create_text(width/2, 70, fill="white", font=("Microsoft YaHei", 16, "bold"),
                               tags="prompt", width=width-80, justify="center")
    # Wait for release so both mouse events are consumed by the overlay.
    _region_canvas.bind("<ButtonRelease-1>", lambda event: record_region_corner(Point(event.x_root, event.y_root)))
    def outline(event):
        _region_canvas.delete("selection")
        if _region_corner is not None:
            _region_canvas.create_rectangle(_region_corner.x-x, _region_corner.y-y,
                                            event.x, event.y, outline="#50cfff", width=3, tags="selection")
    _region_canvas.bind("<Motion>", outline)
    overlay.bind("<Escape>", cancel_region_selection)
    overlay.grab_set()
    overlay.focus_force()
    region_prompt("请点击游戏画面左上角")
    update_controls()


def close_force_dialog(_=None):
    global _force_dialog, _force_dialog_pending, _last_s_press, _dialog_escape_until
    if getattr(_,"keysym",None) == "Escape":
        # Tk and the global listener receive the same physical Escape.
        # Its later copy should not pause analysis after closing the dialog.
        _dialog_escape_until = time.monotonic()+.25
    dialog = _force_dialog
    _force_dialog = None
    _force_dialog_pending = False
    _last_s_press = None
    if dialog is not None:
        for operation in (dialog.grab_release,dialog.destroy):
            try:
                operation()
            except tkinter.TclError:
                pass
    if "shoot" in _ui:
        update_shot_controls()
    return "break"


def open_force_dialog():
    global _force_dialog, _force_dialog_pending, _dialog_escape_until
    if not _force_dialog_pending or _force_dialog is not None:
        return
    _force_dialog_pending = False
    if _region_overlay is not None or _window_drag is not None or _shot_controller is None or _shot_controller.busy:
        return
    try:
        target = shot_target(_game_config)
    except (ValueError,OSError) as exc:
        _ui["status"].set(str(exc))
        return
    _dialog_escape_until = 0
    reset_inputs()
    dialog = _force_dialog = tkinter.Toplevel(_tk)
    dialog.title("指定力度发射")
    dialog.transient(_tk)
    dialog.resizable(False,False)
    dialog.wm_attributes("-topmost",True)
    frame = ttk.Frame(dialog,padding=16)
    frame.pack(fill="both",expand=True)
    ttk.Label(frame,text="输入力度（大于 0，最大 100，支持小数）").pack(anchor="w")
    value = tkinter.StringVar()
    entry = ttk.Entry(frame,textvariable=value,width=30)
    entry.pack(fill="x",pady=10)
    error = tkinter.StringVar()
    ttk.Label(frame,textvariable=error,foreground="#b33b30").pack(anchor="w")
    def submit(_=None):
        if _force_dialog is not dialog:
            return "break"
        try:
            force = parse_force(value.get())
        except ValueError as exc:
            error.set(str(exc))
            entry.focus_set()
            return "break"
        close_force_dialog()
        if _shot_controller.submit(force,target):
            _ui["status"].set(f"正在按指定力度 {force:g} 发射；Esc 可中止蓄力")
        return "break"
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x",pady=(12,0))
    ttk.Button(buttons,text="取消",command=close_force_dialog).pack(side="right")
    ttk.Button(buttons,text="发射",command=submit).pack(side="right",padx=8)
    dialog.bind("<Return>",submit)
    dialog.bind("<Escape>",close_force_dialog)
    dialog.protocol("WM_DELETE_WINDOW",close_force_dialog)
    dialog.grab_set()
    entry.focus_force()
    if "shoot" in _ui:
        update_shot_controls()


def clear_shot_selection():
    global _shot_selection_ready
    _shot_selection_ready = False
    if "shoot" in _ui:
        table = _ui["table"]
        table.selection_remove(*table.selection())
        update_shot_controls()


def selected_shot():
    """Resolve only a selected row from the currently displayed snapshot."""
    if _mode != AUTO_MODE:
        raise ValueError("请切换到自动分析模式后选择目标")
    result = _displayed_result
    if (not _shot_selection_ready or result is None or result.error
            or result.stale_age is not None or result.phase in {"computing", "failed"}):
        raise ValueError("等待本次计算完成，再选择目标")
    rows = _ui["table"].selection()
    if len(rows) != 1:
        raise ValueError("请选择一个目标行")
    try:
        index = int(rows[0])
        if not 0 <= index < len(result.targets):
            raise ValueError
        estimate = result.targets[index]
    except (ValueError, IndexError):
        raise ValueError("人物列表已更新，请重新选择目标") from None
    if estimate.player.identity == "自己":
        raise ValueError("不能选择自己作为发射目标")
    if estimate.status != "可用" or estimate.direction not in {"左", "右"}:
        raise ValueError("该目标当前没有可用的发射力度")
    parse_force(str(estimate.force))
    force = parse_force(f"{estimate.force:.2f}")
    return estimate, force, "left" if estimate.direction == "左" else "right"


def update_shot_controls(_=None):
    if "shoot" not in _ui:
        return
    enabled = False
    try:
        estimate, force, direction = selected_shot()
        summary = f"{estimate.player.label} · 向{estimate.direction} · 力度 {force:.2f}"
        if _region_overlay is not None or _window_drag is not None:
            summary += " · 正在绑定或校准"
        elif _force_dialog is not None or _force_dialog_pending:
            summary += " · 请先关闭指定力度弹框"
        elif _shot_controller is None or _shot_controller.busy:
            summary += " · 发射中" if _shot_controller is not None else " · 发射尚未就绪"
        else:
            shot_target(_game_config)
            enabled = True
    except (ValueError, OSError) as exc:
        summary = str(exc)
    _ui["selected_target"].set(summary)
    _ui["shoot"].configure(state="normal" if enabled else "disabled")
    if "target_details" in _ui:
        update_target_details()


def update_target_details():
    """Details stay readable even for self/failed rows that cannot be fired."""
    result = _displayed_result
    rows = _ui["table"].selection()
    message = "选中人物查看水平距离、高低差和完整状态"
    if result is not None and len(rows) == 1:
        try:
            index = int(rows[0])
            if not 0 <= index < len(result.targets):
                raise ValueError
            target = result.targets[index]
            fmt = lambda value: "—" if value is None else f"{value:.2f}"
            state = "旧结果，等待重识别" if result.stale_age is not None else target.status
            message = (f"{target.player.label} · {target.player.identity} · 向{target.direction or '—'}\n"
                       f"水平距离 {fmt(target.dx)} · 高低差 {fmt(target.dy)}\n{state}")
        except (ValueError, IndexError):
            pass
    _ui["target_details"].set(message)


def fire_selected_target():
    """A button click authorizes one shot with the displayed force and direction."""
    global _dialog_escape_until
    if (_region_overlay is not None or _window_drag is not None
            or _force_dialog is not None or _force_dialog_pending
            or _shot_controller is None or _shot_controller.busy):
        update_shot_controls()
        return
    try:
        estimate, force, direction = selected_shot()
        target = shot_target(_game_config)
        if _displayed_result.input_state is None:
            raise ValueError("当前结果缺少发射校验信息，请按 t 重新计算")
        target = dict(target, shot_state=_displayed_result.input_state)
        if _shot_controller.submit(force, target, direction=direction):
            _dialog_escape_until = 0
            reset_inputs()
            message = f"正在向{estimate.direction}发射 {estimate.player.label}，力度 {force:.2f}；Esc 可中止"
            _ui["status"].set(message)
            logger.info(message)
    except (ValueError, OSError) as exc:
        _ui["status"].set(str(exc))
    update_shot_controls()


def report_shot_status(message):
    logger.info(message)
    _ui_actions.put(("shot_status", message))


def check_shot_state(target):
    """Fresh, read-only checks using the frozen binding and selected self."""
    baseline = target.get("shot_state")
    if baseline is None:
        raise ShotStateChanged("当前结果缺少角度和位置，请重新计算")
    frame = _capture_region(resolve_region(target))
    ratio = frame.shape[1] / _REF_GAME_REGION_WIDTH
    minimap = crop_region(frame, MINIMAP_REGION)
    # manual_hint returns the measured centre. own_hint would substitute the
    # cached coordinates during a halo gap and could conceal a small movement.
    detection = recognize_players(minimap, ratio,
                                  manual_hint=(baseline.x*ratio, baseline.y*ratio))
    if detection.error:
        raise ShotStateChanged("无法确认自己的当前位置")
    own = next(p for p in detection.players if p.identity == "自己")
    if math.hypot(own.x/ratio-baseline.x, own.y/ratio-baseline.y) >= 1:
        raise ShotStateChanged("人物位置发生变化，原距离和高低差已失效")
    text = recognize(crop_region(frame, _DEG_REGION)).strip()
    if not text.isdigit() or not 0 <= int(text) <= 180:
        raise ShotStateChanged("无法确认当前角度")
    degree = int(text)
    # A left/right mirror can show 65 -> 115 while retaining the same elevation.
    if min(degree, 180-degree) != min(baseline.degree, 180-baseline.degree):
        raise ShotStateChanged(f"角度由 {baseline.degree} 变为 {degree}，原力度已失效")


def refresh_after_shot_change():
    _ui_actions.put(("refresh", None))


def clear_results(message):
    global _last_valid_result, _last_valid_at, _pending_failure, _displayed_result
    _last_valid_result = _last_valid_at = _pending_failure = None
    _displayed_result = None
    clear_shot_selection()
    _ui["table"].delete(*_ui["table"].get_children())
    _ui["canvas"].delete("all")
    _ui["parameters"].delete("all")
    _ui["summary"].set("等待识别")
    _ui["status"].set(message)


def display_analysis_result(result, now=None):
    """Retain a complete old snapshot briefly, always labelled as stale."""
    global _last_valid_result, _last_valid_at, _pending_failure
    now = time.monotonic() if now is None else now
    if result.phase in {"computing", "failed"}:
        _last_valid_result = _last_valid_at = _pending_failure = None
        render_result(result)
        return
    if not result.error:
        _last_valid_result, _last_valid_at, _pending_failure = result, now, None
        render_result(result)
    elif (result.error_kind in {"self_missing", "parameters"}
          and _last_valid_result is not None and now-_last_valid_at < 2):
        _pending_failure = result
        age = now-_last_valid_at
        render_result(replace(_last_valid_result, stale_age=age,
                              error=f"旧结果，等待重识别（{age:.1f} 秒前）；{result.error}"))
    else:
        _last_valid_result = _last_valid_at = _pending_failure = None
        render_result(result)


def expire_stale_result(now=None):
    global _last_valid_result, _last_valid_at, _pending_failure
    now = time.monotonic() if now is None else now
    if _pending_failure is not None and now-_last_valid_at >= 2:
        failure = _pending_failure
        _last_valid_result = _last_valid_at = _pending_failure = None
        render_result(failure)


def select_own_player(event):
    if _mode != AUTO_MODE or _region_overlay is not None or _window_drag is not None:
        return
    table = _ui["table"]
    if table.identify_region(event.x, event.y) != "cell" or table.identify_column(event.x) != "#1":
        return
    row = table.identify_row(event.y)
    result = _displayed_result
    if not row or result is None:
        return
    if result.stale_age is not None or result.frame is None:
        _ui["status"].set("请按 t 获取当前截图，再勾选自己")
        return "break"
    index = int(row)
    target = result.targets[index]
    own_index = None if result.manual_self and target.player.identity == "自己" else index
    clear_shot_selection()
    try:
        _analysis_worker.select_self(result, own_index)
    except ValueError as exc:
        _ui["status"].set(str(exc))
        return "break"
    _ui["status"].set("已取消手动选择，正在恢复蓝圈识别…" if own_index is None
                      else f"已选择 {target.player.label} 为自己，正在按显示的截图重算力度…")
    return "break"


def render_result(result):
    global _preview_photo, _parameter_photos, _displayed_result, _shot_selection_ready
    clear_shot_selection()
    _displayed_result = result
    _shot_selection_ready = not result.error and result.stale_age is None and result.phase not in {"computing", "failed"}
    table, canvas = _ui["table"], _ui["canvas"]
    table.delete(*table.get_children())
    counts = {kind: sum(t.player.identity == kind for t in result.targets)
              for kind in ("自己", "队友", "敌人", "身份不确定")}
    clock = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
    wind = "—" if result.wind is None else f"{result.wind:.1f} {result.wind_direction}"
    degree = "—" if result.degree is None else f"{result.degree}°"
    _ui["summary"].set(f"自己 {counts['自己']} · 队友 {counts['队友']} · 敌人 {counts['敌人']} · 未确认 {counts['身份不确定']}    风力 {wind}    角度 {degree}    更新 {clock}")
    if result.phase == "computing":
        status = result.error or "正在计算本次出手力度…"
    elif result.phase == "complete":
        status = f"本次计算完成（尝试 {result.attempts} 次）；本轮角度/位置变化时重算，结束后等待下次出手；t 可重算"
        if result.tracking_note:
            status += "；"+result.tracking_note
    elif result.phase == "failed":
        status = f"本次计算未完成（尝试 {result.attempts} 次）；{result.error}；t 可重试"
    else:
        status = result.error or result.tracking_note or "计算完成；t 可重新计算"
    if result.error_kind in {"self_missing", "self_ambiguous"} and result.targets:
        status += "；可在左侧“我”列勾选自己后计算"
    if _shot_controller is None or not _shot_controller.busy:
        _ui["status"].set(status)
    for index, target in enumerate(result.targets):
        fmt = lambda value: "—" if value is None else f"{value:.2f}"
        checked = "☑" if result.manual_self and target.player.identity == "自己" else "☐"
        table.insert("", "end", iid=str(index), values=(checked, target.player.label, target.player.identity,
                     target.direction, fmt(target.dx), fmt(target.dy), fmt(target.force),
                     "旧结果，等待重识别" if result.stale_age is not None else target.status),
                     tags=(target.player.identity,))
    canvas.delete("all")
    parameter_canvas = _ui["parameters"]
    parameter_canvas.delete("all")
    _parameter_photos = []
    for x, label, crop in ((0, "风力截取", result.wind_image), (130, "角度截取", result.degree_image)):
        parameter_canvas.create_text(x+58, 8, text=label, fill="#c8ced8", font=("Microsoft YaHei", 8))
        if crop is not None:
            image = Image.fromarray(crop)
            scale = min(116/image.width, 48/image.height)
            photo = ImageTk.PhotoImage(image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.NEAREST))
            _parameter_photos.append(photo)
            parameter_canvas.create_image(x, 22, anchor="nw", image=photo)
    draw_minimap(result)
    update_shot_controls()


def draw_minimap(result):
    """Resize the displayed snapshot without re-analysis or clearing selection."""
    global _preview_photo
    canvas = _ui["canvas"]
    canvas.delete("all")
    if result is not None and result.minimap is not None:
        image = Image.fromarray(result.minimap)
        factor = min(int(canvas["width"]) / image.width, int(canvas["height"]) / image.height)
        size = (round(image.width * factor), round(image.height * factor))
        _preview_photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR))
        canvas.create_image(0, 0, anchor="nw", image=_preview_photo)
        colours = {"自己": "#50cfff", "队友": "#70f898", "敌人": "#ff748d", "身份不确定": "#ffcf65"}
        for target in result.targets:
            player = target.player
            x, y = player.x * factor, player.y * factor
            canvas.create_oval(x-9, y-9, x+9, y+9, outline=colours[player.identity], width=2)
            canvas.create_text(min(max(x, 15), size[0]-15), max(y-16, 9),
                               text=player.label, fill="white", font=("Microsoft YaHei", 10, "bold"))


def toggle_sidecar():
    global _game_config
    if _sidecar is None:
        _ui["sidecar_enabled"].set(False)
        _ui["independent_focus"].set(False)
        _ui["sidecar_status"].set("当前系统不支持窗口联动")
        return
    previous = _game_config
    settings = sidecar_settings(previous)
    settings["enabled"] = bool(_ui["sidecar_enabled"].get())
    settings["independent_focus"] = bool(_ui["independent_focus"].get())
    settings["width"] = int(_ui["sidecar_width"].get())
    saved = previous.get("sidecar", {})
    config = dict(previous, sidecar=dict(saved if isinstance(saved, dict) else {}, **settings))
    try:
        dump_config(config, _GAME_CONFIG_PATH)
    except Exception as exc:
        _ui["sidecar_enabled"].set(sidecar_settings(previous)["enabled"])
        _ui["independent_focus"].set(sidecar_settings(previous)["independent_focus"])
        _ui["sidecar_width"].set(str(sidecar_settings(previous)["width"]))
        _ui["sidecar_status"].set(f"侧栏设置保存失败：{exc}")
        logger.exception("侧栏设置保存失败")
        return
    _game_config = config
    _sidecar.set_enabled(settings["enabled"], config)


def cycle_sidecar_width():
    choices = (360, 440, 520, 640)
    current = int(_ui["sidecar_width"].get())
    _ui["sidecar_width"].set(str(choices[(choices.index(current)+1) % len(choices)]
                                 if current in choices else 440))
    toggle_sidecar()


def probe_sidecar_focus():
    if _sidecar is None:
        _ui["sidecar_status"].set("当前系统不支持窗口焦点检查")
        return
    try:
        facts = _sidecar.focus_probe(_game_config)
        logger.info("侧栏焦点检查：%s", facts)
        _ui["focus_status"].set(
            f"大厅前台：{'是' if facts['hall_foreground'] else '否'} · "
            f"Flash 焦点：{'是' if facts['flash_focus'] else '否'}\n"
            "画面是否常亮需同时观察；出现灰幕时点击游戏恢复")
    except (ValueError, OSError) as exc:
        _ui["focus_status"].set(str(exc))


def report_sidecar_status(message):
    _ui["sidecar_status"].set(message)
    logger.info(message)


def invalidate_sidecar_binding():
    _fire_cancel.set()
    if _shot_controller is not None:
        _shot_controller.cancel()
    if _analysis_worker is not None:
        _analysis_worker.pause()
    clear_results("游戏绑定失效，请重新拖动图标绑定后恢复监听")
    update_controls()


def poll_ui():
    if _stop_signal:
        return
    try:
        sidecar = globals().get("_sidecar")
        if sidecar is not None:
            sidecar.poll(_game_config)
        poll_ui_events()
    except Exception:
        logger.exception("界面轮询发生异常，将继续监听")
    finally:
        if not _stop_signal:
            _tk.after(100, poll_ui)


def poll_ui_events():
    shot_message = None
    while True:
        try:
            action, value = _ui_actions.get_nowait()
        except Empty:
            break
        if action == "quit":
            on_destroy()
            return
        if action == "clear":
            clear_results(value)
        elif action == "refresh":
            refresh_analysis()
        elif action == "mark_region":
            bind_game_under_mouse()
        elif action == "cancel_binding":
            cancel_window_binding()
        elif action == "force_dialog":
            open_force_dialog()
        elif action == "cancel_force":
            close_force_dialog()
        elif action == "shot_status":
            shot_message = value
        elif action == "paused":
            _analysis_worker.pause()
            reset_inputs(True)
            if _region_overlay is not None:
                cancel_region_selection()
            _ui["status"].set("已暂停；拖动图标可重新绑定，点击开始恢复自动分析")
        update_controls()
    result = _analysis_worker.take_result()
    if result is not None and _mode == AUTO_MODE:
        display_analysis_result(result)
    if _mode == AUTO_MODE:
        expire_stale_result()
    if shot_message is not None:
        _ui["status"].set(shot_message)
    if "shoot" in _ui:
        update_shot_controls()


def build_ui(root):
    root.title("DSS · 多人力度分析")
    available_width = max(320, root.winfo_screenwidth()-80)
    available_height = max(320, root.winfo_screenheight()-100)
    root.geometry(f"{min(1120, available_width)}x{min(1000, available_height)}")
    root.minsize(min(760, available_width), min(600, available_height))
    root.wm_attributes("-topmost", True)
    root.configure(bg="#20242b")
    style = ttk.Style(root)
    style.theme_use("clam")
    table_font = tkfont.Font(root=root, family="Microsoft YaHei", size=9)
    style.configure("Treeview", font=table_font, rowheight=table_font.metrics("linespace")+8,
                    background="#292e38", fieldbackground="#292e38", foreground="white")
    style.configure("Treeview.Heading", font=table_font, background="#394150", foreground="white")
    style.map("Treeview", background=[("selected", "#315d86")],
              foreground=[("selected", "white")])
    container = tkinter.Frame(root, bg="#20242b")
    container.pack(fill="both", expand=True, padx=12, pady=12)
    # Reserve the footer before allocating the scrollable preview and table.
    footer = tkinter.Frame(container, bg="#20242b")
    footer.pack(side="bottom", fill="x")
    shot_bar = tkinter.Frame(footer, bg="#20242b")
    shot_bar.pack(fill="x", pady=(8, 0))
    shoot = ttk.Button(shot_bar, text="发射", command=fire_selected_target, state="disabled")
    shoot.pack(side="right", padx=(12, 0))
    selected_target = tkinter.StringVar(master=root, value="请选择一个目标行")
    selected_label = tkinter.Label(shot_bar, textvariable=selected_target, bg="#20242b", fg="#c8ced8",
                                    anchor="w", justify="left")
    selected_label.pack(side="left", fill="x", expand=True)
    text_widget = tkinter.Text(footer, height=3, width=1, border=0, bg="#15181f",
                               fg="#c8ced8", state="disabled", wrap="word")
    text_widget.pack(fill="x", pady=(10, 0))
    target_details = tkinter.StringVar(master=root, value="选中人物查看水平距离、高低差和完整状态")
    details_label = tkinter.Label(footer, textvariable=target_details, bg="#20242b", fg="#c8ced8",
                                 anchor="w", justify="left", wraplength=900)
    details_label.pack(fill="x", before=shot_bar, pady=(6, 0))
    dockbar = tkinter.Frame(container, bg="#20242b")
    dockbar.pack(fill="x", pady=(0, 8))
    dock_actions = tkinter.Frame(dockbar, bg="#20242b")
    dock_actions.pack(fill="x")
    finder = tkinter.Frame(dock_actions, bg="#20242b", cursor="hand2",
                           takefocus=True, padx=2, pady=2)
    finder.pack(side="left", padx=(0, 8))
    finder_icon = tkinter.Canvas(finder, width=32, height=32, bg="#20242b", highlightthickness=1,
                                 highlightbackground="#50cfff", cursor="hand2", takefocus=False)
    finder_icon.pack(side="left", padx=(0, 4))
    finder_icon.create_oval(7, 7, 25, 25, outline="#50cfff", width=2)
    finder_icon.create_line(16, 2, 16, 30, fill="#50cfff", width=2)
    finder_icon.create_line(2, 16, 30, 16, fill="#50cfff", width=2)
    finder_label = tkinter.Label(finder, text="拖动绑定", bg="#20242b", fg="#50cfff", cursor="hand2")
    finder_label.pack(side="left", padx=(0, 4))
    # Tk child events do not bubble to their containing frame. Start dragging
    # from the icon, text or padding; the frame owns the global release grab.
    for widget in (finder, finder_icon, finder_label):
        widget.bind("<ButtonPress-1>", start_window_drag)
        widget.bind("<ButtonRelease-1>", finish_window_drag)
        widget.bind("<Escape>", cancel_window_binding)
    sidecar_enabled = tkinter.BooleanVar(master=root, value=False)
    sidecar_width = tkinter.StringVar(master=root, value="440")
    ttk.Checkbutton(dock_actions, text="连接右侧", variable=sidecar_enabled,
                    command=toggle_sidecar).pack(side="left")
    ttk.Label(dock_actions, text="宽度").pack(side="left", padx=(8, 2))
    # Buttons avoid native combobox popups, which would take foreground focus
    # even though the attached panel itself is non-activating.
    width_selector = ttk.Button(dock_actions, textvariable=sidecar_width,
                                 command=cycle_sidecar_width, width=4, takefocus=False)
    width_selector.pack(side="left")
    help_dialog = None
    def show_help():
        nonlocal help_dialog
        if help_dialog is not None and help_dialog.winfo_exists():
            help_dialog.lift()
            return
        help_dialog = tkinter.Toplevel(root)
        help_dialog.title("操作说明")
        help_dialog.transient(root)
        help_dialog.resizable(False, False)
        help_dialog.wm_attributes("-topmost", True)
        body = tkinter.Frame(help_dialog, bg="#20242b", padx=20, pady=20)
        body.pack(fill="both", expand=True)
        tkinter.Label(body, text="选行后点击下方发射\n自动朝向，角度手调\nss：指定力度　t：计算\n顶部拖动绑定，或 r\nEsc：暂停／中止发射\n“我”列：指定自己",
                      justify="left", wraplength=min(680, available_width),
                      bg="#20242b", fg="#c8ced8", font=("Microsoft YaHei", 10)).pack(anchor="w")
        # Keep this informational window modeless; it never blocks analysis.
        ttk.Button(body, text="关闭", command=help_dialog.destroy,
                   takefocus=False).pack(anchor="e", pady=(16, 0))
    help_button = ttk.Button(dock_actions, text="说明", command=show_help, takefocus=False, width=5)
    help_button.pack(side="left", padx=(4, 0))
    ttk.Button(dock_actions, text="检查焦点", command=probe_sidecar_focus,
               takefocus=False).pack(side="right")
    sidecar_status = tkinter.StringVar(master=root, value="独立窗口；连接后辅助贴在大厅右侧")
    focus_status = tkinter.StringVar(master=root, value="")
    focus_label = tkinter.Label(dockbar, textvariable=focus_status, bg="#20242b", fg="#ffcf65",
                               anchor="w", justify="left", wraplength=900)
    focus_label.pack(fill="x")
    independent_focus = tkinter.BooleanVar(master=root, value=False)
    ttk.Checkbutton(dockbar, text="独立保留焦点（鼠标操作）", variable=independent_focus,
                    command=toggle_sidecar, takefocus=False).pack(anchor="w", pady=(4, 0))
    content = tkinter.Frame(container, bg="#20242b")
    content.pack(fill="both", expand=True)
    content.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)
    viewport = tkinter.Canvas(content, bg="#20242b", highlightthickness=0)
    viewport.grid(row=0, column=0, sticky="nsew")
    vertical = ttk.Scrollbar(content, orient="vertical", command=viewport.yview)
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal = ttk.Scrollbar(content, orient="horizontal", command=viewport.xview)
    horizontal.grid(row=1, column=0, sticky="ew")
    viewport.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    outer = tkinter.Frame(viewport, bg="#20242b")
    body_window = viewport.create_window(0, 0, window=outer, anchor="nw")
    toolbar = tkinter.Frame(outer, bg="#20242b")
    toolbar.pack(fill="x")
    binding_bar = tkinter.Frame(toolbar, bg="#20242b")
    binding_bar.pack(fill="x")
    action_bar = tkinter.Frame(toolbar, bg="#20242b")
    action_bar.pack(fill="x", pady=(6, 0))
    mode = tkinter.StringVar(value=AUTO_MODE)
    selector = ttk.Combobox(binding_bar, textvariable=mode, values=(AUTO_MODE, MANUAL_MODE), state="readonly", width=16)
    selector.pack(side="left")
    selector.bind("<<ComboboxSelected>>", lambda _: select_mode())
    compact_mode = tkinter.Frame(binding_bar, bg="#20242b")
    for text, value in (("自动", AUTO_MODE), ("手动", MANUAL_MODE)):
        ttk.Radiobutton(compact_mode, text=text, value=value, variable=mode,
                        command=select_mode, takefocus=False).pack(side="left")
    mark = ttk.Button(action_bar, text="精细校准", command=start_region_selection)
    mark.pack(side="left")
    start = ttk.Button(action_bar, text="开始", command=toggle_analysis)
    start.pack(side="left", padx=8)
    refresh = ttk.Button(action_bar, text="计算一次 (t)", command=refresh_analysis)
    refresh.pack(side="left")
    calibration = tkinter.StringVar(value="绑定窗口：将顶部“拖动绑定”拖到游戏画面内松开；也可把鼠标放到游戏内按 r")
    calibration_label = tkinter.Label(outer, textvariable=calibration, bg="#20242b", fg="#50cfff", anchor="w", wraplength=900)
    calibration_label.pack(fill="x", pady=(10, 0))
    summary, status = tkinter.StringVar(value="等待识别"), tkinter.StringVar(value="请拖动顶部“拖动绑定”绑定游戏窗口，无需标记两个角")
    summary_label = tkinter.Label(outer, textvariable=summary, bg="#20242b", fg="white", anchor="w", wraplength=750)
    summary_label.pack(fill="x", pady=(12, 8))
    preview = tkinter.Frame(outer, bg="#20242b")
    preview.pack(fill="x")
    canvas = tkinter.Canvas(preview, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg="#15181f", highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nw")
    parameter_frame = tkinter.Frame(preview, bg="#20242b")
    parameter_frame.grid(row=0, column=1, sticky="nw", padx=(18, 0))
    parameters = tkinter.Canvas(parameter_frame, width=250, height=74, bg="#15181f", highlightthickness=0)
    parameters.pack(anchor="w")
    status_label = tkinter.Label(outer, textvariable=status, bg="#20242b", fg="#ffcf65", anchor="w", justify="left", wraplength=750)
    status_label.pack(fill="x", pady=8)
    table_frame = tkinter.Frame(outer)
    table_frame.pack(fill="both", expand=True)
    columns = ("我", "编号", "身份", "方向", "水平距离", "高低差", "建议力度", "状态")
    table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8, selectmode="browse")
    table_width = 0
    for column, width in zip(columns, (36, 48, 72, 45, 80, 70, 82, 250)):
        width = max(width, table_font.measure(column)+20)
        if column == "状态":
            width = max(width, table_font.measure("当前角度没有有效的正力度解")+20)
        table.heading(column, text=column)
        table.column(column, width=width, minwidth=width, anchor="center" if column != "状态" else "w", stretch=column == "状态")
        table_width += width
    table.bind("<Button-1>", select_own_player)
    table.bind("<<TreeviewSelect>>", update_shot_controls)
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    table.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    for identity, colour in (("自己", "#50cfff"), ("队友", "#70f898"), ("敌人", "#ff748d"), ("身份不确定", "#ffcf65")):
        table.tag_configure(identity, foreground=colour)
    layout = {"wide": None, "docked": False, "compact": None, "preview_size": None,
              "keep_focus": False, "button_mode": None}
    def fit_content(_=None):
        width = viewport.winfo_width()
        if width <= 1:
            width = 440 if layout["docked"] else min(1120, available_width)-24
        compact = layout["docked"] or width < 640
        button_mode = compact or layout["keep_focus"]
        if button_mode != layout["button_mode"]:
            layout["button_mode"] = button_mode
            if button_mode:
                selector.pack_forget()
                compact_mode.pack(side="left")
            else:
                compact_mode.pack_forget()
                selector.pack(side="left")
        if compact != layout["compact"]:
            layout["compact"] = compact
            table.configure(displaycolumns=("我", "编号", "身份", "方向", "建议力度") if compact else columns)
            for column, narrow, full in zip(columns[:4]+("建议力度",),
                                           (32, 44, 76, 42, 78), (36, 48, 72, 45, 82)):
                column_width = max(narrow if compact else full, table_font.measure(column)+16)
                table.column(column, width=column_width, minwidth=column_width)
        preview_width = max(120, min(PREVIEW_WIDTH, width)) if compact else PREVIEW_WIDTH
        preview_height = round(preview_width * PREVIEW_HEIGHT / PREVIEW_WIDTH)
        if compact:
            scale = float(root.tk.call("tk", "scaling"))*72/96
            preview_height = min(preview_height, round(120*scale))
        preview_size = preview_width, preview_height
        if preview_size != layout["preview_size"]:
            layout["preview_size"] = preview_size
            canvas.configure(width=preview_width, height=preview_height)
            if _ui.get("canvas") is canvas:
                draw_minimap(_displayed_result)
        wide = not compact and width >= PREVIEW_WIDTH+18+parameter_frame.winfo_reqwidth()
        if wide != layout["wide"]:
            layout["wide"] = wide
            parameter_frame.grid_configure(row=0 if wide else 1, column=1 if wide else 0,
                                           padx=(18, 0) if wide else 0, pady=0 if wide else (8, 0))
        minimum_width = (sum(table.column(c, "minwidth") for c in ("我", "编号", "身份", "方向", "建议力度"))+20
                         if compact else max(PREVIEW_WIDTH, table_width+20))
        body_width = max(width, minimum_width, toolbar.winfo_reqwidth())
        for label in (calibration_label, summary_label, status_label):
            label.configure(wraplength=body_width)
        body_height = max(viewport.winfo_height(), outer.winfo_reqheight())
        viewport.itemconfigure(body_window, width=body_width, height=body_height)
        viewport.configure(scrollregion=(0, 0, body_width, body_height))
        for label in (focus_label, details_label):
            label.configure(wraplength=max(100, container.winfo_width()))
        selected_label.configure(wraplength=max(100, container.winfo_width()-shoot.winfo_reqwidth()-12))

    def set_sidecar_layout(docked):
        layout["docked"] = bool(docked)
        root.minsize(300 if docked else min(760, available_width),
                     320 if docked else min(600, available_height))
        root.resizable(not docked, not docked)
        root.wm_attributes("-topmost", not docked)
        fit_content()

    def set_keep_focus_layout(enabled):
        layout["keep_focus"] = bool(enabled)
        fit_content()

    def native_wheel(delta, horizontal=False):
        # An inactive panel still receives mouse input. Keep wheel handling on
        # this Tk thread rather than transferring keyboard focus to the panel.
        units = -int(delta/120) or (-1 if delta > 0 else 1)
        (viewport.xview_scroll if horizontal else viewport.yview_scroll)(units, "units")
    viewport.bind("<Configure>", fit_content)
    outer.bind("<Configure>", fit_content)
    def scroll_content(event):
        if event.widget not in (table, text_widget):
            viewport.yview_scroll(-int(event.delta/120), "units")
            return "break"
    root.bind("<MouseWheel>", scroll_content)
    root.bind("<Configure>", lambda event: fit_content() if event.widget is root else None, add="+")
    root.update_idletasks()
    return {"mode": mode, "start": start, "refresh": refresh, "summary": summary,
            "status": status, "canvas": canvas, "table": table, "log": text_widget,
            "table_font": table_font, "parameters": parameters, "calibration": calibration, "mark": mark,
            "finder": finder, "shoot": shoot, "selected_target": selected_target,
            "viewport": viewport, "preview_parameters": parameter_frame, "help_button": help_button,
            "target_details": target_details, "details_label": details_label,
            "sidecar_enabled": sidecar_enabled, "sidecar_width": sidecar_width,
            "sidecar_status": sidecar_status, "focus_status": focus_status,
            "independent_focus": independent_focus,
            "set_keep_focus_layout": set_keep_focus_layout,
            "mode_selector": selector, "mode_buttons": compact_mode,
            "set_sidecar_layout": set_sidecar_layout, "native_wheel": native_wheel}


def run():
    global _game_config, _tk, _ui, _analysis_worker, _shot_controller, _sidecar

    setup_diagnostics()
    _tk = tkinter.Tk()
    _tk.report_callback_exception = report_callback_exception
    _ui = build_ui(_tk)
    _tk.protocol("WM_DELETE_WINDOW", on_destroy)
    setup_logger(_ui["log"])
    _shot_controller = ShotController(lambda: space_press(pause=False),lambda: space_release(pause=False),
                                     focus_shot_target,verify_shot_target,
                                     report_shot_status,
                                     _PRESS_DURATION_PER_FORCE,
                                     tap_direction=direction_tap, check_state=check_shot_state,
                                     on_state_changed=refresh_after_shot_change)
    setup_km(_km_queue)
    threading.Thread(target=km_listen_queue, daemon=True).start()

    config = load_config(_GAME_CONFIG_PATH)
    # Saved preferences are reusable; a game binding needs an explicit action
    # in this session before any screen capture or analysis can start.
    _game_config = dict(config or {}, region=(0, 0, 0, 0))
    _game_config.pop("window", None)
    settings = sidecar_settings(_game_config)
    _ui["sidecar_enabled"].set(settings["enabled"])
    _ui["independent_focus"].set(settings["independent_focus"])
    _ui["sidecar_width"].set(str(settings["width"]))
    try:
        _sidecar = SidecarController(WindowsPanel(_tk), _ui["set_sidecar_layout"],
                                     report_sidecar_status, invalidate_sidecar_binding,
                                     _ui["native_wheel"], _ui["set_keep_focus_layout"])
    except OSError as exc:
        _ui["sidecar_enabled"].set(False)
        _ui["independent_focus"].set(False)
        _ui["sidecar_status"].set(f"无法启用窗口联动：{exc}")

    analyzer = SnapshotAnalyzer(recognize_wind, recognize, recognize_ten_units)
    turn_detector = OwnTurnDetector()
    _analysis_worker = TurnAnalysisWorker(capture_game_frame, analyzer, turn_detector,
                                          input_reader=analyzer.probe, active_detector=turn_detector.active)
    _analysis_worker.configure(_game_config["region"], False)
    _analysis_worker.start()
    update_controls()
    _ui["status"].set("等待绑定游戏窗口")
    region_prompt("未绑定游戏窗口，请将顶部“拖动绑定”拖到游戏内；或在游戏内按 r")
    if _sidecar is not None:
        _sidecar.set_enabled(settings["enabled"], _game_config)
    _tk.after(100, poll_ui)

    logger.info(f"DSS 初始化完毕!{'（配置已加载）' if config else ''}")
    _tk.mainloop()


def capture_game_frame(region):
    config = dict(_game_config)
    if tuple(config["region"]) == tuple(region):
        region = resolve_region(config)
    return _capture_region(region)


if __name__ == "__main__":
    run()
