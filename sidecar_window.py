"""External, reversible docking. All native writes target this process's panel."""

import ctypes
from ctypes import wintypes
import logging
import os

from window_region import WindowsClients, resolve_region


DEFAULT_WIDTH = 440
MIN_WIDTH, MAX_WIDTH = 360, 640
KEEP_LIT_NOTE = "常亮未验证：贴边与保留焦点不能保证消除游戏灰幕"
VERIFIED_KEEP_LIT_NOTE = "保持焦点已启用；36 大厅实测侧栏操作可保持亮色"


class BindingInvalid(ValueError):
    pass


def sidecar_settings(config):
    saved = config.get("sidecar", {})
    if not isinstance(saved, dict):
        saved = {}
    width = saved.get("width", DEFAULT_WIDTH)
    if isinstance(width, bool) or not isinstance(width, int):
        width = DEFAULT_WIDTH
    return {"enabled": saved.get("enabled") is True,
            "independent_focus": saved.get("independent_focus") is True,
            "width": min(MAX_WIDTH, max(MIN_WIDTH, width))}


def dock_rectangle(bounds, work_area, width, min_height=320):
    """Coordinates share the caller's DPI context; negative monitor origins work."""
    left, top, right, bottom = bounds
    wl, wt, wr, wb = work_area
    if width <= 0 or right <= left or bottom-top < min_height:
        raise ValueError("大厅高度不足，请手动调整大厅后重试连接")
    if right < wl or right+width > wr or top < wt or bottom > wb:
        raise ValueError("右侧空间不足，请手动移动或缩小大厅后连接；辅助不会覆盖游戏")
    return right, top, width, bottom-top


def same_identity(state, saved):
    return state is not None and all(state.get(k) == saved.get(k)
                                     for k in ("hwnd", "pid", "class_name"))


class WindowsPanel:
    """Manage the Tk wrapper only, without reparenting or changing the hall."""

    NOACTIVATE = 0x08000000
    WM_MOUSEACTIVATE, WM_NCDESTROY = 0x21, 0x82
    WM_MOUSEWHEEL, WM_MOUSEHWHEEL = 0x20A, 0x20E
    MA_NOACTIVATE = 3
    GWLP_WNDPROC, GWL_EXSTYLE = -4, -20
    SWP_NOACTIVATE, SWP_NOOWNERZORDER = 0x10, 0x200

    def __init__(self, root, clients=None):
        self.root = root
        self.clients = clients or WindowsClients()
        self.api = self.clients.api
        self.hwnd = 0
        self._old_proc = self._callback = None
        self._added_style = 0
        self._wheel_callback = None
        self._active_calls = None
        self._retired_callbacks = []
        pointer = ctypes.c_ssize_t
        suffix = "PtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "W"
        self.get_long = getattr(self.api, "GetWindowLong"+suffix)
        self.set_long = getattr(self.api, "SetWindowLong"+suffix)
        self.get_long.argtypes, self.get_long.restype = [wintypes.HWND, ctypes.c_int], pointer
        self.set_long.argtypes, self.set_long.restype = [wintypes.HWND, ctypes.c_int, pointer], pointer
        self.api.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                           wintypes.WPARAM, wintypes.LPARAM]
        self.api.CallWindowProcW.restype = pointer
        self.api.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self.api.SetWindowPos.restype = wintypes.BOOL
        self.api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.api.ShowWindow.restype = wintypes.BOOL
        self.proc_type = ctypes.WINFUNCTYPE(pointer, wintypes.HWND, wintypes.UINT,
                                           wintypes.WPARAM, wintypes.LPARAM)
        class ProcedurePointer(ctypes.Union):
            _fields_ = [("function", self.proc_type), ("address", ctypes.c_void_p)]
        self.pointer_type = ProcedurePointer

    def _handle(self):
        self.root.update_idletasks()
        hwnd = self.clients.root(self.root.winfo_id())
        state = self.clients.window_state(hwnd)
        if not state or state["pid"] != os.getpid():
            raise OSError("无法找到辅助自身窗口")
        return hwnd

    def _write_long(self, index, value):
        ctypes.set_last_error(0)
        old = self.set_long(self.hwnd, index, value)
        if old == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return old

    def enable(self, wheel_callback=None):
        if self._callback is not None:
            return
        self.hwnd = self._handle()
        self._wheel_callback = wheel_callback
        style = self.get_long(self.hwnd, self.GWL_EXSTYLE)
        self._added_style = self.NOACTIVATE & ~style
        self._write_long(self.GWL_EXSTYLE, style | self.NOACTIVATE)
        self._old_proc = self.get_long(self.hwnd, self.GWLP_WNDPROC)
        if not self._old_proc:
            self.disable()
            raise OSError("无法读取辅助窗口消息处理器")

        active_calls = self._active_calls = [0]
        original_proc = self._old_proc
        def procedure(hwnd, message, wparam, lparam):
            active_calls[0] += 1
            try:
                if message == self.WM_MOUSEACTIVATE:
                    return self.MA_NOACTIVATE
                if message in (self.WM_MOUSEWHEEL, self.WM_MOUSEHWHEEL) and self._wheel_callback:
                    try:
                        delta = ctypes.c_short((wparam >> 16) & 0xffff).value
                        self._wheel_callback(delta, message == self.WM_MOUSEHWHEEL)
                        return 0
                    except Exception:
                        logging.getLogger(__name__).exception("侧栏滚动失败")
                return self.api.CallWindowProcW(original_proc, hwnd, message, wparam, lparam)
            finally:
                active_calls[0] -= 1

        self._callback = self.proc_type(procedure)
        try:
            self._write_long(self.GWLP_WNDPROC,
                             self.pointer_type(function=self._callback).address)
        except Exception:
            self._callback = None
            self.disable()
            raise

    def disable(self):
        if self.hwnd and self.api.IsWindow(self.hwnd):
            if self._callback is not None:
                self._write_long(self.GWLP_WNDPROC, self._old_proc)
            if self._added_style:
                style = self.get_long(self.hwnd, self.GWL_EXSTYLE)
                self._write_long(self.GWL_EXSTYLE, style & ~self._added_style)
        if self._callback is not None and self._active_calls[0]:
            # A close/detach button can run inside this very native callback.
            # Keep its C trampoline alive until a later main-thread poll.
            self._retired_callbacks.append((self._callback, self._active_calls))
        self._callback = self._old_proc = None
        self._active_calls = None
        self._added_style = 0
        self._wheel_callback = None
        self.release_callbacks()

    def release_callbacks(self):
        self._retired_callbacks[:] = [(callback, active) for callback, active
                                     in self._retired_callbacks if active[0]]

    def snapshot(self):
        hwnd = self._handle()
        return self.clients.window_state(hwnd)["bounds"]

    def width_pixels(self, width):
        # Use the panel's own coordinate space, also used by GetWindowRect.
        # Tk scaling is pixels per typographic point (72 points per inch).
        scale = float(self.root.tk.call("tk", "scaling")) * 72 / 96
        return round(width * scale)

    def place(self, rectangle):
        x, y, width, height = rectangle
        # Tk remembers a requested client size. Updating only the native outer
        # rectangle lets its next geometry pass restore the old wide window.
        bounds = self.clients.window_state(self.hwnd)["bounds"]
        client = wintypes.RECT()
        if not self.api.GetClientRect(self.hwnd, ctypes.byref(client)):
            raise OSError("无法读取辅助客户区")
        client_width = max(1, width - (bounds[2]-bounds[0]-client.right))
        client_height = max(1, height - (bounds[3]-bounds[1]-client.bottom))
        self.root.geometry(f"{client_width}x{client_height}")
        self.root.update_idletasks()
        if not self.api.SetWindowPos(self.hwnd, None, x, y, width, height,
                                    self.SWP_NOACTIVATE | self.SWP_NOOWNERZORDER | 0x4):
            raise ctypes.WinError(ctypes.get_last_error())

    def needs_place(self, rectangle):
        state = self.clients.window_state(self.hwnd)
        if state is None:
            raise OSError("辅助窗口已关闭")
        left, top, right, bottom = state["bounds"]
        return (left, top, right-left, bottom-top) != rectangle or state["minimized"]

    def show(self):
        # Show without activation: Tk deiconify can otherwise activate the hall's
        # companion on restoration. Tk observes the native mapping messages.
        self.api.ShowWindow(self.hwnd, 4)  # SW_SHOWNOACTIVATE

    def follow_foreground(self, hall_hwnd):
        foreground = self.clients.foreground()
        if foreground == hall_hwnd:
            if not self.api.SetWindowPos(self.hwnd, None, 0, 0, 0, 0,
                                        self.SWP_NOACTIVATE | self.SWP_NOOWNERZORDER | 0x1 | 0x2):
                raise ctypes.WinError(ctypes.get_last_error())

    def hide(self):
        self.api.ShowWindow(self.hwnd, 0)  # SW_HIDE

    def restore(self, bounds):
        # Changing Tk resizability can recreate its native wrapper.
        self.hwnd = self._handle()
        left, top, right, bottom = bounds
        self.place((left, top, right-left, bottom-top))
        self.show()


class SidecarController:
    """Poll from Tk's main thread; identity is checked before every placement."""

    def __init__(self, panel, on_layout, on_status, on_invalid=None, wheel_callback=None,
                 on_focus_layout=None):
        self.panel, self.clients = panel, panel.clients
        self.on_layout, self.on_status = on_layout, on_status
        self.on_invalid = on_invalid or (lambda: None)
        self.wheel_callback = wheel_callback
        self.on_focus_layout = on_focus_layout or (lambda _: None)
        self.independent_focus = False
        self._focus_rejected = False
        self.enabled = self.linked = False
        self.width = DEFAULT_WIDTH
        self._target = self._rejected = None
        self._saved_bounds = self._last_rectangle = None
        self._hidden = False
        self._status = None
        self._last_foreground = None

    def status(self, message):
        if message != self._status:
            self._status = message
            self.on_status(message)

    @staticmethod
    def fingerprint(config):
        binding = config.get("window") or {}
        content = binding.get("content") or {}
        return tuple(binding.get(k) for k in ("hwnd", "pid", "class_name")) + tuple(
            content.get(k) for k in ("hwnd", "pid", "class_name"))

    def set_enabled(self, enabled, config):
        self.enabled = bool(enabled)
        self.width = sidecar_settings(config)["width"]
        self._rejected = None
        self._focus_rejected = False
        if not enabled:
            self.detach()
        self.poll(config)

    def poll_independent_focus(self, config):
        requested = sidecar_settings(config)["independent_focus"]
        if self._focus_rejected:
            return
        try:
            if requested != self.independent_focus:
                self.on_focus_layout(requested)
                if requested:
                    self.panel.enable(self.wheel_callback)
                else:
                    self.panel.disable()
                self.independent_focus = requested
            self.status("独立窗口 · 鼠标操作保留前台焦点；请先点击游戏一次"
                        if requested else "独立窗口；连接后辅助贴在大厅右侧")
        except OSError as exc:
            self.panel.disable()
            self.independent_focus = False
            self.on_focus_layout(False)
            self._focus_rejected = True
            self.status(f"独立窗口保留焦点未启用：{exc}")

    def detach(self):
        if self.linked or self._saved_bounds is not None:
            # Restore the native procedure before the layout can recreate a Tk
            # wrapper or destroy this root. Never touch the external window.
            self.panel.disable()
            self.on_layout(False)
            if self._saved_bounds is not None:
                self.panel.restore(self._saved_bounds)
        self.linked = self._hidden = False
        self._saved_bounds = self._last_rectangle = self._target = None
        self._last_foreground = None

    def poll(self, config):
        self.panel.release_callbacks()
        if not self.enabled:
            self.poll_independent_focus(config)
            return
        if self.independent_focus:
            self.panel.disable()
            self.independent_focus = False
            self.on_focus_layout(False)
        fingerprint = self.fingerprint(config)
        if fingerprint == self._rejected:
            return
        if self.linked and fingerprint != self._target:
            self.detach()
        binding = config.get("window") or {}
        try:
            if not binding.get("content"):
                raise ValueError("请先拖动图标绑定大厅中的 Flash 游戏，再连接侧栏")
            hall = self.clients.window_state(binding["hwnd"])
            if not same_identity(hall, binding):
                raise BindingInvalid("大厅已关闭或身份改变，请重新绑定")
            saved = binding["content"]
            content = self.clients.window_state(saved["hwnd"])
            if not same_identity(content, saved) or self.clients.root(saved["hwnd"]) != binding["hwnd"]:
                raise BindingInvalid("Flash 游戏已重载或关闭，请重新绑定")
            if hall["minimized"] or not hall["visible"]:
                if self.linked and not self._hidden:
                    self.panel.hide()
                    self._hidden = True
                self.status("大厅已最小化或隐藏，恢复大厅后侧栏自动跟随")
                return
            try:
                resolve_region(config, self.clients)
            except ValueError as exc:
                raise BindingInvalid(str(exc)) from exc
            self.width = sidecar_settings(config)["width"]
            try:
                rectangle = dock_rectangle(hall["bounds"], self.clients.work_area(binding["hwnd"]),
                                           self.panel.width_pixels(self.width),
                                           self.panel.width_pixels(320))
            except ValueError as exc:
                self.detach()
                self.status(str(exc))
                return  # Automatically retry when the user makes space.
            if not self.linked:
                self._saved_bounds = self.panel.snapshot()
                self.on_layout(True)
                self.panel.enable(self.wheel_callback)
                self.linked = True
                self._target = fingerprint
            if rectangle != self._last_rectangle or self.panel.needs_place(rectangle):
                self.panel.place(rectangle)
                self._last_rectangle = rectangle
            if self._hidden or not self.panel.clients.window_state(self.panel.hwnd)["visible"]:
                self.panel.show()
                self._hidden = False
                self._last_foreground = None
            foreground = self.clients.foreground()
            if foreground != self._last_foreground:
                self.panel.follow_foreground(binding["hwnd"])
                self._last_foreground = foreground
            note = (VERIFIED_KEEP_LIT_NOTE if binding.get("class_name") == "36JBCOM_Browser"
                    else KEEP_LIT_NOTE)
            self.status("已连接右侧 · 鼠标操作保留游戏焦点\n" + note)
        except (ValueError, OSError) as exc:
            self.detach()
            self._rejected = fingerprint
            if isinstance(exc, BindingInvalid):
                self.on_invalid()
            self.status(f"侧栏已停止联动：{exc}")

    def focus_probe(self, config):
        """Return facts, never a fabricated 'keep lit supported' conclusion."""
        binding = config.get("window") or {}
        resolve_region(config, self.clients)
        if not binding.get("content"):
            raise ValueError("请先绑定 Flash 游戏窗口")
        foreground = self.clients.foreground()
        focus = self.clients.keyboard_focus(binding["content"]["hwnd"])
        cursor = self.clients.cursor_position()
        point_window = self.clients.window_at(cursor)
        return {"hall_foreground": foreground == binding["hwnd"],
                "flash_focus": self.clients.has_keyboard_focus(binding["content"]["hwnd"]),
                "foreground": foreground, "focus": focus,
                "cursor_window": point_window,
                "cursor_root": self.clients.root(point_window) if point_window else 0,
                "cursor": cursor, "linked": self.linked}

    def close(self):
        self.enabled = False
        self.independent_focus = False
        # No restoration/mapping on exit; only restore our original procedure.
        self.panel.disable()
