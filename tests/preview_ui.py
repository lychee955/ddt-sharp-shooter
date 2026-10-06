"""Preview the actual UI against the supplied sample; never capture or fire."""

import argparse
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from analysis import AnalysisWorker, analyze_frame, WIND_REGION, DEGREE_REGION
from ocr import recognize, recognize_ten_units, recognize_wind


def sample_frame():
    fixture = Path(__file__).parent / "fixtures"
    frame = np.zeros((900, 1500, 3), dtype=np.uint8)
    frame[45:185, 1230:1500] = np.array(Image.open(fixture / "minimap.png").convert("RGB").resize((270,140), Image.Resampling.BILINEAR))
    for name, region in (("wind.png",WIND_REGION),("degree.png",DEGREE_REGION)):
        x,y,w,h = region
        frame[y:y+h,x:x+w] = np.array(Image.open(fixture/name).convert("RGB").resize((w,h), Image.Resampling.BILINEAR))
    return frame


def run(seconds=0, compact=False, sidecar=False, independent_focus=False):
    frame = sample_frame()
    analyze = lambda image: analyze_frame(image,recognize_wind,recognize,recognize_ten_units)
    main._tk = main.tkinter.Tk()
    main._ui = main.build_ui(main._tk)
    main._tk.title("DSS · 测试预览（示例数据）")
    main.setup_logger(main._ui["log"])
    main._game_config = {"region": (0,0,1500,900)}
    if sidecar or independent_focus:
        from sidecar_window import WindowsPanel, SidecarController
        main._game_config = main.load_config(main._GAME_CONFIG_PATH) or main._game_config
        saved = dict(main._game_config.get("sidecar", {}))
        saved["independent_focus"] = independent_focus
        main._game_config = dict(main._game_config, sidecar=saved)
        # Preview controls must never overwrite the user's game configuration.
        main.dump_config = lambda *_: None
        main._sidecar = SidecarController(WindowsPanel(main._tk), main._ui["set_sidecar_layout"],
                                          main._ui["sidecar_status"].set,
                                          wheel_callback=main._ui["native_wheel"],
                                          on_focus_layout=main._ui["set_keep_focus_layout"])
    main._analysis_worker = AnalysisWorker(lambda _: frame.copy(), analyze)
    main._analysis_worker.configure(main._game_config["region"], True)
    main._analysis_worker.start()
    main.update_controls()
    if compact:
        main._ui["set_sidecar_layout"](True)
        scale = float(main._tk.tk.call("tk", "scaling"))*72/96
        main._tk.geometry(f"{round(440*scale)}x{round(700*scale)}")
    if sidecar or independent_focus:
        main._ui["sidecar_enabled"].set(sidecar)
        main._ui["independent_focus"].set(independent_focus)
        main._sidecar.set_enabled(sidecar, main._game_config)
        previous_probe = {"value": None}
        def watch_focus():
            if main._stop_signal:
                return
            try:
                facts = main._sidecar.focus_probe(main._game_config)
                summary = (facts["hall_foreground"], facts["flash_focus"],
                           facts["cursor_root"] == main._sidecar.panel.hwnd,
                           main._sidecar.linked)
                if summary != previous_probe["value"]:
                    print(f"{time.monotonic():.1f} foreground={summary[0]} flash_focus={summary[1]} "
                          f"cursor_on_panel={summary[2]} linked={summary[3]}", flush=True)
                    previous_probe["value"] = summary
            except (ValueError, OSError) as exc:
                print(f"focus probe: {exc}", flush=True)
            main._tk.after(500, watch_focus)
        main._tk.after(500, watch_focus)
    def close():
        main._stop_signal = True
        main._analysis_worker.close()
        if main._sidecar is not None:
            main._sidecar.close()
        main._tk.destroy()
    main._tk.protocol("WM_DELETE_WINDOW",close)
    if seconds:
        main._tk.after(round(seconds*1000), close)
    main._tk.after(100,main.poll_ui)
    main._tk.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds",type=float,default=0)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--sidecar", action="store_true")
    parser.add_argument("--independent-focus", action="store_true")
    args = parser.parse_args()
    run(args.seconds, args.compact, args.sidecar, args.independent_focus)
