"""Preview the actual UI against the supplied sample; never capture or fire."""

import argparse
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


def run(seconds=0):
    frame = sample_frame()
    analyze = lambda image: analyze_frame(image,recognize_wind,recognize,recognize_ten_units)
    main._tk = main.tkinter.Tk()
    main._ui = main.build_ui(main._tk)
    main._tk.title("DSS · 截图预览（不连接游戏）")
    main.setup_logger(main._ui["log"])
    main._game_config = {"region": (0,0,1500,900)}
    main._analysis_worker = AnalysisWorker(lambda _: frame.copy(), analyze)
    main._analysis_worker.configure(main._game_config["region"], True)
    main._analysis_worker.start()
    main.update_controls()
    def close():
        main._stop_signal = True
        main._analysis_worker.close()
        main._tk.destroy()
    main._tk.protocol("WM_DELETE_WINDOW",close)
    if seconds:
        main._tk.after(round(seconds*1000), close)
    main._tk.after(100,main.poll_ui)
    main._tk.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds",type=float,default=0)
    run(parser.parse_args().seconds)
