"""Binding lifecycle checks without mouse injection, capture or game input."""

import tempfile
import tkinter
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import dump_config, load_config
from analysis import TurnAnalysisWorker
from sidecar_window import sidecar_settings
from test_auto_recognition import load_functions


class WindowBindingUITests(unittest.TestCase):
    def setUp(self):
        self.old_config = {"region": (20, 30, 1500, 900), "custom": "keep"}
        self.target = {"region": (100, 200, 2000, 1200),
                       "window": {"hwnd": 100, "content": {"hwnd": 200}}}
        self.state = load_functions("main.py", {
            "prepare_window_binding", "start_window_drag", "release_window_drag",
            "cancel_window_binding", "finish_window_binding", "finish_window_drag",
            "bind_game_under_mouse", "update_controls",
        }, {
            "_game_config": self.old_config, "_window_drag": None, "_region_overlay": None,
            "_mode": "auto", "AUTO_MODE": "auto", "_tk": Mock(), "tkinter": tkinter,
            "_analysis_worker": Mock(running=True), "_shot_controller": Mock(),
            "_fire_cancel": Mock(), "close_force_dialog": Mock(), "reset_inputs": Mock(),
            "_ui": {key: Mock() for key in ("finder", "status", "start", "refresh", "mark")},
            "_last_s_press": 1, "region_prompt": Mock(), "clear_results": Mock(),
            "bind_window_at": Mock(return_value=self.target),
            "resolve_region": Mock(side_effect=lambda config: config["region"]),
            "dump_config": Mock(), "_GAME_CONFIG_PATH": "unused.json", "logger": Mock(),
            "_ten_units_pixels": 80, "_enemy_pos": (1, 2, False), "_tmp_pos": (1, 2),
            "get_curr_mouse_pos": Mock(return_value=(500, 500)),
        })

    def test_drag_drop_binds_and_resumes_analysis_without_corner_selection(self):
        self.state["start_window_drag"]()
        self.state["_ui"]["finder"].grab_set_global.assert_called_once()
        self.state["_analysis_worker"].pause.assert_called_once()
        self.state["_shot_controller"].cancel.assert_called_once()
        self.state["finish_window_drag"](SimpleNamespace(x_root=500, y_root=600))
        self.state["bind_window_at"].assert_called_once_with((500, 600))
        config = self.state["_game_config"]
        self.assertEqual(config["region"], self.target["region"])
        self.assertEqual(config["custom"], "keep")
        self.state["dump_config"].assert_called_once_with(config, "unused.json")
        self.state["_analysis_worker"].configure.assert_called_once_with(config["region"], True)
        self.assertIsNone(self.state["_window_drag"])
        self.assertIsNone(self.state["_region_overlay"])
        self.assertEqual(self.state["_ten_units_pixels"], 0)
        self.assertIsNone(self.state["_enemy_pos"])
        self.state["_ui"]["finder"].grab_release.assert_called_once()

    def test_r_binds_under_mouse_without_grabbing_or_showing_overlay(self):
        self.state["bind_game_under_mouse"]()
        self.state["bind_window_at"].assert_called_once_with((500, 500))
        self.state["_ui"]["finder"].grab_set_global.assert_not_called()
        self.assertEqual(self.state["_game_config"]["region"], self.target["region"])

    def test_cancel_restores_previous_running_or_paused_state(self):
        for running in (True, False):
            self.state["_analysis_worker"].running = running
            self.state["start_window_drag"]()
            self.state["cancel_window_binding"]()
            self.state["cancel_window_binding"]()
            self.state["finish_window_drag"](SimpleNamespace(x_root=500, y_root=500))
            self.assertIs(self.state["_game_config"], self.old_config)
            self.state["_analysis_worker"].configure.assert_called_with(self.old_config["region"], running)
        self.state["dump_config"].assert_not_called()
        self.state["bind_window_at"].assert_not_called()

    def test_failed_lookup_validation_or_save_preserves_binding_and_releases_grab(self):
        for operation in ("bind_window_at", "resolve_region", "dump_config"):
            with self.subTest(operation=operation):
                self.state[operation].side_effect = OSError("失败")
                self.state["start_window_drag"]()
                self.state["finish_window_binding"]((500, 500))
                self.assertIs(self.state["_game_config"], self.old_config)
                self.assertIsNone(self.state["_window_drag"])
                self.state["_analysis_worker"].configure.assert_called_with(self.old_config["region"], True)
                self.assertIn("绑定失败", self.state["region_prompt"].call_args.args[0])
                self.state[operation].side_effect = (lambda config: config["region"]) if operation == "resolve_region" else None

    def test_grab_failure_recovers_and_repeated_start_does_not_replace_prior_state(self):
        self.state["_ui"]["finder"].grab_set_global.side_effect = tkinter.TclError("grab failed")
        self.state["start_window_drag"]()
        self.assertIsNone(self.state["_window_drag"])
        self.state["_ui"]["finder"].grab_set_global.side_effect = None
        self.state["start_window_drag"]()
        previous = self.state["_window_drag"]
        self.state["start_window_drag"]()
        self.assertIs(self.state["_window_drag"], previous)

    def test_mode_change_cancel_does_not_restart_worker(self):
        self.state["start_window_drag"]()
        self.state["cancel_window_binding"](resume=False)
        self.state["_analysis_worker"].configure.assert_not_called()
        self.assertIsNone(self.state["_window_drag"])

    def test_escape_before_mouse_release_cancels_without_waiting_for_ui_poll(self):
        self.state["start_window_drag"]()
        self.state["_window_drag"]["cancelled"] = True
        self.state["finish_window_binding"]((500, 500))
        self.assertIs(self.state["_game_config"], self.old_config)
        self.state["bind_window_at"].assert_not_called()
        self.state["dump_config"].assert_not_called()


class ConfigSaveTests(unittest.TestCase):
    def test_failed_replace_preserves_original_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"game_config.json"
            dump_config({"region": [1, 2, 3, 4]}, str(path))
            with patch("config.os.replace", side_effect=OSError("cannot replace")):
                with self.assertRaises(OSError):
                    dump_config({"region": [5, 6, 7, 8]}, str(path))
            self.assertEqual(load_config(str(path)), {"region": [1, 2, 3, 4]})
            self.assertEqual(list(Path(directory).iterdir()), [path])


class StartupBindingTests(unittest.TestCase):
    def test_startup_requires_new_binding_without_capturing_saved_windows_or_regions(self):
        for saved in (None, {"region": [20, 30, 1500, 900]},
                      {"region": [20, 30, 1500, 900], "window": {"hwnd": 123},
                       "custom": "keep", "sidecar": {"width": 640, "enabled": True,
                                                       "independent_focus": True}}):
            with self.subTest(saved=saved):
                before = dict(saved) if saved else None
                ui = {key: Mock() for key in ("log", "sidecar_enabled", "sidecar_width",
                       "independent_focus", "status", "sidecar_status", "set_sidecar_layout",
                       "set_keep_focus_layout", "native_wheel")}
                capture = Mock(side_effect=AssertionError("startup must not capture"))
                names = ("setup_diagnostics", "report_callback_exception", "on_destroy", "setup_logger",
                         "ShotController", "space_press", "space_release", "focus_shot_target",
                         "verify_shot_target", "report_shot_status", "direction_press", "direction_release",
                         "direction_tap", "check_shot_state", "refresh_after_shot_change",
                         "setup_km", "km_listen_queue", "WindowsPanel", "SidecarController",
                         "report_sidecar_status", "invalidate_sidecar_binding", "recognize_wind",
                         "recognize", "recognize_ten_units", "OwnTurnDetector", "SnapshotAnalyzer",
                         "update_controls", "region_prompt", "poll_ui")
                namespace = {name: Mock() for name in names}
                namespace.update({
                    "tkinter": SimpleNamespace(Tk=Mock(return_value=Mock())),
                    "threading": SimpleNamespace(Thread=Mock()), "_km_queue": None,
                    "_GAME_CONFIG_PATH": "unused.json", "_PRESS_DURATION_PER_FORCE": .04,
                    "_game_config": {"region": (10, 20, 1500, 900), "window": {"hwnd": 999}},
                    "build_ui": Mock(return_value=ui), "load_config": Mock(return_value=saved),
                    "sidecar_settings": sidecar_settings, "logger": Mock(),
                    "TurnAnalysisWorker": TurnAnalysisWorker, "capture_game_frame": capture,
                })
                state = load_functions("main.py", {"run"}, namespace)
                state["run"]()
                worker = state["_analysis_worker"]
                try:
                    self.assertFalse(worker.running)
                    self.assertIsNone(worker.take_result())
                    capture.assert_not_called()
                    self.assertNotIn("window", state["_game_config"])
                    self.assertEqual(state["_game_config"]["region"], (0, 0, 0, 0))
                    ui["status"].set.assert_called_with("等待绑定游戏窗口")
                    if saved and "sidecar" in saved:
                        ui["sidecar_width"].set.assert_called_with("640")
                        ui["independent_focus"].set.assert_called_with(True)
                        self.assertEqual(state["_game_config"]["custom"], "keep")
                    self.assertEqual(saved, before)
                finally:
                    worker.close()
                    worker._thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
