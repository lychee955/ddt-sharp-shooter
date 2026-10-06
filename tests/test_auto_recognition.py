"""Regression checks without screen capture, OCR model loading or keyboard input."""

import ast
import math
import tkinter
from queue import Empty, Queue
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class AutoRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.state = load_functions("main.py", {
            "_screen_region", "recognize_and_fire", "_get_curr_force",
            "handle_inputs", "reset_inputs", "fire",
            "record_region_corner", "region_prompt", "close_region_overlay",
            "cancel_region_selection", "start_region_selection",
            "capture_game_frame",
            "poll_ui",
            "poll_ui_events",
            "close_force_dialog",
        }, {
            "math": math,
            "tkinter": tkinter,
            "Empty": Empty,
            "_game_config": {"region": (443, 438, 2002, 1208)},
            "_REF_GAME_REGION_WIDTH": 1500,
            "_MINIMAP_REGION": (1170, 35, 330, 150),
            "_WIND_REGION": (709, 22, 84, 54),
            "_DEG_REGION": (43, 835, 64, 36),
            "_FORCE_REGION": (225, 860, 745, 28),
            "_GAME_CONFIG_PATH": "unused.json",
            "_PRESS_DURATION_PER_FORCE": 0.04,
            "_ten_units_pixels": 0,
            "_enemy_pos": (120, 20, False),
            "_tmp_pos": None,
            "_cmd_flag": 2,
            "_cmd_typing": "",
            "logger": Mock(),
            "time": Mock(),
            "recognize_ten_units": Mock(return_value=100),
            "recognize_wind": Mock(return_value=(2.5, False)),
            "recognize": Mock(return_value="65"),
            "recognize_force": Mock(return_value=30),
            "calc_force": Mock(return_value=30),
            "space_press": Mock(),
            "space_release": Mock(),
            "dump_config": Mock(),
            "AUTO_MODE": "auto",
            "MANUAL_MODE": "manual",
            "_mode": "manual",
            "_analysis_worker": None,
            "_shot_controller": None,
            "_force_dialog": None,
            "_force_dialog_pending": False,
            "_last_s_press": None,
            "_dialog_escape_until": 0,
            "_ui_actions": Mock(),
            "_fire_cancel": Mock(wait=Mock(return_value=False)),
            "_region_corner": None,
            "_region_overlay": None,
            "_window_drag": None,
            "_region_canvas": None,
            "_ui": {"calibration": Mock(), "status": Mock()},
            "clear_results": Mock(),
            "update_controls": Mock(),
            "bind_region": Mock(return_value=None),
            "resolve_region": Mock(side_effect=lambda config: config["region"]),
            "_capture_region": Mock(),
        })
        self.real_fire = self.state["fire"]
        self.state["fire"] = Mock()

    def test_scaled_minimap_keeps_absolute_screen_offset(self):
        region = self.state["_screen_region"](self.state["_MINIMAP_REGION"])
        self.assertEqual(region, (2004, 484, 440, 200))

    def test_reference_size_keeps_offsets_unchanged(self):
        self.state["_game_config"]["region"] = (443, 438, 1500, 900)
        self.assertEqual(self.state["_screen_region"](self.state["_MINIMAP_REGION"]),
                         (1613, 473, 330, 150))

    def test_invalid_game_region_does_not_capture(self):
        for region in ((0, 0, 0, 0), (1, 2, -100, 900), (1, 2, 1500, 0)):
            with self.subTest(region=region):
                self.state["_game_config"]["region"] = region
                with self.assertRaisesRegex(ValueError, "游戏区域无效"):
                    self.state["recognize_and_fire"]()
        self.state["recognize_ten_units"].assert_not_called()

    def test_success_reads_correct_regions_and_calculates_distance(self):
        self.state["recognize_and_fire"]()
        self.state["recognize_ten_units"].assert_called_once_with((2004, 484, 440, 200))
        self.state["recognize_wind"].assert_called_once_with((1389, 467, 112, 72))
        self.state["recognize"].assert_called_once_with((500, 1552, 85, 48))
        self.state["calc_force"].assert_called_once_with(65, 2.5, 12, 2)
        self.state["fire"].assert_called_once_with(30)

    def test_missing_scale_cannot_divide_by_zero_or_fire(self):
        self.state["recognize_ten_units"].return_value = 0
        with self.assertRaisesRegex(ValueError, "小地图标尺识别失败"):
            self.state["recognize_and_fire"]()
        self.state["fire"].assert_not_called()
        self.assertEqual(self.state["_ten_units_pixels"], 0)

    def test_invalid_angle_does_not_calculate_or_fire(self):
        for value in ("", "abc", "999"):
            with self.subTest(value=value):
                self.state["recognize"].return_value = value
                with self.assertRaisesRegex(ValueError, "角度识别失败"):
                    self.state["recognize_and_fire"]()
        self.state["calc_force"].assert_not_called()
        self.state["fire"].assert_not_called()

    def test_invalid_calculated_force_does_not_fire(self):
        for force in (float("nan"), float("inf"), -1, 0, 101):
            with self.subTest(force=force):
                self.state["calc_force"].return_value = force
                with self.assertRaisesRegex(ValueError, "发射力度无效"):
                    self.state["recognize_and_fire"]()
        self.state["fire"].assert_not_called()

    def test_failure_logs_specific_reason_and_returns_to_ready(self):
        self.state["recognize_ten_units"].return_value = 0
        self.state["handle_inputs"]("t")
        self.state["logger"].exception.assert_called_once_with("自动识别或发射失败")
        messages = [call.args[0] for call in self.state["logger"].info.call_args_list]
        self.assertTrue(any("失败原因：ValueError: 小地图标尺识别失败" in msg for msg in messages))
        self.assertIsNone(self.state["_enemy_pos"])
        self.assertEqual(self.state["_cmd_flag"], 2)

    def test_force_bar_uses_same_coordinate_transform(self):
        self.state["_get_curr_force"]()
        self.state["recognize_force"].assert_called_once_with((743, 1585, 994, 37))

    def test_new_region_discards_stale_scale_and_target(self):
        self.state["_region_corner"] = SimpleNamespace(x=100, y=100)
        self.state["_analysis_worker"] = Mock()
        self.state["_ten_units_pixels"] = 100
        self.state["record_region_corner"](SimpleNamespace(x=1600,y=1000))
        self.assertEqual(self.state["_game_config"]["region"], (100, 100, 1500, 900))
        self.assertEqual(self.state["_ten_units_pixels"], 0)
        self.assertIsNone(self.state["_enemy_pos"])

    def test_reversed_region_does_not_overwrite_config(self):
        self.state["_region_corner"] = SimpleNamespace(x=100, y=100)
        self.state["record_region_corner"](SimpleNamespace(x=50,y=50))
        self.state["dump_config"].assert_not_called()
        self.assertEqual(self.state["_game_config"]["region"], (443, 438, 2002, 1208))
        self.assertEqual(self.state["_region_corner"].x,100)
        self.assertIn("右下角位置无效",self.state["_ui"]["calibration"].set.call_args.args[0])

    def test_r_queues_binding_on_ui_thread_without_recording_corners(self):
        self.state["_mode"] = "auto"
        self.state["_analysis_worker"] = Mock()
        self.state["handle_inputs"]("r")
        self.state["_ui_actions"].put.assert_called_once_with(("mark_region",None))
        self.state["_analysis_worker"].pause.assert_not_called()
        self.assertIsNone(self.state["_region_corner"])
        self.state["dump_config"].assert_not_called()

    def test_first_corner_prompts_for_second_and_success_is_explicit(self):
        self.state["_mode"] = "auto"
        worker = self.state["_analysis_worker"] = Mock()
        overlay = self.state["_region_overlay"] = Mock()
        self.state["record_region_corner"](SimpleNamespace(x=100,y=200))
        self.state["_ui"]["calibration"].set.assert_called_with("左上角已标记，请点击游戏画面右下角")
        self.state["dump_config"].assert_not_called()
        self.state["record_region_corner"](SimpleNamespace(x=1600,y=1100))
        self.assertEqual(self.state["_game_config"]["region"],(100,200,1500,900))
        self.assertIn("游戏区域标记成功",self.state["_ui"]["calibration"].set.call_args.args[0])
        worker.configure.assert_called_once_with((100,200,1500,900),True)
        overlay.destroy.assert_called_once()
        self.assertIsNone(self.state["_region_corner"])
        self.state["space_press"].assert_not_called()

    def test_failed_save_keeps_original_config_and_second_corner_step(self):
        self.state["_region_corner"] = SimpleNamespace(x=100,y=100)
        self.state["dump_config"].side_effect = OSError("无法保存")
        self.state["record_region_corner"](SimpleNamespace(x=1600,y=1000))
        self.assertEqual(self.state["_game_config"]["region"],(443,438,2002,1208))
        self.assertIsNotNone(self.state["_region_corner"])
        self.assertIn("保存失败",self.state["_ui"]["calibration"].set.call_args.args[0])

    def test_selection_blocks_refresh_and_fire_keys(self):
        self.state["_region_overlay"] = Mock()
        self.state["_analysis_worker"] = Mock()
        for key in "ttl30tyyet":
            self.state["handle_inputs"](key)
        self.state["fire"].assert_not_called()
        self.state["_analysis_worker"].refresh.assert_not_called()

    def test_window_drag_blocks_shortcuts_and_escape_marks_cancel_before_ui_poll(self):
        drag = self.state["_window_drag"] = {"running": True}
        self.state["_analysis_worker"] = Mock()
        for key in "ssttl30ty":
            self.state["handle_inputs"](key)
        self.state["_ui_actions"].put.assert_not_called()
        self.state["fire"].assert_not_called()
        self.state["_analysis_worker"].refresh.assert_not_called()
        self.state["handle_inputs"]("esc")
        self.assertTrue(drag["cancelled"])
        self.state["_ui_actions"].put.assert_called_once_with(("cancel_binding", None))

    def test_cancel_discards_first_corner_and_keeps_original_region(self):
        self.state["_region_corner"] = SimpleNamespace(x=100,y=200)
        overlay = self.state["_region_overlay"] = Mock()
        self.state["cancel_region_selection"]()
        self.assertIsNone(self.state["_region_corner"])
        self.assertEqual(self.state["_game_config"]["region"],(443,438,2002,1208))
        self.state["dump_config"].assert_not_called()
        overlay.grab_release.assert_called_once()
        self.assertIn("已取消",self.state["_ui"]["calibration"].set.call_args.args[0])

    def test_escape_only_queues_ui_work_without_logging_or_worker_locks(self):
        self.state["_mode"] = "auto"
        self.state["_analysis_worker"] = Mock()
        self.state["_region_overlay"] = Mock()
        self.state["handle_inputs"]("esc")
        self.state["_ui_actions"].put.assert_called_once_with(("paused",None))
        self.state["_analysis_worker"].pause.assert_not_called()
        self.state["logger"].info.assert_not_called()
        self.state["_fire_cancel"].set.assert_called_once()

    def test_repeated_cancel_does_not_close_overlay_or_log_twice(self):
        overlay = self.state["_region_overlay"] = Mock()
        self.assertEqual(self.state["cancel_region_selection"](),"break")
        self.assertEqual(self.state["cancel_region_selection"](),"break")
        overlay.grab_release.assert_called_once()
        overlay.destroy.assert_called_once()
        self.state["logger"].info.assert_called_once()

    def test_overlay_already_destroyed_still_clears_selection_and_controls(self):
        overlay = self.state["_region_overlay"] = Mock()
        overlay.grab_release.side_effect = tkinter.TclError("already destroyed")
        overlay.destroy.side_effect = tkinter.TclError("already destroyed")
        self.state["_region_corner"] = SimpleNamespace(x=100,y=200)
        self.state["cancel_region_selection"]()
        self.assertIsNone(self.state["_region_overlay"])
        self.assertIsNone(self.state["_region_corner"])
        self.assertIn("已取消",self.state["_ui"]["calibration"].set.call_args.args[0])

    def test_local_and_global_escape_keep_ui_polling_and_allow_reopening(self):
        overlay = self.state["_region_overlay"] = Mock()
        worker = self.state["_analysis_worker"] = Mock()
        worker.take_result.return_value = None
        self.state["_mode"] = "auto"
        self.state["_ui_actions"] = Queue()
        self.state["_stop_signal"] = False
        self.state["expire_stale_result"] = Mock()
        root = self.state["_tk"] = Mock()
        self.state["cancel_region_selection"]()
        self.state["handle_inputs"]("esc")
        self.state["poll_ui"]()
        overlay.destroy.assert_called_once()
        self.assertIsNone(self.state["_region_overlay"])
        root.after.assert_called_once_with(100,self.state["poll_ui"])
        self.state["handle_inputs"]("r")
        self.assertEqual(self.state["_ui_actions"].get_nowait(),("mark_region",None))

    def test_success_saves_client_binding_and_capture_resolves_current_origin(self):
        self.state["_mode"] = "auto"
        self.state["_analysis_worker"] = Mock()
        self.state["_region_corner"] = SimpleNamespace(x=100,y=200)
        binding = {"hwnd":123,"offset":[10,20,1500,900]}
        self.state["bind_region"].return_value = binding
        self.state["record_region_corner"](SimpleNamespace(x=1600,y=1100))
        self.assertEqual(self.state["_game_config"]["window"],binding)
        self.assertIn("自动跟随",self.state["_ui"]["calibration"].set.call_args.args[0])
        self.state["resolve_region"].side_effect = None
        self.state["resolve_region"].return_value = (500,300,1500,900)
        self.state["capture_game_frame"]((100,200,1500,900))
        self.state["_capture_region"].assert_called_once_with((500,300,1500,900))

    def test_new_static_region_does_not_retain_previous_window_binding(self):
        self.state["_game_config"]["window"] = {"hwnd":123}
        self.state["_region_corner"] = SimpleNamespace(x=100,y=200)
        self.state["_analysis_worker"] = Mock()
        self.state["record_region_corner"](SimpleNamespace(x=1600,y=1100))
        self.assertNotIn("window",self.state["_game_config"])

    def test_repeated_r_keeps_recorded_first_corner(self):
        point = self.state["_region_corner"] = SimpleNamespace(x=100,y=200)
        overlay = self.state["_region_overlay"] = Mock()
        self.state["start_region_selection"]()
        self.assertIs(self.state["_region_corner"],point)
        overlay.lift.assert_called_once()

    def test_space_released_even_if_wait_fails(self):
        self.state["_fire_cancel"].wait.side_effect = [False, RuntimeError("wait failed")]
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            self.real_fire(30)
        self.state["space_press"].assert_called_once()
        self.state["space_release"].assert_called_once()

    def test_analysis_key_paths_never_fire(self):
        self.state["_mode"] = "auto"
        worker = self.state["_analysis_worker"] = Mock()
        for key in "ttl30tyyt":
            self.state["handle_inputs"](key)
        worker.refresh.assert_not_called()
        self.assertEqual(self.state["_ui_actions"].put.call_args_list,
                         [unittest.mock.call(("refresh", None))]*4)
        self.state["fire"].assert_not_called()
        self.state["space_press"].assert_not_called()

    def test_global_q_never_quits_in_analysis_manual_or_during_binding(self):
        for mode, drag, overlay, busy in (("auto", None, None, False),
                                         ("manual", None, None, False),
                                         ("auto", {"running": True}, None, False),
                                         ("auto", None, Mock(), False),
                                         ("auto", None, None, True)):
            with self.subTest(mode=mode, drag=drag, busy=busy):
                self.state.update(_mode=mode, _window_drag=drag, _region_overlay=overlay,
                                  _shot_controller=Mock(busy=busy))
                self.state["_ui_actions"].reset_mock()
                self.state["handle_inputs"]("q")
                self.state["_ui_actions"].put.assert_not_called()
                self.state["_shot_controller"].cancel.assert_not_called()

    def test_ui_poll_exception_is_logged_and_next_poll_is_scheduled(self):
        worker = self.state["_analysis_worker"] = Mock()
        worker.take_result.side_effect = RuntimeError("bad snapshot")
        self.state.update(_ui_actions=Queue(), _stop_signal=False, _mode="auto", _tk=Mock())
        self.state["poll_ui"]()
        self.state["logger"].exception.assert_called_once()
        self.state["_tk"].after.assert_called_once_with(100, self.state["poll_ui"])
        worker.take_result.side_effect = None
        worker.take_result.return_value = None
        self.state["expire_stale_result"] = Mock()
        self.state["poll_ui"]()
        self.assertEqual(self.state["_tk"].after.call_count, 2)

    def test_stopped_ui_poll_does_not_schedule_or_read_results(self):
        self.state.update(_stop_signal=True, _tk=Mock(), _analysis_worker=Mock())
        self.state["poll_ui"]()
        self.state["_tk"].after.assert_not_called()
        self.state["_analysis_worker"].take_result.assert_not_called()

    def test_fire_is_blocked_in_analysis_mode(self):
        self.state["_mode"] = "auto"
        with self.assertRaisesRegex(ValueError, "选中行点击发射"):
            self.real_fire(30)
        self.state["space_press"].assert_not_called()

    def test_mode_change_during_launch_delay_prevents_key_press(self):
        def change_mode(_):
            self.state["_mode"] = "auto"
            return False
        self.state["_fire_cancel"].wait.side_effect = change_mode
        self.real_fire(30)
        self.state["space_press"].assert_not_called()


class WindRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.state = load_functions("ocr.py", {"recognize_wind"}, {
            "ndarray": np.ndarray,
            "np": np,
            "_recognize_digit": Mock(),
            "_recognize_red_wind": Mock(return_value=None),
            "_binarize_image_by_reference": Mock(return_value=np.zeros((10, 20))),
            "_left_side_more_dark": Mock(return_value=False),
            "color": SimpleNamespace(rgb2gray=lambda image: np.zeros((10, 20))),
            "measure": SimpleNamespace(find_contours=Mock(return_value=[])),
        })
        self.image = np.zeros((10, 20, 3), dtype=np.uint8)

    def test_empty_digits_report_wind_failure(self):
        self.state["_recognize_digit"].return_value = ""
        with self.assertRaisesRegex(ValueError, "风力数字识别失败"):
            self.state["recognize_wind"](self.image)

    def test_long_ocr_text_is_not_interpreted_as_wind(self):
        self.state["_recognize_digit"].return_value = "5062"
        with self.assertRaisesRegex(ValueError, "风力数字识别失败"):
            self.state["recognize_wind"](self.image)

    def test_missing_integer_contour_reports_wind_failure(self):
        self.state["_recognize_digit"].return_value = "05"
        with self.assertRaisesRegex(ValueError, "风力整数位轮廓识别失败"):
            self.state["recognize_wind"](self.image)

    def test_missing_decimal_contour_reports_wind_failure(self):
        self.state["_recognize_digit"].return_value = "20"
        with self.assertRaisesRegex(ValueError, "风力小数位轮廓识别失败"):
            self.state["recognize_wind"](self.image)

    def test_valid_wind_retains_original_value(self):
        self.state["_recognize_digit"].return_value = "25"
        self.assertEqual(self.state["recognize_wind"](self.image), (2.5, False))


if __name__ == "__main__":
    unittest.main()
