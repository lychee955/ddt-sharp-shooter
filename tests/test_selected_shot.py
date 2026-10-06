"""Selected-row shots and responsive preview checks without game input."""

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from analysis import AnalysisResult, TargetEstimate
from vision import PlayerMarker
from turn import InputState


def snapshot():
    return AnalysisResult(targets=(
        TargetEstimate(PlayerMarker(20, 30, (0, 255, 0), "自己", "我"), status="自己"),
        TargetEstimate(PlayerMarker(80, 50, (255, 0, 0), "敌人", "敌1"),
                       "右", 3, 2, 78.934, "可用"),
    ), minimap=np.zeros((150, 330, 3), dtype=np.uint8), phase="complete", timestamp=1,
       degree=65, input_state=InputState(65, 20, 30),
       frame=np.zeros((1, 1, 3), dtype=np.uint8))


class SelectedShotTests(unittest.TestCase):
    def setUp(self):
        self.ui = {key: Mock() for key in ("table", "shoot", "selected_target", "status")}
        self.ui["table"].selection.return_value = ("1",)
        self.controller = Mock(busy=False)
        self.controller.submit.side_effect = self.submit
        self.target = {"window": {"hwnd": 123}}
        patcher = patch.multiple(main, _ui=self.ui, _mode=main.AUTO_MODE,
                                 _displayed_result=snapshot(), _shot_selection_ready=True,
                                 _shot_controller=self.controller, _window_drag=None, _region_overlay=None,
                                 _force_dialog=None, _force_dialog_pending=False,
                                 shot_target=Mock(return_value=self.target), reset_inputs=Mock())
        patcher.start()
        self.addCleanup(patcher.stop)

    def submit(self, *args, **kwargs):
        self.controller.busy = True
        return True

    def test_selected_button_shoots_displayed_decimal_and_direction_once(self):
        main.update_shot_controls()
        self.ui["shoot"].configure.assert_called_with(state="normal")
        self.ui["selected_target"].set.assert_called_with("敌1 · 向右 · 力度 78.93")
        main.fire_selected_target()
        main.fire_selected_target()
        self.controller.submit.assert_called_once_with(78.93, dict(self.target, shot_state=snapshot().input_state), direction="right")
        self.ui["shoot"].configure.assert_called_with(state="disabled")

    def test_valid_teammate_and_uncertain_targets_can_shoot_left(self):
        for identity in ("队友", "身份不确定"):
            with self.subTest(identity=identity):
                base = snapshot()
                target = replace(base.targets[1], player=replace(base.targets[1].player, identity=identity), direction="左")
                main._displayed_result = replace(base, targets=(base.targets[0], target))
                self.controller.busy = False
                main.fire_selected_target()
                self.controller.submit.assert_called_with(78.93, dict(self.target, shot_state=base.input_state), direction="left")

    def test_no_selection_self_and_outdated_indices_do_not_shoot(self):
        for rows in ((), ("0",), ("-1",), ("2",), ("1", "0"), ("invalid",)):
            with self.subTest(rows=rows):
                self.ui["table"].selection.return_value = rows
                main.update_shot_controls()
                main.fire_selected_target()
                self.ui["shoot"].configure.assert_called_with(state="disabled")
                self.controller.submit.assert_not_called()

    def test_failed_stale_computing_and_pending_results_cannot_shoot(self):
        for result in (None, replace(snapshot(), error="失败"), replace(snapshot(), stale_age=1),
                       replace(snapshot(), phase="computing"), replace(snapshot(), phase="failed")):
            with self.subTest(result=result):
                main._displayed_result = result
                main.fire_selected_target()
                self.controller.submit.assert_not_called()
        main._displayed_result = snapshot()
        main.clear_shot_selection()
        main.fire_selected_target()
        self.controller.submit.assert_not_called()

    def test_unusable_and_out_of_range_force_or_direction_cannot_shoot(self):
        base = snapshot()
        variants = [replace(base.targets[1], force=value)
                    for value in (None, 0, -1, 100.001, float("inf"), float("nan"), .001)]
        variants += [replace(base.targets[1], status="当前角度力度不足"),
                     replace(base.targets[1], direction="垂直")]
        for target in variants:
            with self.subTest(target=target):
                main._displayed_result = replace(base, targets=(base.targets[0], target))
                main.fire_selected_target()
                self.controller.submit.assert_not_called()

    def test_blocked_states_and_lost_window_disable_button(self):
        for field, value in (("_window_drag", {}), ("_region_overlay", Mock()),
                             ("_force_dialog", Mock()), ("_force_dialog_pending", True),
                             ("_mode", main.MANUAL_MODE)):
            with self.subTest(field=field), patch.object(main, field, value):
                main.update_shot_controls()
                main.fire_selected_target()
                self.ui["shoot"].configure.assert_called_with(state="disabled")
                self.controller.submit.assert_not_called()
        main.shot_target.side_effect = ValueError("游戏窗口已关闭")
        main.update_shot_controls()
        main.fire_selected_target()
        self.ui["shoot"].configure.assert_called_with(state="disabled")
        self.ui["status"].set.assert_called_with("游戏窗口已关闭")
        self.controller.submit.assert_not_called()

    def test_manual_refresh_clears_selection_and_paused_results_remain_usable(self):
        with patch.object(main, "_analysis_worker", Mock(running=False)) as worker:
            main.fire_selected_target()
            self.controller.submit.assert_called_once()
            self.controller.busy = False
            main.refresh_analysis()
            worker.refresh.assert_called_once()
            self.ui["table"].selection_remove.assert_called_with("1")
            main.fire_selected_target()
            self.controller.submit.assert_called_once()


class PreviewAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = main.tkinter.Tk()
        self.root.withdraw()
        self.ui = main.build_ui(self.root)
        self.root.wm_attributes("-topmost", False)
        patcher = patch.multiple(main, _ui=self.ui, _mode=main.AUTO_MODE,
                                 _displayed_result=None, _shot_selection_ready=False,
                                 _shot_controller=Mock(busy=False), _analysis_worker=Mock(running=False),
                                 _window_drag=None, _region_overlay=None, _force_dialog=None,
                                 _force_dialog_pending=False, shot_target=Mock(return_value={"window": {"hwnd": 123}}))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.root.destroy)

    def test_double_preview_size_marker_alignment_and_clear_on_render(self):
        main.render_result(snapshot())
        self.assertEqual(int(self.ui["canvas"]["width"]), 720)
        self.assertEqual(int(self.ui["canvas"]["height"]), 380)
        factor = 720/330
        self.assertEqual(main._preview_photo.width(), 720)
        own_ring = self.ui["canvas"].find_all()[1]
        coords = self.ui["canvas"].coords(own_ring)
        self.assertAlmostEqual((coords[0]+coords[2])/2, 20*factor)
        self.assertAlmostEqual((coords[1]+coords[3])/2, 30*factor)
        self.ui["table"].selection_set("1")
        main.update_shot_controls()
        self.assertEqual(str(self.ui["shoot"]["state"]), "normal")
        main.render_result(snapshot())
        self.assertEqual(self.ui["table"].selection(), ())
        self.assertEqual(str(self.ui["shoot"]["state"]), "disabled")
        self.assertEqual(str(self.ui["table"]["selectmode"]), "browse")
        self.ui["table"].selection_set("1")
        main.clear_results("已清空")
        self.assertEqual(self.ui["table"].selection(), ())
        self.assertEqual(str(self.ui["shoot"]["state"]), "disabled")

    def test_normal_cell_selects_target_without_changing_self(self):
        main.render_result(snapshot())
        table = self.ui["table"]
        self.root.geometry("1400x1050+30000+30000")
        self.root.deiconify()
        self.root.update()
        for column in ("#2", "#1"):
            box = table.bbox("1", column)
            self.assertTrue(box)
            table.event_generate("<Button-1>", x=box[0]+box[2]//2, y=box[1]+box[3]//2)
            table.event_generate("<ButtonRelease-1>", x=box[0]+box[2]//2, y=box[1]+box[3]//2)
            self.root.update()
            if column == "#2":
                self.assertEqual(table.selection(), ("1",))
                self.assertEqual(str(self.ui["shoot"]["state"]), "normal")
                main._analysis_worker.select_self.assert_not_called()
            else:
                main._analysis_worker.select_self.assert_called_once_with(main._displayed_result, 1)
                self.assertEqual(table.selection(), ())
                self.assertEqual(str(self.ui["shoot"]["state"]), "disabled")

    def test_footer_visible_and_parameters_reflow_for_small_and_large_windows(self):
        # Map off screen so Tk performs real geometry without showing a test window.
        self.root.geometry("1400x1000+30000+30000")
        self.root.deiconify()
        self.root.update()
        self.assertEqual(int(self.ui["preview_parameters"].grid_info()["row"]), 0)
        for size in ("900x650", "760x600"):
            with self.subTest(size=size):
                self.root.geometry(size+"+30000+30000")
                self.root.update()
                self.assertEqual(int(self.ui["preview_parameters"].grid_info()["row"]), 1)
                for key in ("shoot", "log"):
                    widget = self.ui[key]
                    self.assertTrue(widget.winfo_ismapped())
                    self.assertGreater(widget.winfo_width(), 0)
                    self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),
                                         self.root.winfo_rooty()+self.root.winfo_height())
                region = tuple(float(value) for value in self.ui["viewport"]["scrollregion"].split())
                self.assertGreater(region[3], self.ui["viewport"].winfo_height())

    def test_compact_sidebar_preserves_selection_details_and_scales_markers(self):
        self.root.geometry("1120x1000+30000+30000")
        self.root.deiconify()
        self.root.update()
        main.render_result(snapshot())
        self.ui["table"].selection_set("1")
        self.root.update()
        self.ui["set_sidecar_layout"](True)
        self.root.geometry("440x850+30000+30000")
        self.root.update()
        self.assertEqual(self.ui["table"]["displaycolumns"], ("我", "编号", "身份", "方向", "建议力度"))
        self.assertEqual(self.ui["table"].selection(), ("1",))
        self.assertIn("水平距离 3.00", self.ui["target_details"].get())
        self.assertIn("高低差 2.00", self.ui["target_details"].get())
        self.assertIn("可用", self.ui["target_details"].get())
        canvas = self.ui["canvas"]
        self.assertLess(int(canvas["width"]), 440)
        factor = main._preview_photo.width()/330
        own_ring = canvas.find_all()[1]
        coords = canvas.coords(own_ring)
        self.assertAlmostEqual((coords[0]+coords[2])/2, 20*factor, delta=.1)
        for key in ("shoot", "log", "details_label"):
            widget = self.ui[key]
            self.assertTrue(widget.winfo_ismapped())
            self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),
                                 self.root.winfo_rooty()+self.root.winfo_height())
        self.ui["set_sidecar_layout"](False)
        self.root.geometry("1120x1000+30000+30000")
        self.root.update()
        self.assertEqual(len(self.ui["table"]["displaycolumns"]), 8)
        self.assertEqual(self.ui["table"].selection(), ("1",))

    def test_compact_self_row_still_shows_details_and_cannot_fire(self):
        main.render_result(snapshot())
        self.ui["set_sidecar_layout"](True)
        self.ui["table"].selection_set("0")
        main.update_shot_controls()
        self.assertIn("自己", self.ui["target_details"].get())
        self.assertEqual(str(self.ui["shoot"]["state"]), "disabled")

    def test_compact_preview_leaves_first_target_visible_at_hall_height(self):
        self.ui["set_sidecar_layout"](True)
        scale = float(self.root.tk.call("tk", "scaling"))*72/96
        self.root.geometry(f"{round(440*scale)}x{round(628*scale)}+30000+30000")
        self.root.deiconify()
        self.root.update()
        main.render_result(snapshot())
        self.root.update()
        box = self.ui["table"].bbox("0")
        self.assertTrue(box)
        row_bottom = self.ui["table"].winfo_rooty()+box[1]+box[3]
        viewport_bottom = self.ui["viewport"].winfo_rooty()+self.ui["viewport"].winfo_height()
        self.assertLessEqual(row_bottom, viewport_bottom)

    def test_independent_focus_replaces_popup_without_switching_to_compact_layout(self):
        self.root.geometry("1120x1000+30000+30000")
        self.root.deiconify()
        self.root.update()
        main.render_result(snapshot())
        self.ui["table"].selection_set("1")
        original_size = self.root.geometry()
        self.ui["set_keep_focus_layout"](True)
        self.root.update()
        self.assertEqual(self.ui["mode_selector"].winfo_manager(), "")
        self.assertEqual(self.ui["mode_buttons"].winfo_manager(), "pack")
        self.assertEqual(len(self.ui["table"]["displaycolumns"]), 8)
        self.assertEqual(self.ui["table"].selection(), ("1",))
        self.assertEqual(self.root.geometry(), original_size)
        self.ui["set_keep_focus_layout"](False)
        self.root.update()
        self.assertEqual(self.ui["mode_selector"].winfo_manager(), "pack")
        self.assertEqual(self.ui["mode_buttons"].winfo_manager(), "")


if __name__ == "__main__":
    unittest.main()
