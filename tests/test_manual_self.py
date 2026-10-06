"""Manual self selection, snapshot calculations and stale worker protection."""

import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from skimage import color

from analysis import AnalysisResult, AnalysisWorker, SnapshotAnalyzer, TurnAnalysisWorker, MINIMAP_REGION
from test_multi_target import frame_with_map, minimap


def without_halo():
    image = minimap()
    hsv = color.rgb2hsv(image)
    yy, xx = np.indices(image.shape[:2])
    radius = np.hypot(xx-108.5, yy-35.5)
    image[(radius > 10) & (radius < 24) & (hsv[:, :, 0] > .55) & (hsv[:, :, 0] < .72)] = (45, 35, 100)
    return frame_with_map(image)


class ManualSelfTests(unittest.TestCase):
    def analyzer(self):
        return SnapshotAnalyzer(Mock(return_value=(.7, False)), Mock(return_value="27"), Mock(return_value=100))

    def own(self, result):
        return next(t.player for t in result.targets if t.player.identity == "自己")

    def test_failed_halo_snapshot_can_select_self_and_compute_all_other_targets(self):
        analyzer = self.analyzer()
        snapshot = analyzer(without_halo())
        self.assertEqual(snapshot.error_kind, "self_missing")
        self.assertGreater(len(snapshot.targets), 1)
        chosen = snapshot.targets[-1].player
        result = analyzer.analyze_selection(snapshot, len(snapshot.targets)-1)
        self.assertEqual(result.error, "")
        self.assertTrue(result.manual_self)
        self.assertEqual((self.own(result).x, self.own(result).y), (chosen.x, chosen.y))
        self.assertEqual(result.timestamp, snapshot.timestamp)
        self.assertEqual(result.degree, 27)
        self.assertEqual(result.wind, .7)
        self.assertTrue(any(t.force is not None for t in result.targets))
        for target in result.targets:
            if target.player.identity != "自己":
                expected_dx = 0 if abs(target.player.x-chosen.x) < 1 else abs(target.player.x-chosen.x)/100*10
                self.assertAlmostEqual(target.dx, expected_dx)
                self.assertAlmostEqual(target.dy, (chosen.y-target.player.y)/100*10)

    def test_selection_overrides_halo_and_cancel_restores_automatic_self(self):
        analyzer = self.analyzer()
        snapshot = analyzer(frame_with_map())
        original = self.own(snapshot)
        index = next(i for i, t in enumerate(snapshot.targets) if t.player.identity == "敌人")
        selected = analyzer.analyze_selection(snapshot, index)
        self.assertTrue(selected.manual_self)
        self.assertNotEqual(self.own(selected).x, original.x)
        automatic = analyzer.analyze_selection(selected, None)
        self.assertFalse(automatic.manual_self)
        self.assertAlmostEqual(self.own(automatic).x, original.x)

    def test_selection_survives_refresh_resize_and_row_reordering(self):
        analyzer = self.analyzer()
        frame = without_halo()
        snapshot = analyzer(frame)
        selected = analyzer.analyze_selection(snapshot, len(snapshot.targets)-1)
        own = self.own(selected)
        refreshed = analyzer(frame)
        self.assertTrue(refreshed.manual_self)
        self.assertAlmostEqual(self.own(refreshed).x, own.x)
        resized = analyzer(np.array(Image.fromarray(frame).resize((2000, 1200), Image.Resampling.BILINEAR)))
        self.assertEqual(resized.error, "")
        self.assertTrue(resized.manual_self)
        self.assertAlmostEqual(self.own(resized).x/(4/3), own.x, delta=1)

    def test_missing_selected_dot_does_not_switch_self_or_calculate(self):
        analyzer = self.analyzer()
        snapshot = analyzer(without_halo())
        selected = analyzer.analyze_selection(snapshot, 0)
        own = self.own(selected)
        frame = without_halo()
        x, y = round(own.x+MINIMAP_REGION[0]), round(own.y+MINIMAP_REGION[1])
        frame[y-14:y+15, x-14:x+15] = (45, 35, 100)
        result = analyzer(frame)
        self.assertIn("重新勾选", result.error)
        self.assertTrue(all(t.force is None for t in result.targets))

    def test_invalid_parameters_still_fail_and_reset_clears_manual_choice(self):
        analyzer = self.analyzer()
        snapshot = analyzer(without_halo())
        analyzer.readers[1].return_value = ""
        result = analyzer.analyze_selection(snapshot, 0)
        self.assertTrue(result.manual_self)
        self.assertEqual(result.error_kind, "parameters")
        self.assertTrue(all(t.force is None for t in result.targets))
        analyzer.reset()
        self.assertEqual(analyzer(without_halo()).error_kind, "self_missing")

    def test_changed_map_clears_selection_instead_of_selecting_a_different_person(self):
        analyzer = self.analyzer()
        snapshot = analyzer(without_halo())
        analyzer.analyze_selection(snapshot, 0)
        frame = without_halo()
        x, y, width, height = MINIMAP_REGION
        frame[y:y+height, x:x+width] = np.clip(frame[y:y+height, x:x+width].astype(int)+40, 0, 255)
        result = analyzer(frame)
        self.assertFalse(result.manual_self)
        self.assertIsNone(analyzer._manual)
        self.assertTrue(result.error)


class SelectionWorkerTests(unittest.TestCase):
    def test_selection_discards_inflight_result_and_uses_clicked_snapshot_without_capture(self):
        for worker_type in (AnalysisWorker, TurnAnalysisWorker):
            with self.subTest(worker=worker_type.__name__):
                entered, release = threading.Event(), threading.Event()
                analyzer = Mock()
                def old_analysis(_):
                    entered.set()
                    release.wait(2)
                    return AnalysisResult(error="obsolete")
                analyzer.side_effect = old_analysis
                expected = AnalysisResult(manual_self=True)
                analyzer.analyze_selection.return_value = expected
                capture = Mock(return_value="old frame")
                args = (capture, analyzer) if worker_type is AnalysisWorker else (capture, analyzer, lambda _: False)
                worker = worker_type(*args)
                self.addCleanup(worker.close)
                worker.configure((0, 0, 1500, 900), False)
                worker.start()
                worker.refresh()
                self.assertTrue(entered.wait(2))
                snapshot = AnalysisResult(frame=np.zeros((1, 1, 3)))
                worker.select_self(snapshot, 1)
                release.set()
                deadline = time.monotonic()+2
                result = None
                while time.monotonic() < deadline:
                    result = worker.take_result()
                    if result is not None and result.manual_self:
                        break
                    threading.Event().wait(.01)
                self.assertIsNotNone(result)
                self.assertTrue(result.manual_self)
                self.assertFalse(result.error)
                analyzer.analyze_selection.assert_called_once_with(snapshot, 1)
                capture.assert_called_once()
                self.assertFalse(worker.running)


class SelectionUITests(unittest.TestCase):
    def test_first_column_click_selects_exact_row_and_checked_self_can_be_cleared(self):
        import main
        from vision import PlayerMarker
        from analysis import TargetEstimate
        table = Mock()
        table.identify_region.return_value = "cell"
        table.identify_column.return_value = "#1"
        table.identify_row.return_value = "1"
        worker, status = Mock(), Mock()
        snapshot = AnalysisResult(targets=(TargetEstimate(PlayerMarker(20, 30, (255, 0, 0))),
                                         TargetEstimate(PlayerMarker(80, 50, (0, 255, 0), "自己", "我"))),
                                  frame=np.zeros((1, 1, 3)))
        with patch.multiple(main, _ui={"table": table, "status": status}, _analysis_worker=worker,
                            _displayed_result=snapshot, _mode=main.AUTO_MODE, _region_overlay=None, _window_drag=None):
            main.select_own_player(SimpleNamespace(x=10, y=20))
            worker.select_self.assert_called_with(snapshot, 1)
            main._displayed_result = replace(snapshot, manual_self=True)
            main.select_own_player(SimpleNamespace(x=10, y=20))
            worker.select_self.assert_called_with(main._displayed_result, None)
            table.identify_column.return_value = "#2"
            worker.reset_mock()
            main.select_own_player(SimpleNamespace(x=10, y=20))
            worker.select_self.assert_not_called()

    def test_render_adds_single_checked_box_before_player_number(self):
        import main
        analyzer = SnapshotAnalyzer(Mock(return_value=(0, False)), Mock(return_value="27"), Mock(return_value=100))
        result = analyzer.analyze_selection(analyzer(without_halo()), 0)
        root = main.tkinter.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        ui = main.build_ui(root)
        with patch.object(main, "_ui", ui):
            main.render_result(result)
            rows = [ui["table"].item(row, "values") for row in ui["table"].get_children()]
            self.assertEqual(ui["table"]["columns"][0:2], ("我", "编号"))
            self.assertEqual(sum(row[0] == "☑" for row in rows), 1)
            self.assertTrue(all(row[2] == "自己" for row in rows if row[0] == "☑"))
            self.assertTrue(ui["table"].bind("<Button-1>"))


if __name__ == "__main__":
    unittest.main()
