"""Fresh angle and position verification without screen or keyboard input."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import main
from analysis import crop_region, MINIMAP_REGION
from shot import ShotStateChanged
from turn import InputState
from vision import recognize_players
from test_multi_target import frame_with_map
from test_manual_self import without_halo


class ShotStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame_with_map()
        detection = recognize_players(crop_region(self.frame, MINIMAP_REGION))
        own = next(p for p in detection.players if p.identity == "自己")
        self.target = {"region": (100, 200, 1500, 900), "window": {"hwnd": 123},
                       "shot_state": InputState(65, own.x, own.y)}
        self.capture = self.enterContext(patch.object(main, "_capture_region", return_value=self.frame))
        self.resolve = self.enterContext(patch.object(main, "resolve_region", return_value=(300, 400, 1500, 900)))
        self.degree = self.enterContext(patch.object(main, "recognize", return_value="65"))

    def test_fresh_capture_uses_frozen_binding_and_accepts_unchanged_or_mirrored_angle(self):
        for angle in ("65", "115"):
            self.degree.return_value = angle
            main.check_shot_state(self.target)
        self.resolve.assert_called_with(self.target)
        self.capture.assert_called_with((300, 400, 1500, 900))

    def test_one_degree_change_rejects_old_force(self):
        self.degree.return_value = "64"
        with self.assertRaisesRegex(ShotStateChanged, "原力度已失效"):
            main.check_shot_state(self.target)

    def test_small_real_marker_movement_is_measured_even_without_a_halo(self):
        frame = without_halo()
        x, y, width, height = MINIMAP_REGION
        frame[y:y+height, x:x+width] = np.roll(frame[y:y+height, x:x+width], 2, axis=1)
        self.capture.return_value = frame
        with self.assertRaisesRegex(ShotStateChanged, "人物位置发生变化"):
            main.check_shot_state(self.target)

    def test_unchanged_dot_during_halo_gap_is_still_verifiable(self):
        self.capture.return_value = without_halo()
        main.check_shot_state(self.target)

    def test_unchanged_self_coordinates_survive_window_scaling(self):
        for factor in (.75, 4/3, 1.5):
            with self.subTest(factor=factor):
                self.capture.return_value = np.array(Image.fromarray(self.frame).resize(
                    (round(1500*factor), round(900*factor)), Image.Resampling.BILINEAR))
                main.check_shot_state(self.target)

    def test_missing_self_or_unreadable_angle_prevents_firing(self):
        self.capture.return_value = np.zeros_like(self.frame)
        with self.assertRaisesRegex(ShotStateChanged, "无法确认自己的当前位置"):
            main.check_shot_state(self.target)
        self.capture.return_value = self.frame
        for angle in ("", "bad", "999"):
            with self.subTest(angle=angle):
                self.degree.return_value = angle
                with self.assertRaisesRegex(ShotStateChanged, "无法确认当前角度"):
                    main.check_shot_state(self.target)

    def test_missing_baseline_does_not_capture_or_trust_an_old_result(self):
        with self.assertRaises(ShotStateChanged):
            main.check_shot_state({"window": {"hwnd": 123}})
        self.capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
