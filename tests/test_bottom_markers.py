"""Bottom-clipped dots and halos, including the user's reported screenshot."""

from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageDraw

from analysis import MINIMAP_REGION, SnapshotAnalyzer
from vision import recognize_players


FIXTURE = Path(__file__).parent / "fixtures" / "bottom_minimap.png"


def clipped_map(*, halo=True, rectangle=False):
    image = Image.new("RGB", (330, 150), (245, 220, 150))
    draw = ImageDraw.Draw(image)
    if halo:
        draw.ellipse((88, 108, 112, 132), outline=(0, 55, 230), width=3)
    if rectangle:
        draw.rectangle((94, 113, 106, 126), fill=(191, 0, 156))
    else:
        draw.ellipse((94, 114, 106, 126), fill=(191, 0, 156))
    draw.ellipse((254, 114, 266, 126), fill=(0, 145, 255))
    draw.rectangle((0, 120, 329, 149), fill=(35, 39, 97))
    return image


class BottomMarkerTests(unittest.TestCase):
    def test_reported_screenshot_finds_self_and_enemy_at_multiple_scales(self):
        image = Image.open(FIXTURE).convert("RGB")
        for factor in (.75, 1, 1.5, 2):
            with self.subTest(factor=factor):
                resized = image.resize((round(image.width*factor), round(image.height*factor)),
                                       Image.Resampling.BILINEAR)
                result = recognize_players(np.array(resized), 4/3*factor)
                self.assertEqual(result.error, "")
                self.assertEqual([p.identity for p in result.players], ["自己", "敌人"])
                own, enemy = result.players
                self.assertAlmostEqual(own.x/factor, 305, delta=1.5)
                self.assertAlmostEqual(own.y/factor, 160, delta=1.5)
                self.assertAlmostEqual(enemy.x/factor, 349, delta=1.5)
                self.assertAlmostEqual(enemy.y/factor, 149, delta=1.5)

    def test_both_half_dots_keep_circle_centers_instead_of_visible_centroids(self):
        result = recognize_players(np.array(clipped_map()))
        self.assertEqual(result.error, "")
        self.assertEqual([p.identity for p in result.players], ["自己", "敌人"])
        for player, x in zip(result.players, (100, 260)):
            self.assertAlmostEqual(player.x, x, delta=1)
            self.assertAlmostEqual(player.y, 120, delta=1)

    def test_image_itself_can_end_at_the_half_dot(self):
        result = recognize_players(np.array(clipped_map())[:120])
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.players), 2)
        self.assertAlmostEqual(result.players[0].y, 120, delta=1)

    def test_clipped_dot_without_halo_remains_unconfirmed(self):
        result = recognize_players(np.array(clipped_map(halo=False)))
        self.assertEqual(result.error_kind, "self_missing")
        self.assertEqual(len(result.players), 2)
        self.assertFalse(any(p.identity == "自己" for p in result.players))

    def test_small_blue_patch_does_not_replace_visible_half_ring(self):
        image = clipped_map(halo=False)
        ImageDraw.Draw(image).arc((88, 108, 112, 132), 200, 250,
                                 fill=(0, 55, 230), width=3)
        result = recognize_players(np.array(image))
        self.assertEqual(result.error_kind, "self_missing")
        self.assertFalse(any(p.identity == "自己" for p in result.players))

    def test_rectangle_at_boundary_is_not_a_player_even_with_halo(self):
        result = recognize_players(np.array(clipped_map(rectangle=True)))
        self.assertEqual(result.error_kind, "self_missing")
        self.assertEqual(len(result.players), 1)
        self.assertAlmostEqual(result.players[0].x, 260, delta=1)

    def test_reported_frame_calculates_and_preserves_manual_selection(self):
        from ocr import recognize_ten_units

        frame = np.zeros((1200, 2000, 3), dtype=np.uint8)
        x, y, width, height = (int(v*4/3) for v in MINIMAP_REGION)
        frame[y:y+height, x:x+width] = np.array(Image.open(FIXTURE).convert("RGB"))
        analyzer = SnapshotAnalyzer(Mock(return_value=(3.1, False)),
                                    Mock(return_value="58"), recognize_ten_units)
        result = analyzer(frame)
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.targets), 2)
        enemy = result.targets[1]
        self.assertEqual(enemy.status, "可用")
        self.assertAlmostEqual(enemy.dx, 44/199*10, delta=.1)
        self.assertAlmostEqual(enemy.dy, 11/199*10, delta=.1)
        self.assertGreater(enemy.force, 0)
        selected = analyzer.analyze_selection(result, 0)
        self.assertEqual(selected.error, "")
        self.assertTrue(selected.manual_self)
        refreshed = analyzer(frame)
        self.assertEqual(refreshed.error, "")
        self.assertTrue(refreshed.manual_self)


if __name__ == "__main__":
    unittest.main()
