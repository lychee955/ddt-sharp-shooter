"""Real screenshot, changing palettes, trajectory and worker regressions."""

from dataclasses import replace
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
from skimage import color

from analysis import AnalysisResult, AnalysisWorker, SnapshotAnalyzer, analyze_frame, crop_region, estimate_targets, MINIMAP_REGION
from force import calc_force, solve_force, trajectory
from vision import PlayerMarker, recognize_players
from turn import InputState


FIXTURES = Path(__file__).parent / "fixtures"
SCALE = 4 / 3


def minimap():
    return np.array(Image.open(FIXTURES / "minimap.png").convert("RGB"))


def recolour(image, team, enemy):
    result = image.copy()
    detection = recognize_players(image, SCALE)
    yy, xx = np.indices(image.shape[:2])
    for player in detection.players:
        local = np.hypot(xx - player.x, yy - player.y) <= 10
        same = np.linalg.norm(image.astype(float) - player.rgb, axis=2) < 110
        result[local & same] = team if player.identity in {"自己", "队友"} else enemy
    return result


def frame_with_map(image=None):
    frame = np.zeros((900, 1500, 3), dtype=np.uint8)
    image = minimap() if image is None else image
    frame[45:185, 1230:1500] = np.array(Image.fromarray(image).resize((270, 140), Image.Resampling.BILINEAR))
    return frame


class MarkerTests(unittest.TestCase):
    def assert_eight(self, image, scale=SCALE):
        result = recognize_players(image, scale)
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.players), 8)
        self.assertEqual([p.identity for p in result.players].count("自己"), 1)
        self.assertEqual([p.identity for p in result.players].count("队友"), 3)
        self.assertEqual([p.identity for p in result.players].count("敌人"), 4)
        self.assertEqual(len(set(p.label for p in result.players)), 8)
        own = next(p for p in result.players if p.identity == "自己")
        self.assertAlmostEqual(own.x / scale, 81.2, delta=1)
        self.assertAlmostEqual(own.y / scale, 26.5, delta=1)
        return result

    def test_uploaded_map_has_eight_and_no_decorations(self):
        self.assert_eight(minimap())

    def test_new_map_transparent_halo_and_terrain(self):
        image = Image.open(FIXTURES / "faded_halo_map.png").convert("RGB")
        for factor in (1, .75, 1.5):
            with self.subTest(factor=factor):
                resized = np.array(image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.BILINEAR))
                result = recognize_players(resized, 2052/1500*factor)
                self.assertEqual(result.error, "")
                self.assertEqual([p.identity for p in result.players], ["自己", "队友", "敌人", "敌人"])
                self.assertAlmostEqual(result.players[0].x/factor, 33, delta=1)

    def test_new_map_without_transparent_halo_stays_uncertain(self):
        image = np.array(Image.open(FIXTURES / "faded_halo_map.png").convert("RGB"))
        yy, xx = np.indices(image.shape[:2])
        radius = np.hypot(xx-33, yy-49)
        image[(radius >= 10) & (radius <= 24)] = (250, 227, 144)
        result = recognize_players(image, 2052/1500)
        self.assertIn("未找到", result.error)
        self.assertFalse(any(p.identity == "自己" for p in result.players))

    def test_flat_orange_dots_are_separated_from_same_hue_terrain(self):
        image = np.array(Image.open(FIXTURES/"latest_map.png").convert("RGB"))
        result = recognize_players(image,1971/1500)
        orange = [p for p in result.players if p.rgb == (255,133,0)]
        self.assertEqual(len(orange),2)
        self.assertGreater(abs(orange[0].x-orange[1].x),20)

    def test_dynamic_palettes_including_blue_and_neutral_dots(self):
        palettes = [((230, 25, 25), (240, 220, 30)),
                    ((0, 200, 230), (240, 100, 20)),
                    ((0, 60, 220), (230, 25, 25)),
                    ((220, 220, 220), (230, 25, 25)),
                    ((191, 0, 156), (0, 210, 2))]
        for team, enemy in palettes:
            with self.subTest(team=team, enemy=enemy):
                self.assert_eight(recolour(minimap(), team, enemy))

    def test_successive_frames_do_not_remember_team_colour(self):
        self.assert_eight(recolour(minimap(), (230, 25, 25), (240, 220, 30)))
        self.assert_eight(recolour(minimap(), (191, 0, 156), (0, 210, 2)))

    def test_marker_sizes_follow_scale(self):
        image = Image.fromarray(minimap())
        for factor in (.75, 1.5, 2):
            with self.subTest(factor=factor):
                resized = np.array(image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.BILINEAR))
                self.assert_eight(resized, SCALE*factor)

    def test_missing_halo_has_no_self(self):
        image = minimap()
        hsv = color.rgb2hsv(image)
        yy, xx = np.indices(image.shape[:2])
        ring = (np.hypot(xx-108.5, yy-35.5) > 10) & (np.hypot(xx-108.5, yy-35.5) < 24)
        image[ring & (hsv[:,:,0]>.55) & (hsv[:,:,0]<.72)] = (45, 35, 100)
        result = recognize_players(image, SCALE)
        self.assertIn("未找到", result.error)
        self.assertFalse(any(p.identity == "自己" for p in result.players))

    def test_two_halos_are_ambiguous(self):
        image = Image.fromarray(minimap())
        ImageDraw.Draw(image).ellipse((91, 90, 126, 125), outline=(0, 55, 230), width=4)
        result = recognize_players(np.array(image), SCALE)
        self.assertIn("多个", result.error)

    def test_similar_but_uncertain_team_colour_is_not_assumed(self):
        image = minimap()
        own_rgb = np.array([0, 210, 2])
        own_lab = color.rgb2lab(own_rgb.reshape(1,1,3)/255)[0,0]
        chosen = None
        for red in range(60, 200):
            rgb = np.array([red, 210, 2])
            lab = color.rgb2lab(rgb.reshape(1,1,3)/255)[0,0]
            if 17 < color.deltaE_ciede2000(lab, own_lab) < 20:
                chosen = tuple(rgb.tolist())
                break
        self.assertIsNotNone(chosen)
        image = recolour(image, (0, 210, 2), chosen)
        result = recognize_players(image, SCALE)
        self.assertEqual(result.error, "")
        self.assertTrue(any(p.identity == "身份不确定" for p in result.players))

    def test_overlapping_dots_are_not_invented_as_two_targets(self):
        image = Image.fromarray(minimap())
        draw = ImageDraw.Draw(image)
        draw.rectangle((320, 20, 355, 50), fill=(45, 35, 100))
        draw.ellipse((325, 26, 341, 42), fill=(191, 0, 156))
        draw.ellipse((332, 26, 348, 42), fill=(191, 0, 156))
        result = recognize_players(np.array(image), SCALE)
        self.assertEqual(result.error, "")
        self.assertLess(len(result.players), 9)


class TrajectoryTests(unittest.TestCase):
    def test_checked_solution_reaches_targets(self):
        for degree, wind, dx, dy in ((65,0,12,1), (30,2,15,-2), (115,-2,12,1), (9,0,17,0), (0,0,10,-5)):
            with self.subTest(degree=degree, wind=wind):
                solution = solve_force(degree, wind, dx, dy)
                self.assertGreater(solution.flight_time, 0)
                self.assertLess(solution.residual, 1e-6)
                x, y = trajectory(solution.force, degree, wind, solution.flight_time)
                self.assertAlmostEqual(x, dx, places=6)
                self.assertAlmostEqual(y, dy, places=6)

    def test_above_100_is_not_clamped_in_analysis(self):
        solution = solve_force(9,0,17,0)
        self.assertGreater(solution.force,100)
        self.assertEqual(calc_force(9,0,17,0),100)
        self.assertGreater(calc_force(9,0,17,0,clamp=False),100)

    def test_backward_angle_does_not_change_wind(self):
        self.assertAlmostEqual(solve_force(65,2,12,1).force, solve_force(115,2,12,1).force)

    def test_unreachable_and_invalid_inputs_fail(self):
        for args in ((0,0,10,2), (90,0,10,0), (65,0,-1,0), (999,0,10,0), (65,float("nan"),10,0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                solve_force(*args)

    def test_nonconvergence_fails(self):
        with patch("force.brentq", return_value=(1, Mock(converged=False))):
            with self.assertRaisesRegex(ValueError,"未收敛"):
                solve_force(65,0,12,0)


class SnapshotTests(unittest.TestCase):
    def readers(self, degree="65"):
        return Mock(return_value=(0,False)), Mock(return_value=degree), Mock(return_value=100)

    def test_single_snapshot_computes_seven_other_people(self):
        readers = self.readers()
        result = analyze_frame(frame_with_map(), *readers)
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.targets), 8)
        self.assertEqual(sum(t.player.identity == "自己" for t in result.targets), 1)
        self.assertEqual(sum(t.dx is not None for t in result.targets), 7)
        for reader in readers:
            reader.assert_called_once()

    def test_wide_layout_preserves_left_self_and_both_map_edges_at_each_scale(self):
        from ocr import recognize_ten_units
        image = np.array(Image.open(FIXTURES/"wide_minimap.png").convert("RGB"))
        frame = np.zeros((900,1500,3),dtype=np.uint8)
        x,y,w,h = MINIMAP_REGION
        frame[y:y+h,x:x+w] = image
        for scale in (.75,1,4/3):
            with self.subTest(scale=scale):
                resized = np.array(Image.fromarray(frame).resize(
                    (round(1500*scale),round(900*scale)),Image.Resampling.BILINEAR))
                result = analyze_frame(resized,Mock(return_value=(1.6,True)),
                                       Mock(return_value="1"),recognize_ten_units)
                self.assertEqual(result.error,"")
                self.assertEqual([t.player.identity for t in result.targets],["自己","敌人"])
                self.assertAlmostEqual(result.targets[0].player.x/scale,36.7,delta=1)
                self.assertAlmostEqual(result.targets[1].dx,16.55,delta=.25)
                self.assertEqual(result.targets[1].direction,"右")
                self.assertEqual(result.minimap.shape[1],int(330*scale))
                old = recognize_players(crop_region(resized,(1230,45,270,140)),scale)
                self.assertEqual(old.error_kind,"self_missing")

    def test_invalid_ocr_clears_force_in_current_frame(self):
        frame = frame_with_map()
        previous = analyze_frame(frame,*self.readers())
        self.assertTrue(any(t.force is not None for t in previous.targets))
        current = analyze_frame(frame,*self.readers(""))
        self.assertIn("角度识别失败",current.error)
        self.assertTrue(all(t.force is None for t in current.targets))

    def test_no_self_does_not_read_or_estimate(self):
        readers = self.readers()
        result = analyze_frame(np.zeros((900,1500,3),dtype=np.uint8),*readers)
        self.assertIn("蓝色光圈",result.error)
        for reader in readers:
            reader.assert_not_called()

    def test_missing_scale_clears_force(self):
        readers = self.readers()
        readers[2].return_value = 0
        result = analyze_frame(frame_with_map(),*readers)
        self.assertIn("标尺",result.error)
        self.assertTrue(all(t.force is None for t in result.targets))

    def test_garbled_wind_is_not_displayed_or_used(self):
        readers = self.readers()
        readers[0].return_value = (506.2,False)
        result = analyze_frame(frame_with_map(),*readers)
        self.assertIn("风力识别结果无效",result.error)
        self.assertIsNone(result.wind)
        self.assertTrue(all(t.force is None for t in result.targets))

    def test_directions_wind_and_height_are_relative_to_each_target(self):
        players = (PlayerMarker(100,100,(1,2,3),"自己","我"),
                   PlayerMarker(80,90,(4,5,6),"敌人","敌1"),
                   PlayerMarker(120,110,(4,5,6),"队友","友1"))
        with patch("analysis.solve_force", return_value=Mock(force=30)) as solve:
            estimates = estimate_targets(players,65,2,True,100)
        self.assertEqual([call.args for call in solve.call_args_list],[(65,2,2,1),(65,-2,2,-1)])
        self.assertEqual([t.direction for t in estimates],["—","左","右"])

    def test_above_limit_and_uncertain_identity_have_explicit_status(self):
        own = PlayerMarker(0,0,(1,2,3),"自己","我")
        target = PlayerMarker(170,0,(4,5,6),"敌人","敌1")
        result = estimate_targets((own,target),9,0,False,100)
        self.assertGreater(result[1].force,100)
        self.assertEqual(result[1].status,"当前角度力度不足")
        result = estimate_targets((own,replace(target,identity="身份不确定")),9,0,False,100)
        self.assertGreater(result[1].force,100)
        self.assertEqual(result[1].status,"当前角度力度不足")
        self.assertEqual(result[1].player.identity,"身份不确定")

    def test_uncertain_identity_uses_same_geometry_wind_and_solver(self):
        own = PlayerMarker(100,100,(1,2,3),"自己","我")
        players = (own,PlayerMarker(80,90,(4,5,6),"身份不确定","?1"),
                   PlayerMarker(120,110,(7,8,9),"身份不确定","?2"))
        with patch("analysis.solve_force",return_value=Mock(force=35)) as solve:
            estimates = estimate_targets(players,65,2,True,100)
        self.assertEqual([call.args for call in solve.call_args_list],[(65,2,2,1),(65,-2,2,-1)])
        self.assertEqual([t.force for t in estimates],[None,35,35])
        self.assertEqual([t.status for t in estimates],["自己","可用","可用"])
        self.assertTrue(all(t.player.identity == "身份不确定" for t in estimates[1:]))
        with patch("analysis.solve_force",side_effect=ValueError("没有有效轨迹解")):
            failed = estimate_targets(players,65,2,True,100)
        self.assertEqual(failed[1].status,"没有有效轨迹解")
        self.assertIsNone(failed[1].force)

    def test_subpixel_horizontal_noise_does_not_produce_zero_force(self):
        own = PlayerMarker(100,100,(1,2,3),"自己","我")
        other = PlayerMarker(100.05,120,(4,5,6),"队友","友1")
        estimates = estimate_targets((own,other),9,0,False,100)
        self.assertEqual(estimates[1].direction,"垂直")
        self.assertIsNone(estimates[1].force)


class OCRAndUITests(unittest.TestCase):
    def test_full_map_scale_uses_viewport_not_outer_frame(self):
        from ocr import recognize_ten_units
        for name,width in (("wide_minimap.png",144),("minimap.png",135),
                           ("faded_halo_map.png",149),("latest_map.png",128)):
            image = Image.open(FIXTURES/name).convert("RGB")
            for scale in (.75,1,1.5):
                with self.subTest(name=name,scale=scale):
                    resized = image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BILINEAR)
                    self.assertAlmostEqual(recognize_ten_units(np.array(resized)),width*scale,delta=2)

    def test_outer_frame_alone_is_not_a_distance_scale(self):
        from ocr import recognize_ten_units
        image = np.zeros((150,330,3),dtype=np.uint8)
        image[[0,-1],:] = (160,163,169)
        image[:,[0,-1]] = (160,163,169)
        self.assertEqual(recognize_ten_units(image),0)
        # The viewport can share its left/top edges with the minimap frame.
        image[:81,140] = (160,163,169)
        image[80,:141] = (160,163,169)
        self.assertEqual(recognize_ten_units(image),140)

    def test_latest_screenshot_parameter_crops(self):
        from ocr import recognize, recognize_wind
        angle = np.array(Image.open(FIXTURES/"latest_angle.png").convert("RGB"))
        wind = np.array(Image.open(FIXTURES/"latest_wind.png").convert("RGB"))
        self.assertEqual(recognize(angle), "65")
        self.assertEqual(recognize_wind(wind), (.7, False))

    def test_real_wind_ocr_keeps_zero(self):
        from ocr import recognize_wind
        wind = np.array(Image.open(FIXTURES/"wind.png").convert("RGB"))
        self.assertEqual(recognize_wind(wind)[0],0.0)

    def test_real_snapshot_ocr_and_table_rendering_without_keyboard(self):
        import main
        from preview_ui import sample_frame
        from ocr import recognize,recognize_wind,recognize_ten_units
        result = analyze_frame(sample_frame(),recognize_wind,recognize,recognize_ten_units)
        self.assertEqual(result.error,"")
        self.assertEqual(result.wind,0.0)
        self.assertEqual(result.degree,9)
        self.assertEqual(len(result.targets),8)
        result = replace(result,phase="complete",attempts=2)
        root = main.tkinter.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        ui = main.build_ui(root)
        ui["calibration"].set("游戏区域标记成功，已保存")
        with patch.object(main,"_ui",ui), patch.object(main,"_analysis_worker",Mock(running=False)), patch.object(main,"space_press") as press:
            main.render_result(result)
            root.update_idletasks()
            self.assertEqual(len(ui["table"].get_children()),8)
            self.assertEqual(len(ui["canvas"].find_all()),17)
            self.assertIn("自己 1",ui["summary"].get())
            self.assertEqual(ui["calibration"].get(),"游戏区域标记成功，已保存")
            self.assertIn("等待下次出手",ui["status"].get())
            self.assertIn("尝试 2 次",ui["status"].get())
            press.assert_not_called()


class TrackingTests(unittest.TestCase):
    def without_halo(self, image=None):
        image = minimap() if image is None else image.copy()
        hsv = color.rgb2hsv(image)
        yy, xx = np.indices(image.shape[:2])
        radius = np.hypot(xx-108.5, yy-35.5)
        image[(radius > 10) & (radius < 24) & (hsv[:,:,0]>.55) & (hsv[:,:,0]<.72)] = (45,35,100)
        return frame_with_map(image)

    def analyzer(self):
        self.now = 0
        return SnapshotAnalyzer(Mock(return_value=(0,False)), Mock(return_value="65"),
                                Mock(return_value=100), clock=lambda: self.now)

    def test_probe_reads_angle_and_self_without_wind_scale_or_solver(self):
        analyzer = self.analyzer()
        with patch("analysis.solve_force") as solve:
            state = analyzer.probe(frame_with_map())
        self.assertIsInstance(state,InputState)
        self.assertEqual(state.degree,65)
        self.assertAlmostEqual(state.x,81.2+1230-MINIMAP_REGION[0],delta=1)
        analyzer.readers[0].assert_not_called()
        analyzer.readers[2].assert_not_called()
        solve.assert_not_called()
        self.assertEqual(analyzer.probe(self.without_halo()),state)
        analyzer.readers[1].return_value = ""
        self.assertIsNone(analyzer.probe(frame_with_map()))

    def test_successful_analysis_records_baseline_in_normalized_coordinates(self):
        frame = np.array(Image.fromarray(frame_with_map()).resize((2000,1200)))
        analyzer = self.analyzer()
        result = analyzer(frame)
        self.assertEqual(result.error,"")
        self.assertEqual(result.input_state.degree,65)
        own = next(t.player for t in result.targets if t.player.identity == "自己")
        self.assertAlmostEqual(result.input_state.x,own.x/(4/3))
        self.assertAlmostEqual(result.input_state.y,own.y/(4/3))

    def test_halo_gap_keeps_cached_position_and_current_colours_for_thirty_seconds(self):
        analyzer = self.analyzer()
        self.assertEqual(analyzer(frame_with_map()).error, "")
        self.now = 1
        swapped = recolour(minimap(), (191,0,156), (0,210,2))
        current = analyzer(self.without_halo(swapped))
        self.assertEqual(current.error, "")
        self.assertTrue(current.tracking_note)
        self.assertEqual([t.player.identity for t in current.targets].count("队友"),3)
        self.assertTrue(any(t.force is not None for t in current.targets))
        self.now = 30
        continuing = analyzer(self.without_halo(swapped))
        self.assertEqual(continuing.error, "")
        self.assertTrue(continuing.uses_cached_self)
        self.assertEqual(next(t.player for t in current.targets if t.player.identity == "自己"),
                         next(t.player for t in continuing.targets if t.player.identity == "自己"))

    def test_cache_coordinates_are_not_moved_towards_nearby_dot(self):
        frame = self.without_halo()
        image = frame[45:185,1230:1500]
        own = next(p for p in recognize_players(frame_with_map()[45:185,1230:1500]).players if p.identity == "自己")
        hint = (own.x-2, own.y)
        result = recognize_players(image, own_hint=hint)
        cached = next(p for p in result.players if p.identity == "自己")
        self.assertEqual((cached.x,cached.y),hint)
        self.assertTrue(result.used_hint)

    def test_transient_missing_dot_does_not_delete_cache_but_long_loss_does(self):
        analyzer = self.analyzer()
        analyzer(frame_with_map())
        missing = self.without_halo()
        missing[55:87,1296:1326] = (45,35,100)
        self.now = 1
        self.assertTrue(analyzer(missing).error)
        self.now = 2
        self.assertEqual(analyzer(self.without_halo()).error, "")
        self.now = 3
        analyzer(missing)
        self.now = 6
        self.assertTrue(analyzer(self.without_halo()).error)

    def test_initial_missing_halo_and_reset_do_not_guess_self(self):
        analyzer = self.analyzer()
        self.assertIn("蓝色光圈",analyzer(self.without_halo()).error)
        self.assertEqual(analyzer(frame_with_map()).error, "")
        analyzer.reset()
        self.assertIn("蓝色光圈",analyzer(self.without_halo()).error)

    def test_changed_map_does_not_reuse_self(self):
        analyzer = self.analyzer()
        analyzer(frame_with_map())
        frame = self.without_halo()
        frame[45:185,1230:1500] = np.clip(frame[45:185,1230:1500].astype(int)+40,0,255)
        result = analyzer(frame)
        self.assertFalse(result.tracking_note)
        self.assertTrue(result.error)

    def test_missing_dot_at_last_position_is_not_assumed(self):
        analyzer = self.analyzer()
        analyzer(frame_with_map())
        frame = self.without_halo()
        frame[55:87,1296:1326] = (45,35,100)
        result = analyzer(frame)
        self.assertIn("蓝色光圈",result.error)
        self.assertTrue(all(t.force is None for t in result.targets))

    def test_pause_and_region_change_reset_tracking(self):
        analyzer = self.analyzer()
        analyzer(frame_with_map())
        worker = AnalysisWorker(Mock(),analyzer)
        worker.pause()
        self.assertIn("蓝色光圈",analyzer(self.without_halo()).error)
        analyzer(frame_with_map())
        worker.configure((0,0,1500,900),True)
        self.assertIn("蓝色光圈",analyzer(self.without_halo()).error)


class WorkerTests(unittest.TestCase):
    def wait_for_result(self,worker):
        until = time.monotonic()+2
        while time.monotonic()<until:
            result = worker.take_result()
            if result is not None:
                return result
            threading.Event().wait(.01)
        self.fail("worker did not publish a result")

    def test_refresh_while_paused_captures_exactly_once(self):
        capture = Mock(return_value="frame")
        worker = AnalysisWorker(capture,lambda frame: AnalysisResult(error=frame))
        worker.configure((1,2,1500,900))
        worker.start()
        self.addCleanup(worker.close)
        worker.refresh()
        self.assertEqual(self.wait_for_result(worker).error,"frame")
        self.assertFalse(worker.running)
        capture.assert_called_once_with((1,2,1500,900))

    def test_pause_discards_inflight_result(self):
        entered, release, finished = threading.Event(),threading.Event(),threading.Event()
        def analyze(frame):
            entered.set()
            release.wait(2)
            finished.set()
            return AnalysisResult(error="obsolete")
        worker = AnalysisWorker(Mock(return_value="frame"),analyze)
        worker.configure((0,0,1500,900),True)
        worker.start()
        self.addCleanup(worker.close)
        self.assertTrue(entered.wait(2))
        worker.pause()
        release.set()
        self.assertTrue(finished.wait(2))
        self.assertIsNone(worker.take_result())

    def test_new_region_discards_old_work(self):
        entered, release = threading.Event(),threading.Event()
        def capture(region):
            if region[0]==1:
                entered.set()
                release.wait(2)
            return region
        worker = AnalysisWorker(capture,lambda frame: AnalysisResult(error=str(frame[0])),interval=10)
        worker.configure((1,0,1500,900),True)
        worker.start()
        self.addCleanup(worker.close)
        self.assertTrue(entered.wait(2))
        worker.configure((2,0,1500,900),True)
        release.set()
        self.assertEqual(self.wait_for_result(worker).error,"2")

    def test_periodic_refresh_has_bounded_result_queue(self):
        capture = Mock(return_value="frame")
        worker = AnalysisWorker(capture,lambda _: AnalysisResult(),interval=.02)
        worker.configure((0,0,1500,900),True)
        worker.start()
        self.addCleanup(worker.close)
        threading.Event().wait(.09)
        self.assertGreaterEqual(capture.call_count,2)
        self.assertLessEqual(worker.results.qsize(),1)


if __name__ == "__main__":
    unittest.main()
