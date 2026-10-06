"""Own-turn announcement and bounded per-turn calculation checks."""

from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from analysis import AnalysisResult, TurnAnalysisWorker
from turn import InputChangeGate, InputState, OwnTurnDetector, TurnLatch


FIXTURES = Path(__file__).parent/"fixtures"


def banner_frame(name):
    frame = np.zeros((900,1500,3),dtype=np.uint8)
    frame[160:590,300:1300] = np.array(Image.open(FIXTURES/name).convert("RGB"))
    return frame


class TurnDetectionTests(unittest.TestCase):
    def test_two_maps_and_scaled_announcements(self):
        detect = OwnTurnDetector()
        for name in ("own_turn_area.png","other_turn_area.png"):
            frame = Image.fromarray(banner_frame(name))
            for scale in (.75,1,4/3):
                with self.subTest(name=name,scale=scale):
                    resized = frame.resize((round(1500*scale),round(900*scale)),Image.Resampling.BILINEAR)
                    self.assertTrue(detect(np.array(resized)))

    def test_latch_rearms_only_after_sustained_absence(self):
        latch = TurnLatch()
        self.assertTrue(latch.observe(True,0))
        self.assertFalse(latch.observe(True,1))
        self.assertFalse(latch.observe(False,1.5))
        self.assertFalse(latch.observe(True,2))
        latch.observe(False,3)
        latch.observe(False,4.5)
        self.assertTrue(latch.observe(True,5))

class InputChangeTests(unittest.TestCase):
    def setUp(self):
        self.gate = InputChangeGate()
        self.baseline = InputState(65,80,25)
        self.gate.reset(self.baseline)

    def test_unchanged_and_small_position_jitter_do_not_recompute(self):
        for state in (self.baseline,InputState(65,81,25),InputState(65,79,25))*3:
            self.assertFalse(self.gate.observe(state))

    def test_angle_change_requires_two_stable_samples(self):
        for degree in (64,63,62):
            self.assertFalse(self.gate.observe(InputState(degree,80,25)))
        self.assertTrue(self.gate.observe(InputState(62,80,25)))
        for _ in range(5):
            self.assertFalse(self.gate.observe(InputState(62,80,25)))
        self.gate.reset(InputState(62,80,25))
        self.assertFalse(self.gate.observe(InputState(62,80,25)))

    def test_position_change_alone_triggers_once(self):
        self.assertFalse(self.gate.observe(InputState(65,83,26)))
        self.assertTrue(self.gate.observe(InputState(65,83.1,26)))
        self.assertFalse(self.gate.observe(InputState(65,83,26)))

    def test_invalid_observation_interrupts_stability(self):
        changed = InputState(66,80,25)
        self.assertFalse(self.gate.observe(changed))
        self.assertFalse(self.gate.observe(None))
        self.assertFalse(self.gate.observe(changed))
        self.assertTrue(self.gate.observe(changed))

    def test_failed_change_can_be_retried_after_reverting(self):
        changed = InputState(66,80,25)
        self.gate.observe(changed)
        self.assertTrue(self.gate.observe(changed))
        self.assertFalse(self.gate.observe(changed))
        self.assertFalse(self.gate.observe(self.baseline))
        self.assertFalse(self.gate.observe(changed))
        self.assertTrue(self.gate.observe(changed))


class TurnWorkerTests(unittest.TestCase):
    def worker(self,analyze,detector=None,**kwargs):
        worker = TurnAnalysisWorker(Mock(return_value="frame"),analyze,
                                    detector or Mock(return_value=True),interval=.01,retry_delay=.005,**kwargs)
        worker.configure((0,0,1500,900),True)
        self.addCleanup(worker.close)
        return worker

    def wait_for(self,worker,phase):
        deadline = time.monotonic()+2
        while time.monotonic()<deadline:
            result = worker.take_result()
            if result is not None and result.phase == phase:
                return result
            threading.Event().wait(.005)
        self.fail(f"no result with phase {phase}")

    def test_waiting_and_success_do_not_repeat_analysis(self):
        visible = threading.Event()
        analyze = Mock(return_value=AnalysisResult())
        worker = self.worker(analyze,lambda _: visible.is_set())
        worker.start()
        threading.Event().wait(.06)
        analyze.assert_not_called()
        visible.set()
        self.wait_for(worker,"complete")
        threading.Event().wait(.06)
        analyze.assert_called_once_with("frame")

    def test_next_turn_and_manual_key_each_start_one_batch(self):
        visible = threading.Event()
        visible.set()
        analyze = Mock(return_value=AnalysisResult())
        worker = self.worker(analyze,lambda _: visible.is_set())
        worker.latch.release_after = .02
        worker.start()
        self.wait_for(worker,"complete")
        worker.refresh()
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,2)
        visible.clear()
        threading.Event().wait(.05)
        visible.set()
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,3)

    def test_halo_retry_stops_as_soon_as_successful(self):
        analyze = Mock(side_effect=[AnalysisResult(error="没有蓝圈",error_kind="self_missing"),
                                   AnalysisResult(error="没有蓝圈",error_kind="self_missing"),
                                   AnalysisResult()])
        worker = self.worker(analyze)
        worker.start()
        result = self.wait_for(worker,"complete")
        self.assertEqual(result.attempts,3)
        threading.Event().wait(.05)
        self.assertEqual(analyze.call_count,3)

    def test_five_failed_attempts_wait_for_manual_or_next_turn(self):
        analyze = Mock(return_value=AnalysisResult(error="没有蓝圈",error_kind="self_missing"))
        worker = self.worker(analyze)
        worker.start()
        result = self.wait_for(worker,"failed")
        self.assertEqual(result.attempts,5)
        threading.Event().wait(.06)
        self.assertEqual(analyze.call_count,5)

    def test_parameter_failure_is_not_retried(self):
        analyze = Mock(return_value=AnalysisResult(error="角度为空",error_kind="parameters"))
        worker = self.worker(analyze)
        worker.start()
        self.wait_for(worker,"failed")
        analyze.assert_called_once()

    def test_pause_interrupts_retries_and_discards_results(self):
        entered = threading.Event()
        def analyze(_):
            entered.set()
            return AnalysisResult(error="没有蓝圈",error_kind="self_missing")
        worker = self.worker(Mock(side_effect=analyze))
        worker.retry_delay = .2
        worker.start()
        self.assertTrue(entered.wait(1))
        worker.pause()
        threading.Event().wait(.04)
        worker.analyze.assert_called_once()
        self.assertIsNone(worker.take_result())

    def test_manual_batch_works_while_paused(self):
        analyze = Mock(return_value=AnalysisResult())
        worker = self.worker(analyze)
        worker.pause()
        worker.start()
        worker.refresh()
        self.wait_for(worker,"complete")
        self.assertFalse(worker.running)
        analyze.assert_called_once()

    def test_repeated_manual_requests_during_batch_do_not_queue_more(self):
        entered, release = threading.Event(),threading.Event()
        def analyze(_):
            entered.set()
            release.wait(1)
            return AnalysisResult()
        worker = self.worker(Mock(side_effect=analyze))
        worker.start()
        self.assertTrue(entered.wait(1))
        for _ in range(10):
            worker.refresh()
        release.set()
        self.wait_for(worker,"complete")
        threading.Event().wait(.05)
        worker.analyze.assert_called_once()

    def test_current_turn_changes_recompute_but_unchanged_inputs_do_not(self):
        visible, active = threading.Event(), threading.Event()
        visible.set()
        active.set()
        inputs = {"state": InputState(65,80,25)}
        analyze = Mock(side_effect=lambda _: AnalysisResult(input_state=inputs["state"]))
        probe = Mock(side_effect=lambda _: inputs["state"])
        worker = self.worker(analyze,lambda _: visible.is_set(),input_reader=probe,
                             active_detector=lambda _: active.is_set())
        worker.start()
        self.wait_for(worker,"complete")
        visible.clear()
        threading.Event().wait(.07)
        self.assertGreater(probe.call_count,1)
        analyze.assert_called_once()
        inputs["state"] = InputState(66,80,25)
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,2)
        threading.Event().wait(.07)
        self.assertEqual(analyze.call_count,2)
        inputs["state"] = InputState(66,84,25)
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,3)
        active.clear()
        threading.Event().wait(.04)
        inputs["state"] = InputState(67,84,25)
        threading.Event().wait(.07)
        self.assertEqual(analyze.call_count,3)
        worker.pause()
        worker.refresh()
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,4)

    def test_failed_change_does_not_repeat_and_manual_retry_remains_available(self):
        state = {"degree":65}
        analyze = Mock(side_effect=[AnalysisResult(input_state=InputState(65,80,25)),
                                   AnalysisResult(error="参数失败",error_kind="parameters"),
                                   AnalysisResult(input_state=InputState(66,80,25))])
        worker = self.worker(analyze,input_reader=lambda _: InputState(state["degree"],80,25),
                             active_detector=lambda _: True)
        worker.start()
        self.wait_for(worker,"complete")
        state["degree"] = 66
        self.wait_for(worker,"failed")
        threading.Event().wait(.07)
        self.assertEqual(analyze.call_count,2)
        worker.refresh()
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,3)

    def test_turn_end_stops_probing_and_next_turn_uses_fresh_baseline(self):
        active, visible = threading.Event(),threading.Event()
        active.set()
        visible.set()
        probe = Mock(return_value=InputState(65,80,25))
        analyze = Mock(return_value=AnalysisResult(input_state=InputState(65,80,25)))
        worker = self.worker(analyze,lambda _: visible.is_set(),input_reader=probe,
                             active_detector=lambda _: active.is_set())
        worker.latch.release_after = .02
        worker.start()
        self.wait_for(worker,"complete")
        active.clear()
        visible.clear()
        threading.Event().wait(1.07)
        self.assertFalse(worker._turn_open)
        count = probe.call_count
        threading.Event().wait(.04)
        self.assertEqual(probe.call_count,count)
        active.set()
        visible.set()
        self.wait_for(worker,"complete")
        self.assertEqual(analyze.call_count,2)


if __name__ == "__main__":
    unittest.main()
