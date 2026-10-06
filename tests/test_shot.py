"""Explicit shots, shortcut timing and input validation without key injection."""

import threading
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import tkinter
from types import SimpleNamespace
from queue import Queue

from shot import parse_force, ShotController, ShotStateChanged
from test_auto_recognition import load_functions


class NumericShotTests(unittest.TestCase):
    def controller(self):
        controller = ShotController(Mock(),Mock(),Mock(),Mock(),Mock())
        self.addCleanup(controller.close)
        return controller

    def test_decimal_input_and_invalid_values(self):
        self.assertEqual(parse_force(" 64.25 "),64.25)
        self.assertEqual(parse_force("100"),100)
        for text in ("","zero","0","-1","100.01","nan","inf","-inf"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_force(text)

    def test_precise_decimal_hold_and_one_release_after_focus_verification(self):
        controller = self.controller()
        cancel = Mock(wait=Mock(return_value=False),is_set=Mock(return_value=False))
        controller._run(64.25,"game",cancel)
        self.assertEqual([call.args[0] for call in cancel.wait.call_args_list],[.1,.15,2.57])
        controller.focus.assert_called_once_with("game")
        controller.verify.assert_called_once_with("game")
        controller.press.assert_called_once()
        controller.release.assert_called_once()

    def test_numeric_hold_excludes_keyboard_library_post_press_pause(self):
        clock, timestamps = [0], {}
        def key_down(key,_pause):
            timestamps["down"] = clock[0]
            clock[0] += .1 if _pause else 0
        def key_up(key,_pause):
            timestamps["up"] = clock[0]
        wrappers = load_functions("km.py",{"space_press","space_release"},{
            "pyautogui":SimpleNamespace(keyDown=key_down,keyUp=key_up),
        })
        controller = self.controller()
        controller.press = lambda: wrappers["space_press"](pause=False)
        controller.release = lambda: wrappers["space_release"](pause=False)
        def wait(duration):
            clock[0] += duration
            return False
        cancel = Mock(wait=Mock(side_effect=wait),is_set=Mock(return_value=False))
        controller._run(64.25,"game",cancel)
        self.assertAlmostEqual(timestamps["up"]-timestamps["down"],2.57)

    def test_focus_failure_and_lost_focus_do_not_press_space(self):
        for operation in ("focus","verify"):
            with self.subTest(operation=operation):
                controller = self.controller()
                getattr(controller,operation).side_effect = ValueError("游戏失焦")
                cancel = Mock(wait=Mock(return_value=False),is_set=Mock(return_value=False))
                controller._run(30,"game",cancel)
                controller.press.assert_not_called()
                self.assertIn("失败",controller.notify.call_args.args[0])

    def test_cancel_before_press_and_release_on_hold_failure(self):
        controller = self.controller()
        cancel = Mock(wait=Mock(return_value=True))
        controller._run(30,"game",cancel)
        controller.press.assert_not_called()
        cancel = Mock(wait=Mock(side_effect=[False,False,RuntimeError("interrupted")]),
                      is_set=Mock(return_value=False))
        controller._run(30,"game",cancel)
        controller.press.assert_called_once()
        controller.release.assert_called_once()

    def test_busy_requests_do_not_pile_up_and_cancel_stays_responsive(self):
        controller = self.controller()
        entered,release,finished = threading.Event(),threading.Event(),threading.Event()
        def focus(_):
            entered.set()
            release.wait(1)
        controller.focus.side_effect = focus
        controller.notify.side_effect = lambda _: finished.set()
        self.assertTrue(controller.submit(30,"game"))
        self.assertTrue(entered.wait(1))
        self.assertFalse(controller.submit(40,"game"))
        controller.cancel()
        release.set()
        self.assertTrue(finished.wait(1))
        controller.press.assert_not_called()
        self.assertFalse(controller.busy)

    def test_close_releases_space_before_shutdown_and_blocks_new_shots(self):
        controller = self.controller()
        pressed = threading.Event()
        controller.press.side_effect = pressed.set
        self.assertTrue(controller.submit(100,"game"))
        self.assertTrue(pressed.wait(1))
        controller.close()
        controller.release.assert_called_once()
        self.assertFalse(controller.busy)
        self.assertFalse(controller.submit(20,"game"))

    def test_press_error_still_releases_space(self):
        controller = self.controller()
        controller.press.side_effect = RuntimeError("key down error")
        cancel = Mock(wait=Mock(return_value=False),is_set=Mock(return_value=False))
        controller._run(30,"game",cancel)
        controller.release.assert_called_once()


class DirectionalShotTests(unittest.TestCase):
    def controller(self):
        controller = ShotController(Mock(), Mock(), Mock(), Mock(), Mock(),
                                    press_direction=Mock(), release_direction=Mock())
        self.addCleanup(controller.close)
        return controller

    def test_both_directions_are_released_and_focus_rechecked_before_space(self):
        for direction in ("left", "right"):
            with self.subTest(direction=direction):
                controller = self.controller()
                order = Mock()
                for name in ("focus", "verify", "press_direction", "release_direction", "press", "release"):
                    order.attach_mock(getattr(controller, name), name)
                cancel = Mock(wait=Mock(return_value=False), is_set=Mock(return_value=False))
                controller._run(78.93, "game", cancel, direction)
                self.assertEqual(order.mock_calls, [
                    unittest.mock.call.focus("game"), unittest.mock.call.verify("game"),
                    unittest.mock.call.press_direction(direction), unittest.mock.call.release_direction(direction),
                    unittest.mock.call.verify("game"), unittest.mock.call.press(), unittest.mock.call.release(),
                ])
                waits = [call.args[0] for call in cancel.wait.call_args_list]
                self.assertEqual(waits[:3], [.1, .15, .1])
                self.assertAlmostEqual(waits[3], 78.93*.04)

    def test_cancel_during_direction_or_settling_does_not_press_space(self):
        for waits in ([False, False, True], [False, False, False, True]):
            with self.subTest(waits=waits):
                controller = self.controller()
                controller.check_state = Mock()
                cancel = Mock(wait=Mock(side_effect=waits), is_set=Mock(return_value=False))
                controller._run(30, "game", cancel, "left")
                controller.release_direction.assert_called_once_with("left")
                controller.press.assert_not_called()

    def test_direction_press_or_hold_failure_still_releases_direction(self):
        for operation in ("press", "wait"):
            with self.subTest(operation=operation):
                controller = self.controller()
                if operation == "press":
                    controller.press_direction.side_effect = RuntimeError("key error")
                    waits = [False, False]
                else:
                    waits = [False, False, RuntimeError("hold error")]
                controller._run(30, "game", Mock(wait=Mock(side_effect=waits),
                                                is_set=Mock(return_value=False)), "right")
                controller.release_direction.assert_called_once_with("right")
                controller.press.assert_not_called()
                self.assertIn("失败", controller.notify.call_args.args[0])

    def test_focus_loss_before_direction_or_after_turning_stops_shot(self):
        for failed_check in (1, 2):
            with self.subTest(failed_check=failed_check):
                controller = self.controller()
                controller.verify.side_effect = ([ValueError("失焦")] if failed_check == 1
                                                 else [None, ValueError("失焦")])
                controller._run(30, "game", Mock(wait=Mock(return_value=False),
                                                is_set=Mock(return_value=False)), "left")
                controller.press.assert_not_called()
                if failed_check == 1:
                    controller.press_direction.assert_not_called()
                else:
                    controller.release_direction.assert_called_once_with("left")

    def test_submit_freezes_nested_binding_and_rejects_invalid_direction(self):
        controller = self.controller()
        entered, proceed = threading.Event(), threading.Event()
        received = []
        def focus(target):
            entered.set()
            proceed.wait(1)
            received.append(target["window"]["hwnd"])
        controller.focus.side_effect = focus
        target = {"window": {"hwnd": 123}}
        self.assertTrue(controller.submit(1, target, direction="right"))
        self.assertTrue(entered.wait(1))
        try:
            target["window"]["hwnd"] = 456
            self.assertFalse(controller.submit(1, target, direction="left"))
            with self.assertRaises(ValueError):
                controller.submit(1, target, direction="up")
            controller.cancel()
        finally:
            proceed.set()
            controller.close()
        self.assertEqual(received, [123])
        controller.press.assert_not_called()

    def test_direction_wrappers_disable_default_keyboard_pause(self):
        keys = Mock()
        wrappers = load_functions("km.py", {"direction_press", "direction_release"}, {"pyautogui": keys})
        wrappers["direction_press"]("left")
        wrappers["direction_release"]("left")
        keys.keyDown.assert_called_once_with("left", _pause=False)
        keys.keyUp.assert_called_once_with("left", _pause=False)

    def test_atomic_tap_is_checked_before_turning_and_twice_after_release(self):
        controller = self.controller()
        controller.tap_direction = Mock()
        controller.check_state = Mock()
        order = Mock()
        for name in ("check_state", "tap_direction", "press"):
            order.attach_mock(getattr(controller, name), name)
        cancel = Mock(wait=Mock(return_value=False), is_set=Mock(return_value=False))
        controller._run(30, "game", cancel, "left")
        self.assertEqual(order.mock_calls, [unittest.mock.call.check_state("game"),
                         unittest.mock.call.tap_direction("left"),
                         unittest.mock.call.check_state("game"),
                         unittest.mock.call.check_state("game"), unittest.mock.call.press()])
        controller.press_direction.assert_not_called()
        controller.release_direction.assert_not_called()
        self.assertEqual([c.args[0] for c in cancel.wait.call_args_list], [.1, .15, .1, .05, 1.2])

    def test_changed_state_before_turning_or_after_turning_prevents_space_and_refreshes(self):
        for failed_check in (1, 2, 3):
            with self.subTest(failed_check=failed_check):
                controller = self.controller()
                controller.tap_direction = Mock()
                controller.check_state = Mock(side_effect=[None]*(failed_check-1)+[ShotStateChanged("角度变化")])
                controller.on_state_changed = Mock()
                controller._run(30, "game", Mock(wait=Mock(return_value=False),
                                is_set=Mock(return_value=False)), "right")
                controller.press.assert_not_called()
                controller.on_state_changed.assert_called_once()
                self.assertIn("发射已停止", controller.notify.call_args.args[0])
                if failed_check == 1:
                    controller.tap_direction.assert_not_called()

    def test_no_wait_occurs_between_legacy_arrow_down_and_up(self):
        controller = self.controller()
        order = Mock()
        order.attach_mock(controller.press_direction, "down")
        order.attach_mock(controller.release_direction, "up")
        wait = Mock(return_value=False)
        order.attach_mock(wait, "wait")
        controller._run(30, "game", Mock(wait=wait, is_set=Mock(return_value=False)), "left")
        calls = order.mock_calls
        down = calls.index(unittest.mock.call.down("left"))
        self.assertEqual(calls[down+1], unittest.mock.call.up("left"))

    def test_atomic_tap_failure_stops_before_space(self):
        controller = self.controller()
        controller.tap_direction = Mock(side_effect=OSError("换向按键未完整发送"))
        controller._run(30, "game", Mock(wait=Mock(return_value=False),
                        is_set=Mock(return_value=False)), "left")
        controller.press.assert_not_called()
        controller.press_direction.assert_not_called()
        self.assertIn("换向按键未完整发送", controller.notify.call_args.args[0])


class AtomicDirectionTests(unittest.TestCase):
    def setUp(self):
        self.tap = load_functions("km.py", {"direction_tap"}, {})["direction_tap"]

    def test_both_events_are_queued_in_one_call_with_native_input_size(self):
        import ctypes
        for direction, vk in (("left", 0x25), ("right", 0x27)):
            recorded = []
            def send(count, events, size):
                recorded.append((count, [(e.type, e.ki.wVk, e.ki.dwFlags) for e in events], size))
                return count
            self.tap(direction, api=SimpleNamespace(SendInput=Mock(side_effect=send)))
            self.assertEqual(recorded, [(2, [(1, vk, 1), (1, vk, 3)],
                                        40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)])

    def test_partial_send_releases_arrow_and_never_reports_success(self):
        events = []
        def send(count, inputs, size):
            events.append((count, inputs[0].ki.dwFlags))
            return 1
        with self.assertRaisesRegex(OSError, "未完整发送"):
            self.tap("left", api=SimpleNamespace(SendInput=Mock(side_effect=send)))
        self.assertEqual(events, [(2, 1), (1, 3)])

    def test_blocked_send_and_invalid_keys_are_rejected(self):
        api = SimpleNamespace(SendInput=Mock(return_value=0))
        with self.assertRaises(OSError):
            self.tap("right", api=api)
        api.SendInput.assert_called_once()
        with self.assertRaises(ValueError):
            self.tap("up", api=api)


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.state = load_functions("main.py",{"handle_inputs","close_force_dialog"},{
            "_last_s_press":None,"_force_dialog":None,"_force_dialog_pending":False,
            "_region_overlay":None,"_window_drag":None,"_shot_controller":None,"_ui_actions":Queue(),
            "_ui": {},
            "_dialog_escape_until":0,"tkinter":tkinter,
            "time":Mock(monotonic=Mock(return_value=1)),"_fire_cancel":Mock(),
            "_analysis_worker":Mock(),"_mode":"auto","AUTO_MODE":"auto",
            "reset_inputs":Mock(),"fire":Mock(),"logger":Mock(),
        })

    def press(self,key,now):
        self.state["time"].monotonic.return_value = now
        self.state["handle_inputs"](key)

    def test_two_s_taps_open_once_without_firing_or_changing_analysis_mode(self):
        self.press("s",1)
        self.assertTrue(self.state["_ui_actions"].empty())
        self.press("s",1.3)
        self.assertEqual(self.state["_ui_actions"].get_nowait(),("force_dialog",None))
        for key in "sss30enter":
            self.press(key,1.4)
        self.assertTrue(self.state["_ui_actions"].empty())
        self.state["fire"].assert_not_called()
        self.assertEqual(self.state["_mode"],"auto")

    def test_slow_or_nonconsecutive_taps_and_region_selection_do_not_open(self):
        for keys in (("s","s"),("s","x","s")):
            self.state["_last_s_press"] = None
            for i,key in enumerate(keys):
                self.press(key,1+i)
            self.assertTrue(self.state["_ui_actions"].empty())
        self.state["_region_overlay"] = Mock()
        self.press("s",5)
        self.press("s",5.1)
        self.assertTrue(self.state["_ui_actions"].empty())

    def test_escape_cancels_pending_dialog_before_it_opens(self):
        self.press("s",1)
        self.press("s",1.1)
        self.press("esc",1.2)
        self.assertFalse(self.state["_force_dialog_pending"])
        self.assertEqual(self.state["_ui_actions"].get_nowait(),("force_dialog",None))
        self.assertEqual(self.state["_ui_actions"].get_nowait(),("cancel_force",None))

    def test_s_hold_autorepeat_requires_actual_release_for_second_tap(self):
        state = load_functions("km.py",{"on_press","on_release"},{
            "_s_down":False,"_queue":Queue(),"keyboard":Mock(),
        })
        event = SimpleNamespace(char="s")
        for _ in range(5):
            state["on_press"](event)
        self.assertEqual(state["_queue"].qsize(),1)
        state["on_release"](event)
        state["on_press"](event)
        self.assertEqual(state["_queue"].qsize(),2)

    def test_local_dialog_escape_is_not_processed_again_as_global_pause(self):
        dialog = self.state["_force_dialog"] = Mock()
        self.state["time"].monotonic.return_value = 1
        self.state["close_force_dialog"](SimpleNamespace(keysym="Escape"))
        self.press("esc",1.1)
        self.assertTrue(self.state["_ui_actions"].empty())
        dialog.destroy.assert_called_once()
        self.press("esc",1.5)
        self.assertEqual(self.state["_ui_actions"].get_nowait(),("paused",None))


class ForceDialogTests(unittest.TestCase):
    def setUp(self):
        import main
        import tkinter as tk
        self.main = main
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.controller = Mock(busy=False)
        self.controller.submit.return_value = True
        self.target = {"window":{"hwnd":123}}
        self.status = tk.StringVar(master=self.root)
        self.patcher = patch.multiple(main,create=True,_tk=self.root,_ui={"status":self.status},
                                      _force_dialog=None,_force_dialog_pending=True,
                                      _last_s_press=None,_region_overlay=None,
                                      _shot_controller=self.controller,
                                      shot_target=Mock(return_value=self.target),reset_inputs=Mock())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(main.close_force_dialog)
        original = tk.Toplevel
        def hidden_dialog(*args,**kwargs):
            dialog = original(*args,**kwargs)
            dialog.withdraw()
            return dialog
        self.dialog_patch = patch.object(tk,"Toplevel",side_effect=hidden_dialog)
        self.dialog_patch.start()
        self.addCleanup(self.dialog_patch.stop)

    def widgets(self,widget):
        return [child for item in widget.winfo_children()
                for child in [item]+self.widgets(item)]

    def test_invalid_input_then_decimal_confirmation_submits_once(self):
        from tkinter import ttk
        self.main.open_force_dialog()
        dialog = self.main._force_dialog
        self.assertFalse(dialog.winfo_viewable())
        widgets = self.widgets(dialog)
        entry = next(w for w in widgets if isinstance(w,ttk.Entry))
        button = next(w for w in widgets if isinstance(w,ttk.Button) and w.cget("text") == "发射")
        self.assertTrue(dialog.bind("<Return>"))
        entry.insert(0,"nan")
        button.invoke()
        self.controller.submit.assert_not_called()
        self.assertIs(self.main._force_dialog,dialog)
        entry.delete(0,"end")
        entry.insert(0,"64.25")
        button.invoke()
        self.controller.submit.assert_called_once_with(64.25,self.target)
        self.assertIsNone(self.main._force_dialog)
        self.assertIn("64.25",self.status.get())

    def test_cancel_and_repeat_open_do_not_fire_or_leave_modal_grab(self):
        self.main.open_force_dialog()
        dialog = self.main._force_dialog
        self.main.open_force_dialog()
        self.assertIs(self.main._force_dialog,dialog)
        self.assertTrue(dialog.bind("<Escape>"))
        self.main.close_force_dialog()
        self.main.close_force_dialog()
        self.assertIsNone(self.root.grab_current())
        self.controller.submit.assert_not_called()

    def test_closed_game_reports_reason_without_opening_or_shooting(self):
        self.main.shot_target.side_effect = ValueError("游戏已关闭")
        self.main.open_force_dialog()
        self.assertIsNone(self.main._force_dialog)
        self.assertEqual(self.status.get(),"游戏已关闭")
        self.controller.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
