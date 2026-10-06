"""Docking lifecycle and real own-process Win32 hooks; never send game input."""

import ctypes
from ctypes import wintypes
from copy import deepcopy
import os
import tkinter
import unittest
from unittest.mock import Mock, patch

from sidecar_window import (SidecarController, WindowsPanel, sidecar_settings,
                            dock_rectangle, KEEP_LIT_NOTE, VERIFIED_KEEP_LIT_NOTE)


class DockGeometryTests(unittest.TestCase):
    def test_adjacent_rectangle_and_negative_monitor_coordinates(self):
        self.assertEqual(dock_rectangle((50, 40, 1050, 740), (0, 0, 1920, 1040), 440),
                         (1050, 40, 440, 700))
        self.assertEqual(dock_rectangle((-1800, -850, -800, -150), (-1920, -1080, 0, 0), 550),
                         (-800, -850, 550, 700))

    def test_clipped_height_right_edge_and_tiny_hall_are_rejected(self):
        for bounds in ((50, 40, 1500, 740), (50, -20, 1050, 740),
                       (50, 40, 1050, 1100), (50, 40, 1050, 250)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                dock_rectangle(bounds, (0, 0, 1920, 1040), 440)

    def test_settings_accept_legacy_config_and_do_not_mutate_custom_fields(self):
        config = {"region": [1, 2, 3, 4], "sidecar": {"enabled": True, "width": 550, "custom": "keep"}}
        old = deepcopy(config)
        self.assertEqual(sidecar_settings(config), {"enabled": True, "independent_focus": False, "width": 550})
        self.assertEqual(config, old)
        for malformed in ({}, {"sidecar": None}, {"sidecar": "invalid"},
                          {"sidecar": {"enabled": "true", "width": "bad"}}):
            self.assertEqual(sidecar_settings(malformed), {"enabled": False, "independent_focus": False, "width": 440})
        self.assertEqual(sidecar_settings({"sidecar": {"width": 1000}})["width"], 640)
        self.assertEqual(sidecar_settings({"sidecar": {"width": True}})["width"], 440)


class SidecarLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.hall = {"hwnd": 100, "pid": 900, "class_name": "36JBCOM_Browser",
                     "visible": True, "minimized": False, "bounds": (50, 40, 1050, 740)}
        self.flash = {"hwnd": 200, "pid": 900, "class_name": "MacromediaFlashPlayerActiveX",
                      "visible": True, "minimized": False, "bounds": (50, 40, 1050, 640)}
        self.panel_state = {"hwnd": 300, "pid": 901, "class_name": "TkTopLevel",
                            "visible": True, "minimized": False, "bounds": (10, 20, 1130, 1020)}
        self.states = {100: self.hall, 200: self.flash, 300: self.panel_state}
        self.config = {"region": (50, 40, 1000, 600), "custom": "keep",
                       "window": {k: self.hall[k] for k in ("hwnd", "pid", "class_name")}}
        self.config["window"]["content"] = {k: self.flash[k] for k in ("hwnd", "pid", "class_name")}
        self.panel = Mock(hwnd=300)
        self.clients = self.panel.clients
        self.clients.window_state.side_effect = self.states.get
        self.clients.root.return_value = 100
        self.clients.work_area.return_value = (0, 0, 1920, 1040)
        self.panel.width_pixels.side_effect = lambda width: width
        self.panel.snapshot.return_value = (10, 20, 1130, 1020)
        self.panel.needs_place.return_value = False
        self.layout, self.status, self.invalid = Mock(), Mock(), Mock()
        self.focus_layout = Mock()
        self.controller = SidecarController(self.panel, self.layout, self.status, self.invalid,
                                             on_focus_layout=self.focus_layout)
        patcher = patch("sidecar_window.resolve_region", return_value=self.config["region"])
        self.resolve = patcher.start()
        self.addCleanup(patcher.stop)

    def connect(self):
        self.controller.set_enabled(True, self.config)

    def test_connect_move_resize_and_width_change_never_activate_hall(self):
        old = deepcopy(self.config)
        self.connect()
        self.panel.place.assert_called_once_with((1050, 40, 440, 700))
        self.layout.assert_called_once_with(True)
        self.assertIn(VERIFIED_KEEP_LIT_NOTE, self.status.call_args.args[0])
        self.controller.poll(self.config)
        self.panel.place.assert_called_once()
        self.hall["bounds"] = (70, 80, 1070, 880)
        self.controller.poll(self.config)
        self.panel.place.assert_called_with((1070, 80, 440, 800))
        self.config["sidecar"] = {"width": 520}
        self.controller.poll(self.config)
        self.panel.place.assert_called_with((1070, 80, 520, 800))
        self.clients.activate.assert_not_called()
        self.clients.focus_control.assert_not_called()
        self.assertEqual({k: self.config[k] for k in old}, old)

    def test_minimize_hide_once_and_restore_without_activation(self):
        self.connect()
        self.hall["minimized"] = True
        self.controller.poll(self.config)
        self.controller.poll(self.config)
        self.panel.hide.assert_called_once()
        self.assertTrue(self.controller.linked)
        self.hall["minimized"] = False
        self.controller.poll(self.config)
        self.panel.show.assert_called_once()
        self.clients.activate.assert_not_called()

    def test_dragged_panel_snaps_back_without_moving_hall(self):
        self.connect()
        self.panel.needs_place.return_value = True
        self.controller.poll(self.config)
        self.assertEqual(self.panel.place.call_count, 2)
        self.panel.place.assert_called_with((1050, 40, 440, 700))
        self.clients.activate.assert_not_called()

    def test_closing_minimized_hall_also_detaches_and_invalidates_once(self):
        self.connect()
        self.hall["minimized"] = True
        self.controller.poll(self.config)
        del self.states[100]
        self.controller.poll(self.config)
        self.controller.poll(self.config)
        self.assertFalse(self.controller.linked)
        self.invalid.assert_called_once()
        self.panel.disable.assert_called_once()
        self.panel.restore.assert_called_once_with((10, 20, 1130, 1020))

    def test_reused_hall_and_reloaded_or_reparented_flash_stop_docking(self):
        for field in ("hall_pid", "flash_pid", "flash_root"):
            with self.subTest(field=field):
                self.setUp()
                self.connect()
                if field == "hall_pid":
                    self.hall["pid"] += 1
                elif field == "flash_pid":
                    self.flash["pid"] += 1
                else:
                    self.clients.root.return_value = 999
                self.controller.poll(self.config)
                self.assertFalse(self.controller.linked)
                self.invalid.assert_called_once()

    def test_full_screen_does_not_overlay_and_retries_after_manual_move(self):
        self.hall["bounds"] = (0, 0, 1920, 1040)
        self.connect()
        self.assertFalse(self.controller.linked)
        self.panel.enable.assert_not_called()
        self.panel.place.assert_not_called()
        self.invalid.assert_not_called()
        self.assertIn("右侧空间不足", self.status.call_args.args[0])
        self.hall["bounds"] = (50, 40, 1050, 740)
        self.controller.poll(self.config)
        self.assertTrue(self.controller.linked)

    def test_space_lost_restores_independent_window_and_reconnects(self):
        self.connect()
        self.hall["bounds"] = (800, 40, 1800, 740)
        self.controller.poll(self.config)
        self.panel.restore.assert_called_once_with((10, 20, 1130, 1020))
        self.assertFalse(self.controller.linked)
        self.hall["bounds"] = (50, 40, 1050, 740)
        self.controller.poll(self.config)
        self.assertTrue(self.controller.linked)

    def test_detach_restores_only_panel_and_close_does_not_show_it(self):
        self.connect()
        self.controller.set_enabled(False, self.config)
        self.controller.poll(self.config)
        self.panel.restore.assert_called_once_with((10, 20, 1130, 1020))
        self.layout.assert_called_with(False)
        self.panel.enable.assert_called_once()
        self.connect()
        self.panel.restore.reset_mock()
        self.controller.close()
        self.panel.restore.assert_not_called()

    def test_legacy_binding_does_not_invalidate_its_existing_analysis(self):
        self.config["window"].pop("content")
        self.connect()
        self.assertFalse(self.controller.linked)
        self.invalid.assert_not_called()
        self.panel.enable.assert_not_called()

    def test_native_failure_unwinds_layout_without_invalidating_valid_game(self):
        self.panel.enable.side_effect = OSError("not available")
        self.connect()
        self.panel.disable.assert_called_once()
        self.panel.restore.assert_called_once()
        self.layout.assert_called_with(False)
        self.invalid.assert_not_called()

    def test_probe_returns_foreground_focus_and_cursor_facts_without_activation(self):
        self.clients.foreground.return_value = 100
        self.clients.keyboard_focus.return_value = 200
        self.clients.has_keyboard_focus.return_value = True
        self.clients.cursor_position.return_value = (1200, 300)
        self.clients.window_at.return_value = 300
        facts = self.controller.focus_probe(self.config)
        self.assertTrue(facts["hall_foreground"])
        self.assertTrue(facts["flash_focus"])
        self.assertNotIn("keep_lit_supported", facts)
        self.clients.activate.assert_not_called()

    def test_other_hall_does_not_claim_verified_keep_lit(self):
        self.hall["class_name"] = self.config["window"]["class_name"] = "OtherFlashHost"
        self.connect()
        self.assertIn(KEEP_LIT_NOTE, self.status.call_args.args[0])

    def test_independent_focus_does_not_move_resize_or_activate_either_window(self):
        self.config["sidecar"] = {"independent_focus": True}
        self.controller.set_enabled(False, self.config)
        self.controller.poll(self.config)
        self.assertTrue(self.controller.independent_focus)
        self.assertFalse(self.controller.linked)
        self.panel.enable.assert_called_once()
        self.focus_layout.assert_called_once_with(True)
        self.panel.place.assert_not_called()
        self.panel.restore.assert_not_called()
        self.layout.assert_not_called()
        self.clients.activate.assert_not_called()
        self.resolve.assert_not_called()

    def test_turning_off_independent_focus_restores_normal_activation(self):
        self.config["sidecar"] = {"independent_focus": True}
        self.controller.set_enabled(False, self.config)
        self.config["sidecar"]["independent_focus"] = False
        self.controller.set_enabled(False, self.config)
        self.panel.disable.assert_called_once()
        self.assertFalse(self.controller.independent_focus)
        self.focus_layout.assert_called_with(False)

    def test_docking_and_undocking_reinstall_independent_focus_after_wrapper_restore(self):
        self.config["sidecar"] = {"independent_focus": True}
        self.controller.set_enabled(False, self.config)
        self.connect()
        self.assertTrue(self.controller.linked)
        self.assertFalse(self.controller.independent_focus)
        self.controller.set_enabled(False, self.config)
        self.assertFalse(self.controller.linked)
        self.assertTrue(self.controller.independent_focus)
        self.panel.restore.assert_called_once()
        self.assertEqual(self.panel.enable.call_count, 3)
        self.focus_layout.assert_called_with(True)

    def test_failed_independent_focus_unwinds_and_retries_only_on_user_change(self):
        self.config["sidecar"] = {"independent_focus": True}
        self.panel.enable.side_effect = OSError("failed")
        self.controller.set_enabled(False, self.config)
        self.controller.poll(self.config)
        self.panel.enable.assert_called_once()
        self.panel.disable.assert_called_once()
        self.focus_layout.assert_called_with(False)
        self.panel.enable.side_effect = None
        self.controller.set_enabled(False, self.config)
        self.assertTrue(self.controller.independent_focus)

    def test_closing_independent_focus_restores_handler_without_showing_window(self):
        self.config["sidecar"] = {"independent_focus": True}
        self.controller.set_enabled(False, self.config)
        self.controller.close()
        self.panel.disable.assert_called_once()
        self.panel.restore.assert_not_called()
        self.assertFalse(self.controller.independent_focus)


@unittest.skipUnless(os.name == "nt", "Win32 integration")
class NativePanelTests(unittest.TestCase):
    def test_disabling_from_inside_callback_defers_release_until_it_returns(self):
        root = tkinter.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        panel = WindowsPanel(root)
        self.addCleanup(panel.disable)
        panel.enable(lambda *_: panel.disable())
        hwnd = panel.hwnd
        send = panel.api.SendMessageW
        send.argtypes, send.restype = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], ctypes.c_ssize_t
        send(hwnd, panel.WM_MOUSEWHEEL, 120 << 16, 0)
        self.assertIsNone(panel._callback)
        self.assertEqual(len(panel._retired_callbacks), 1)
        panel.release_callbacks()
        self.assertEqual(panel._retired_callbacks, [])

    def test_native_placement_survives_tk_geometry_pass_and_restores_bounds(self):
        root = tkinter.Tk()
        root.geometry("1120x1000+30000+30000")
        root.update()
        self.addCleanup(root.destroy)
        panel = WindowsPanel(root)
        self.addCleanup(panel.disable)
        original = panel.snapshot()
        root.minsize(300, 320)
        root.resizable(False, False)
        panel.enable()
        requested = (30000, 30000, 440, 700)
        panel.place(requested)
        root.update()
        self.assertFalse(panel.needs_place(requested), panel.clients.window_state(panel.hwnd))
        panel.disable()
        root.resizable(True, True)
        panel.restore(original)
        root.update()
        left, top, right, bottom = original
        self.assertFalse(panel.needs_place((left, top, right-left, bottom-top)))

    def test_real_own_window_hook_and_style_are_reversible_and_do_not_activate(self):
        root = tkinter.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        panel = WindowsPanel(root)
        self.addCleanup(panel.disable)
        hwnd = panel._handle()
        original_style = panel.get_long(hwnd, panel.GWL_EXSTYLE)
        original_proc = panel.get_long(hwnd, panel.GWLP_WNDPROC)
        foreground = panel.clients.foreground()
        wheel = Mock()
        panel.enable(wheel)
        self.assertTrue(panel.get_long(hwnd, panel.GWL_EXSTYLE) & panel.NOACTIVATE)
        self.assertNotEqual(panel.get_long(hwnd, panel.GWLP_WNDPROC), original_proc)
        send = panel.api.SendMessageW
        send.argtypes, send.restype = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], ctypes.c_ssize_t
        self.assertEqual(send(hwnd, panel.WM_MOUSEACTIVATE, 0, 0), panel.MA_NOACTIVATE)
        send(hwnd, panel.WM_MOUSEWHEEL, 120 << 16, 0)
        wheel.assert_called_once_with(120, False)
        self.assertEqual(panel.clients.foreground(), foreground)
        panel.disable()
        panel.disable()
        self.assertEqual(panel.get_long(hwnd, panel.GWL_EXSTYLE), original_style)
        self.assertEqual(panel.get_long(hwnd, panel.GWLP_WNDPROC), original_proc)


if __name__ == "__main__":
    unittest.main()
