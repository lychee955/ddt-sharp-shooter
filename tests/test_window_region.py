"""Window/client/screen coordinate checks without touching desktop windows."""

import os
import unittest
from unittest.mock import Mock

from window_region import bind_region, bind_window_at, resolve_region, shot_target, focus_shot_target, verify_shot_target


class WindowRegionTests(unittest.TestCase):
    def setUp(self):
        self.info = {"hwnd":123,"pid":os.getpid()+1,"class_name":"GameHall",
                     "client":(1000,400,2100,1350)}
        self.clients = Mock()
        self.clients.windows.return_value = [123]
        self.clients.info.return_value = self.info
        self.region = (1020,440,2052,1198)
        self.binding = bind_region(self.region,self.clients)

    def test_offsets_exclude_toolbar_and_reference_table(self):
        self.assertEqual(self.binding["offset"],[20,40,2052,1198])
        self.assertEqual(self.binding["client_size"],[2100,1350])
        self.assertEqual(resolve_region({"window":self.binding},self.clients),self.region)

    def test_moved_window_changes_only_screen_origin(self):
        self.info["client"] = (300,200,2100,1350)
        self.assertEqual(resolve_region({"window":self.binding},self.clients),(320,240,2052,1198))

    def test_resized_client_requires_new_calibration(self):
        self.info["client"] = (1000,400,1050,675)
        with self.assertRaisesRegex(ValueError,"尺寸已改变"):
            resolve_region({"window":self.binding},self.clients)

    def test_missing_or_reused_window_is_rejected(self):
        self.clients.info.return_value = None
        with self.assertRaisesRegex(ValueError,"最小化"):
            resolve_region({"window":self.binding},self.clients)
        self.clients.info.return_value = dict(self.info,pid=9999999)
        with self.assertRaisesRegex(ValueError,"窗口已改变"):
            resolve_region({"window":self.binding},self.clients)

    def test_own_overlay_and_incomplete_client_are_not_bound(self):
        self.clients.info.return_value = dict(self.info,pid=os.getpid())
        self.assertIsNone(bind_region(self.region,self.clients))
        self.clients.info.return_value = dict(self.info,client=(1000,400,200,200))
        self.assertIsNone(bind_region(self.region,self.clients))

    def test_old_config_uses_screen_region(self):
        self.assertEqual(resolve_region({"region":self.region},self.clients),self.region)

    def test_shot_binds_fixed_region_and_restores_only_game_focus(self):
        target = shot_target({"region":self.region},self.clients)
        self.assertEqual(target["window"]["hwnd"],123)
        self.clients.activate.return_value = True
        self.clients.foreground.return_value = 123
        focus_shot_target(target,self.clients)
        verify_shot_target(target,self.clients)
        self.clients.activate.assert_called_once_with(123)
        self.clients.foreground.return_value = 999
        with self.assertRaisesRegex(ValueError,"失去焦点"):
            verify_shot_target(target,self.clients)

    def test_shot_missing_binding_closed_game_and_failed_activation(self):
        self.clients.info.return_value = None
        with self.assertRaisesRegex(ValueError,"绑定"):
            shot_target({"region":self.region},self.clients)
        target = {"region":self.region,"window":self.binding}
        with self.assertRaisesRegex(ValueError,"已关闭"):
            focus_shot_target(target,self.clients)
        self.clients.info.return_value = self.info
        self.clients.activate.return_value = False
        with self.assertRaisesRegex(ValueError,"无法切回"):
            focus_shot_target(target,self.clients)

    def test_shot_unmarked_region_never_selects_an_external_window(self):
        self.clients.windows.reset_mock()
        with self.assertRaisesRegex(ValueError,"标记"):
            shot_target({"region":(0,0,0,0)},self.clients)
        self.clients.windows.assert_not_called()


class AutomaticWindowBindingTests(unittest.TestCase):
    def setUp(self):
        pid = os.getpid()+1
        self.hall = {"hwnd": 100, "pid": pid, "class_name": "36JBCOM_Browser",
                     "client": (100, 200, 2008, 1350)}
        self.flash = {"hwnd": 200, "pid": pid, "class_name": "MacromediaFlashPlayerActiveX",
                      "client": (104, 252, 2000, 1200)}
        self.infos = {100: self.hall, 200: self.flash}
        self.clients = Mock()
        self.clients.window_at.return_value = 200
        self.clients.root.return_value = 100
        self.clients.children.return_value = [200]
        self.clients.info.side_effect = self.infos.get
        self.point = (500, 500)

    def bind(self):
        return bind_window_at(self.point, self.clients)

    def test_drop_uses_only_flash_area_and_saves_both_identities(self):
        target = self.bind()
        self.assertEqual(target["region"], (104, 252, 2000, 1200))
        self.assertEqual(target["window"]["offset"], [4, 52, 2000, 1200])
        self.assertEqual(target["window"]["hwnd"], 100)
        self.assertEqual(target["window"]["content"]["hwnd"], 200)
        self.assertEqual(resolve_region(target, self.clients), target["region"])

    def test_drops_on_toolbar_reference_table_and_edges_are_rejected(self):
        for point in ((500, 220), (500, 1480), (2104, 500), (500, 1452)):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "游戏画面"):
                bind_window_at(point, self.clients)

    def test_drop_on_edit_child_still_selects_containing_flash(self):
        self.clients.window_at.return_value = 300
        self.assertEqual(self.bind()["window"]["content"]["hwnd"], 200)

    def test_moved_and_resized_game_updates_capture_without_calibration(self):
        target = self.bind()
        self.hall["client"] = (-1200, 100, 1004, 740)
        self.flash["client"] = (-1198, 126, 1000, 600)
        self.assertEqual(resolve_region(target, self.clients), (-1198, 126, 1000, 600))
        focus_shot_target(target, self.clients)
        self.clients.activate.assert_called_once_with(100)
        self.clients.foreground.return_value = 100
        verify_shot_target(target, self.clients)
        self.clients.foreground.return_value = 999
        with self.assertRaisesRegex(ValueError, "失去焦点"):
            verify_shot_target(target, self.clients)

    def test_closed_replaced_or_reparented_flash_is_rejected(self):
        target = self.bind()
        del self.infos[200]
        with self.assertRaisesRegex(ValueError, "重载"):
            resolve_region(target, self.clients)
        self.infos[200] = dict(self.flash, class_name="DifferentControl")
        with self.assertRaisesRegex(ValueError, "重载"):
            resolve_region(target, self.clients)
        self.infos[200] = dict(self.flash, pid=self.flash["pid"]+1)
        with self.assertRaisesRegex(ValueError, "重载"):
            resolve_region(target, self.clients)
        self.infos[200] = self.flash
        self.clients.root.return_value = 999
        with self.assertRaisesRegex(ValueError, "重载"):
            resolve_region(target, self.clients)

    def test_hidden_parent_and_own_helper_are_rejected(self):
        target = self.bind()
        del self.infos[100]
        with self.assertRaisesRegex(ValueError, "最小化"):
            resolve_region(target, self.clients)
        with self.assertRaises(ValueError):
            self.bind()
        self.infos[100] = dict(self.hall, pid=os.getpid())
        with self.assertRaisesRegex(ValueError, "拖到游戏"):
            self.bind()

    def test_desktop_unknown_renderers_and_ambiguous_clients_are_rejected(self):
        self.clients.window_at.return_value = 0
        with self.assertRaises(ValueError):
            self.bind()
        self.clients.window_at.return_value = 200
        self.flash["class_name"] = "Internet Explorer_Server"
        with self.assertRaisesRegex(ValueError, "独立"):
            self.bind()
        self.flash["class_name"] = "MacromediaFlashPlayerActiveX"
        self.infos[300] = dict(self.flash, hwnd=300)
        self.clients.children.return_value = [200, 300]
        with self.assertRaisesRegex(ValueError, "独立"):
            self.bind()

    def test_clipped_and_wrong_aspect_clients_are_rejected(self):
        target = self.bind()
        for rectangle in ((90, 252, 2000, 1200), (104, 252, 2000, 1500),
                          (104, 252, 200, 120), (104, 500, 2000, 1200)):
            self.flash["client"] = rectangle
            with self.subTest(rectangle=rectangle), self.assertRaises(ValueError):
                resolve_region(target, self.clients)

    def test_serialized_binding_keeps_dynamic_following(self):
        import json
        target = json.loads(json.dumps(self.bind()))
        self.flash["client"] = (104, 252, 1000, 600)
        self.assertEqual(resolve_region(target, self.clients), (104, 252, 1000, 600))


if __name__ == "__main__":
    unittest.main()
