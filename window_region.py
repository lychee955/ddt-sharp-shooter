"""Bind the game's Flash client directly, or follow a legacy calibrated region."""

import ctypes
from ctypes import wintypes
import os
import logging
import time


class GUIThreadInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT)]


class MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


class WindowsClients:
    def __init__(self):
        if os.name != "nt":
            raise OSError("窗口跟随仅支持 Windows")
        self.api = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentThreadId.argtypes = []
        self.kernel.GetCurrentThreadId.restype = wintypes.DWORD
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        declarations = {
            "EnumWindows": ([self.callback_type, wintypes.LPARAM], wintypes.BOOL),
            "EnumChildWindows": ([wintypes.HWND, self.callback_type, wintypes.LPARAM], wintypes.BOOL),
            "WindowFromPoint": ([wintypes.POINT], wintypes.HWND),
            "GetAncestor": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
            "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            "GetClassNameW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            "IsWindow": ([wintypes.HWND], wintypes.BOOL),
            "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
            "IsIconic": ([wintypes.HWND], wintypes.BOOL),
            "GetWindowRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "MonitorFromWindow": ([wintypes.HWND, wintypes.DWORD], wintypes.HANDLE),
            "GetMonitorInfoW": ([wintypes.HANDLE, ctypes.POINTER(MonitorInfo)], wintypes.BOOL),
            "GetCursorPos": ([ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
            "GetForegroundWindow": ([], wintypes.HWND),
            "SetForegroundWindow": ([wintypes.HWND], wintypes.BOOL),
            "SetFocus": ([wintypes.HWND], wintypes.HWND),
            "IsChild": ([wintypes.HWND, wintypes.HWND], wintypes.BOOL),
            "GetGUIThreadInfo": ([wintypes.DWORD, ctypes.POINTER(GUIThreadInfo)], wintypes.BOOL),
            "AttachThreadInput": ([wintypes.DWORD, wintypes.DWORD, wintypes.BOOL], wintypes.BOOL),
            "PeekMessageW": ([ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                              wintypes.UINT, wintypes.UINT, wintypes.UINT], wintypes.BOOL),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result

    def windows(self):
        handles = []
        callback = self.callback_type(lambda hwnd, _: handles.append(int(hwnd)) or True)
        if not self.api.EnumWindows(callback, 0):
            raise OSError("无法读取窗口列表")
        return handles

    def foreground(self):
        return int(self.api.GetForegroundWindow() or 0)

    def window_at(self, point):
        return int(self.api.WindowFromPoint(wintypes.POINT(*point)) or 0)

    def root(self, hwnd):
        return int(self.api.GetAncestor(hwnd, 2) or 0)  # GA_ROOT

    def children(self, hwnd):
        handles = []
        callback = self.callback_type(lambda child, _: handles.append(int(child)) or True)
        self.api.EnumChildWindows(hwnd, callback, 0)
        return handles

    def activate(self, hwnd):
        if self.foreground() == hwnd:
            return True
        accepted = bool(self.api.SetForegroundWindow(hwnd))
        # Cross-thread activation is asynchronous. Immediately afterwards,
        # GetForegroundWindow may still return the old window, or even NULL.
        # Wait for the actual foreground window before focusing its control.
        deadline = time.monotonic() + .5
        while True:
            foreground = self.foreground()
            if foreground == hwnd:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logging.getLogger(__name__).warning(
                    "窗口激活超时：目标=%s，前台=%s，系统接受=%s",
                    hwnd, foreground, accepted)
                return False
            time.sleep(min(.01, remaining))

    def keyboard_focus(self, hwnd):
        thread = self.api.GetWindowThreadProcessId(hwnd, None)
        info = GUIThreadInfo(cbSize=ctypes.sizeof(GUIThreadInfo))
        if not thread or not self.api.GetGUIThreadInfo(thread, ctypes.byref(info)):
            raise OSError("无法读取游戏控件的键盘焦点")
        return int(info.hwndFocus or 0)

    def has_keyboard_focus(self, hwnd):
        focused = self.keyboard_focus(hwnd)
        return focused == hwnd or bool(focused and self.api.IsChild(hwnd, focused))

    def focus_control(self, hwnd):
        if self.has_keyboard_focus(hwnd):
            return True
        thread = self.api.GetWindowThreadProcessId(hwnd, None)
        current = self.kernel.GetCurrentThreadId()
        if not thread:
            return False
        # SetFocus needs a shared input queue. The shot worker may not yet
        # have a queue; PeekMessage creates one without consuming messages.
        message = wintypes.MSG()
        self.api.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        attached = thread != current
        if attached and not self.api.AttachThreadInput(current, thread, True):
            return False
        try:
            self.api.SetFocus(hwnd)
        finally:
            if attached:
                self.api.AttachThreadInput(current, thread, False)
        # A NULL SetFocus return can mean there was no previous focus.
        return self.has_keyboard_focus(hwnd)

    def info(self, hwnd):
        if not self.api.IsWindow(hwnd) or not self.api.IsWindowVisible(hwnd) or self.api.IsIconic(hwnd):
            return None
        rectangle, origin, pid = wintypes.RECT(), wintypes.POINT(0, 0), wintypes.DWORD()
        if not self.api.GetClientRect(hwnd, ctypes.byref(rectangle)) or not self.api.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        self.api.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        class_name = ctypes.create_unicode_buffer(256)
        self.api.GetClassNameW(hwnd, class_name, len(class_name))
        return {"hwnd": int(hwnd), "pid": pid.value, "class_name": class_name.value,
                "client": (origin.x, origin.y, rectangle.right, rectangle.bottom)}

    def window_state(self, hwnd):
        """Read identity even when hidden/minimized; never activate the window."""
        if not self.api.IsWindow(hwnd):
            return None
        pid, rectangle = wintypes.DWORD(), wintypes.RECT()
        self.api.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = ctypes.create_unicode_buffer(256)
        self.api.GetClassNameW(hwnd, name, len(name))
        if not self.api.GetWindowRect(hwnd, ctypes.byref(rectangle)):
            raise OSError("无法读取窗口外框")
        return {"hwnd": int(hwnd), "pid": pid.value, "class_name": name.value,
                "visible": bool(self.api.IsWindowVisible(hwnd)),
                "minimized": bool(self.api.IsIconic(hwnd)),
                "bounds": (rectangle.left, rectangle.top, rectangle.right, rectangle.bottom)}

    def work_area(self, hwnd):
        monitor = self.api.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
        if not monitor or not self.api.GetMonitorInfoW(monitor, ctypes.byref(info)):
            raise OSError("无法读取游戏所在屏幕的可用范围")
        return (info.rcWork.left, info.rcWork.top, info.rcWork.right, info.rcWork.bottom)

    def cursor_position(self):
        point = wintypes.POINT()
        if not self.api.GetCursorPos(ctypes.byref(point)):
            raise OSError("无法读取鼠标位置")
        return point.x, point.y


def _identity(info):
    return {key: info[key] for key in ("hwnd", "pid", "class_name")}


def _contains(region, point):
    x, y, width, height = region
    return x <= point[0] < x+width and y <= point[1] < y+height


def _validate_game_client(region, parent):
    x, y, width, height = region
    px, py, pw, ph = parent
    if width < 300 or height < 180 or abs(width/height - 5/3) > .06:
        raise ValueError("游戏画面未完整显示，请恢复普通游戏模式后重新绑定")
    if x < px or y < py or x+width > px+pw or y+height > py+ph:
        raise ValueError("游戏画面超出窗口，请完整显示游戏后重新绑定")


def bind_window_at(point, clients=None):
    """Select only a real Flash game client under the drop point, never the hall."""
    clients = clients or WindowsClients()
    hit = clients.window_at(point)
    root = clients.root(hit) if hit else 0
    parent = clients.info(root) if root else None
    if parent is None or parent["pid"] == os.getpid():
        raise ValueError("请将瞄准图标拖到游戏画面内松开")
    candidates = []
    for hwnd in [root, *clients.children(root)]:
        info = clients.info(hwnd)
        if (info is not None and info["class_name"].lower() == "macromediaflashplayeractivex"
                and _contains(info["client"], point)):
            candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("未找到独立的游戏画面，请拖到 36 大厅中的游戏内部，避开工具栏和力度表")
    content = candidates[0]
    region = content["client"]
    _validate_game_client(region, parent["client"])
    cx, cy, cw, ch = parent["client"]
    x, y, width, height = region
    return {"region": tuple(region), "window": _identity(parent) | {
        "client_size": [cw, ch], "offset": [x-cx, y-cy, width, height],
        "content": _identity(content),
    }}


def bind_region(region, clients=None):
    """Bind to the frontmost external client containing the entire selection."""
    try:
        clients = clients or WindowsClients()
        x, y, width, height = region
        for hwnd in clients.windows():
            info = clients.info(hwnd)
            if info is None or info["pid"] == os.getpid():
                continue
            cx, cy, cw, ch = info["client"]
            if cx <= x and cy <= y and x+width <= cx+cw and y+height <= cy+ch:
                return {key: info[key] for key in ("hwnd", "pid", "class_name")} | {
                    "client_size": [cw, ch], "offset": [x-cx, y-cy, width, height],
                }
    except OSError:
        pass
    return None


def resolve_region(config, clients=None):
    binding = config.get("window")
    if not binding:
        return tuple(config["region"])
    clients = clients or WindowsClients()
    info = clients.info(binding["hwnd"])
    if info is None:
        raise ValueError("游戏窗口已关闭、隐藏或最小化；请恢复窗口，重新打开后需拖动图标重新绑定")
    if any(info[key] != binding[key] for key in ("pid", "class_name")):
        raise ValueError("原游戏窗口已改变，请拖动图标重新绑定")
    cx, cy, cw, ch = info["client"]
    if binding.get("content"):
        saved = binding["content"]
        content = clients.info(saved["hwnd"])
        if (content is None or any(content[key] != saved[key] for key in ("pid", "class_name"))
                or clients.root(saved["hwnd"]) != binding["hwnd"]):
            raise ValueError("游戏画面已关闭或重载，请拖动图标重新绑定")
        _validate_game_client(content["client"], info["client"])
        return tuple(content["client"])
    if tuple(binding["client_size"]) != (cw, ch):
        raise ValueError("游戏窗口尺寸已改变，请拖动图标重新绑定，或重新校准游戏画面")
    dx, dy, width, height = binding["offset"]
    if min(dx, dy) < 0 or dx+width > cw or dy+height > ch:
        raise ValueError("保存的游戏区域超出客户区，请拖动图标重新绑定")
    return cx+dx, cy+dy, width, height


def shot_target(config, clients=None):
    """Capture a validated binding before the force dialog takes focus."""
    region = tuple(config.get("region", (0,0,0,0)))
    if len(region) != 4 or region[2] <= 0 or region[3] <= 0:
        raise ValueError("请先拖动图标绑定游戏窗口（尚未标记游戏区域）")
    clients = clients or WindowsClients()
    target = dict(config)
    if not target.get("window"):
        target["window"] = bind_region(target["region"], clients)
    if not target["window"]:
        raise ValueError("请先拖动图标绑定游戏窗口")
    if target["window"]["pid"] == os.getpid():
        raise ValueError("发射窗口不能是辅助程序，请将图标拖到游戏画面重新绑定")
    resolve_region(target, clients)
    return target


def focus_shot_target(target, clients=None):
    clients = clients or WindowsClients()
    resolve_region(target, clients)
    if not clients.activate(target["window"]["hwnd"]):
        raise ValueError("游戏窗口未能在规定时间内激活，请点击游戏画面后重新发射")
    content = target["window"].get("content")
    if content and not clients.focus_control(content["hwnd"]):
        raise ValueError("大厅已激活，但游戏控件未取得键盘焦点；请点击游戏画面后重试发射")


def verify_shot_target(target, clients=None):
    clients = clients or WindowsClients()
    resolve_region(target, clients)
    if clients.foreground() != target["window"]["hwnd"]:
        raise ValueError("游戏窗口已失去焦点，未发送发射按键")
    content = target["window"].get("content")
    if content and not clients.has_keyboard_focus(content["hwnd"]):
        raise ValueError("游戏控件未取得键盘焦点，未发送发射按键；请点击游戏画面后重试")
