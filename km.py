import queue
import pyautogui
from pynput import keyboard


RELEASE_CHAR_PREFIX = "r_"
_queue: queue.Queue
_keyboard_listener: keyboard.Listener
_s_down = False


def on_press(event):
    global _s_down
    try:
        if event.char and event.char.lower() == "s":
            if _s_down:
                return
            _s_down = True
        _queue.put(event.char)
    except AttributeError:
        if event == keyboard.Key.esc:
            _queue.put("esc")
        elif event == keyboard.Key.enter:
            _queue.put("enter")
        elif event == keyboard.Key.backspace:
            _queue.put("delete")


def on_release(event):
    global _s_down
    if getattr(event,"char",None) and event.char.lower() == "s":
        _s_down = False


def space_press(pause=True):
    pyautogui.keyDown("space",_pause=pause)


def space_release(pause=True):
    pyautogui.keyUp("space",_pause=pause)


def direction_press(direction):
    pyautogui.keyDown(direction, _pause=False)


def direction_release(direction):
    pyautogui.keyUp(direction, _pause=False)


def direction_tap(direction, *, api=None):
    """Queue down and up together, without a sleep while an arrow is held."""
    import ctypes
    from ctypes import wintypes

    if direction not in ("left", "right"):
        raise ValueError("换向按键必须为左或右")

    class KeyboardInput(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]

    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                    ("wParamH", wintypes.WORD)]

    class Payload(ctypes.Union):
        _fields_ = [("ki", KeyboardInput), ("mi", MouseInput), ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _anonymous_ = ("payload",)
        _fields_ = [("type", wintypes.DWORD), ("payload", Payload)]

    api = api if api is not None else ctypes.WinDLL("user32", use_last_error=True)
    api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    api.SendInput.restype = wintypes.UINT
    events = (Input*2)()
    for event, flags in zip(events, (1, 3)):  # extended arrow; extended arrow + key up
        event.type = 1  # INPUT_KEYBOARD
        event.ki = KeyboardInput(0x25 if direction == "left" else 0x27, 0, flags, 0, 0)
    sent = api.SendInput(2, events, ctypes.sizeof(Input))
    if sent != 2:
        if sent == 1:
            # A partial send must release the arrow before reporting failure.
            release = (Input*1)(events[1])
            if api.SendInput(1, release, ctypes.sizeof(Input)) != 1:
                raise OSError("换向按键松开失败，请松开方向键后重试")
        raise OSError("换向按键未完整发送，已停止发射")


def stop_listen() -> None:
    _keyboard_listener.stop()


def setup(km_queue):
    global _queue, _keyboard_listener
    _queue = km_queue

    _keyboard_listener = keyboard.Listener(on_press=on_press,on_release=on_release)
    _keyboard_listener.start()


def get_curr_mouse_pos():
    """Get the current mouse position"""
    return pyautogui.position()
