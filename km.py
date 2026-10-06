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
