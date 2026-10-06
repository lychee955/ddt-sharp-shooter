"""Explicit numeric shots, independent of recognition and Tk callbacks."""

import math
import threading
from copy import deepcopy


class ShotStateChanged(ValueError):
    """The displayed force no longer applies to the current game state."""


def parse_force(text):
    try:
        force = float(text.strip())
    except (ValueError, AttributeError):
        raise ValueError("请输入有效力度，例如 64.25") from None
    if not math.isfinite(force) or not 0 < force <= 100:
        raise ValueError("力度必须大于 0 且不超过 100，可输入小数")
    return force


class ShotController:
    """One user-confirmed shot at a time; background work never touches Tk."""

    def __init__(self, press, release, focus, verify, notify, seconds_per_force=.04,
                 *, press_direction=None, release_direction=None, tap_direction=None,
                 check_state=None, on_state_changed=None):
        self.press, self.release = press, release
        self.focus, self.verify, self.notify = focus, verify, notify
        self.seconds_per_force = seconds_per_force
        self.press_direction, self.release_direction = press_direction, release_direction
        self.tap_direction = tap_direction
        self.check_state, self.on_state_changed = check_state, on_state_changed
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._busy = False
        self._closed = False
        self._thread = None

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def cancel(self):
        with self._lock:
            self._cancel.set()

    def close(self):
        with self._lock:
            self._closed = True
            self._cancel.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(.5)

    def submit(self, force, target, *, direction=None):
        force = parse_force(str(force))
        if direction not in (None, "left", "right"):
            raise ValueError("发射方向必须为左或右")
        if direction is not None and self.tap_direction is None and (self.press_direction is None or self.release_direction is None):
            raise ValueError("方向按键尚未配置")
        target = deepcopy(target)
        with self._lock:
            if self._busy or self._closed:
                return False
            cancel = self._cancel = threading.Event()
            self._busy = True
            self._thread = threading.Thread(target=self._run, args=(force,target,cancel,direction), daemon=True)
            self._thread.start()
        return True

    def _run(self, force, target, cancel, direction=None):
        message = "指定力度发射已取消"
        try:
            if cancel.wait(.1):
                return
            self.focus(target)
            if cancel.wait(.15):
                return
            self.verify(target)
            if cancel.is_set():
                return
            if direction is not None:
                if self.check_state is not None:
                    self.check_state(target)
                    self.verify(target)
                if cancel.is_set():
                    return
                if self.tap_direction is not None:
                    self.tap_direction(direction)
                else:
                    try:
                        self.press_direction(direction)
                    finally:
                        self.release_direction(direction)
                # Every wait and screenshot occurs after the arrow is released.
                if cancel.wait(.1):
                    return
                self.verify(target)
                if self.check_state is not None:
                    self.check_state(target)
                    if cancel.wait(.05):
                        return
                    self.verify(target)
                    self.check_state(target)
                    self.verify(target)
                if cancel.is_set():
                    return
            try:
                self.press()
                interrupted = cancel.wait(self.seconds_per_force*force)
            finally:
                self.release()
            message = "蓄力已中止并松开空格" if interrupted else f"已按指定力度 {force:g} 发射"
        except ShotStateChanged as exc:
            message = f"发射已停止：{exc}；正在刷新计算，请重新选择目标"
            if self.on_state_changed is not None:
                self.on_state_changed()
        except Exception as exc:
            message = f"指定力度发射失败：{exc}"
        finally:
            with self._lock:
                self._busy = False
            self.notify(message)
