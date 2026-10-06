"""Explicit numeric shots, independent of recognition and Tk callbacks."""

import math
import threading


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

    def __init__(self, press, release, focus, verify, notify, seconds_per_force=.04):
        self.press, self.release = press, release
        self.focus, self.verify, self.notify = focus, verify, notify
        self.seconds_per_force = seconds_per_force
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

    def submit(self, force, target):
        force = parse_force(str(force))
        with self._lock:
            if self._busy or self._closed:
                return False
            cancel = self._cancel = threading.Event()
            self._busy = True
            self._thread = threading.Thread(target=self._run, args=(force,target,cancel), daemon=True)
            self._thread.start()
        return True

    def _run(self, force, target, cancel):
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
            try:
                self.press()
                interrupted = cancel.wait(self.seconds_per_force*force)
            finally:
                self.release()
            message = "蓄力已中止并松开空格" if interrupted else f"已按指定力度 {force:g} 发射"
        except Exception as exc:
            message = f"指定力度发射失败：{exc}"
        finally:
            with self._lock:
                self._busy = False
            self.notify(message)
