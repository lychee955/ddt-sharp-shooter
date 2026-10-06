"""Tk logging must not call the UI while a background logging lock is held."""

import logging
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import tkinter
import unittest

from logger import redirect_output_to_text_widget


class ThreadBoundText:
    def __init__(self):
        self.owner = threading.get_ident()
        self.callbacks = []
        self.text = ""
        self.destroyed = False

    def check_thread(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("background thread called Tk")
        if self.destroyed:
            raise tkinter.TclError("widget destroyed")

    def after(self, delay, callback):
        self.check_thread()
        self.callbacks.append(callback)

    def configure(self, **kwargs):
        self.check_thread()

    def insert(self, index, message):
        self.check_thread()
        self.text += message

    def see(self, index):
        self.check_thread()

    def tick(self):
        self.callbacks.pop(0)()


class TkLoggingTests(unittest.TestCase):
    def setUp(self):
        self.widget = ThreadBoundText()
        self.handler = redirect_output_to_text_widget(self.widget)
        self.addCleanup(self.handler.close)
        self.log = logging.Logger("test.ui.queue")
        self.log.addHandler(self.handler)

    def test_background_and_escape_logs_wait_for_main_thread_drain(self):
        errors = []
        def background():
            try:
                self.log.info("keyboard Escape")
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=background,daemon=True)
        thread.start()
        thread.join(1)
        self.assertFalse(thread.is_alive(),"logging waited for Tk")
        self.assertEqual(errors,[])
        self.log.info("overlay cancelled")
        self.assertEqual(self.widget.text,"")
        self.widget.tick()
        self.assertEqual(self.widget.text,"keyboard Escape\noverlay cancelled\n")
        self.assertEqual(len(self.widget.callbacks),1)

    def test_many_logs_are_drained_in_bounded_batches(self):
        for i in range(250):
            self.log.info("message %s",i)
        self.widget.tick()
        self.assertEqual(len(self.widget.text.splitlines()),200)
        self.widget.tick()
        self.assertEqual(len(self.widget.text.splitlines()),250)

    def test_destroyed_widget_stops_timer_without_errors(self):
        self.log.info("last message")
        self.widget.destroyed = True
        self.widget.tick()
        self.assertTrue(self.handler._closed)
        self.assertEqual(self.widget.callbacks,[])

    def test_closed_handler_has_no_more_ui_activity(self):
        self.handler.close()
        self.log.info("ignored")
        self.widget.tick()
        self.assertEqual(self.widget.text,"")
        self.assertEqual(self.widget.callbacks,[])

    def test_real_hidden_tk_loop_remains_responsive_during_escape_logging(self):
        # A deadlock would also freeze Tk's timeout callback, so enforce the
        # timeout from a separate process. No keyboard hooks or game capture.
        code = textwrap.dedent('''
            import logging
            import threading
            import tkinter as tk
            from logger import redirect_output_to_text_widget
            root = tk.Tk()
            root.withdraw()
            text = tk.Text(root)
            handler = redirect_output_to_text_widget(text)
            log = logging.Logger("escape.integration")
            log.addHandler(handler)
            entered = threading.Event()
            class SignallingFormatter(logging.Formatter):
                def format(self, record):
                    if record.msg == "global Escape":
                        entered.set()
                    return super().format(record)
            handler.setFormatter(SignallingFormatter())
            def cancel():
                log.info("overlay cancelled")
            root.after(0,cancel)
            worker = threading.Thread(target=lambda: log.info("global Escape"),daemon=True)
            worker.start()
            assert entered.wait(1)
            root.after(200,root.quit)
            root.mainloop()
            worker.join(1)
            assert not worker.is_alive()
            contents = text.get("1.0","end")
            assert "global Escape" in contents and "overlay cancelled" in contents
            handler.close()
            root.destroy()
            print("responsive")
        ''')
        result = subprocess.run([sys.executable,"-B","-c",code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("responsive",result.stdout)


if __name__ == "__main__":
    unittest.main()
