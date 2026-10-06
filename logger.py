import logging
import tkinter as tk
from queue import Empty, Queue


logger = logging.getLogger(__name__)


def setup_logger(text_widget):
    """
    Set up the logger for the application.
    """
    logger.setLevel(logging.INFO)
    logger.addHandler(redirect_output_to_text_widget(text_widget))


def redirect_output_to_text_widget(text_widget):
    class TextHandler(logging.Handler):
        def __init__(self, text_widget):
            logging.Handler.__init__(self)
            self.text_widget = text_widget
            self.messages = Queue()
            self.text_widget.after(50, self.drain)

        def emit(self, record):
            # Logging holds this handler's lock during emit. A worker calling
            # Tk.after here can wait for Tk while the UI waits for that lock.
            # Never call any Tk API from a logging thread (including after).
            if not self._closed:
                self.messages.put(self.format(record))

        def drain(self):
            """Scheduled only by Tk's main thread, without the logging lock."""
            if self._closed:
                return
            messages = []
            for _ in range(200):
                try:
                    messages.append(self.messages.get_nowait())
                except Empty:
                    break
            try:
                if messages:
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert(tk.END, "\n".join(messages) + "\n")
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state="disabled")
                self.text_widget.after(50, self.drain)
            except tk.TclError:
                self.close()

    return TextHandler(text_widget)
