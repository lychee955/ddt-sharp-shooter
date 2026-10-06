"""Check persistent crash diagnostics in an isolated interpreter."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_main_thread_callback_and_native_tracebacks_are_preserved(self):
        code = '''
import faulthandler
from pathlib import Path
import sys
import threading
from runtime_diagnostics import setup_diagnostics, report_callback_exception
from logger import logger
from logging.handlers import RotatingFileHandler
setup_diagnostics(sys.argv[1])
setup_diagnostics(sys.argv[1])
assert len([h for h in logger.handlers if isinstance(h, RotatingFileHandler)]) == 1
try:
    raise RuntimeError("callback failure")
except Exception:
    report_callback_exception(*sys.exc_info())
def fail():
    raise ValueError("worker failure")
thread = threading.Thread(target=fail)
thread.start()
thread.join()
faulthandler.dump_traceback(file=__import__("runtime_diagnostics")._fault_file)
raise RuntimeError("main failure")
'''
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, "-c", code, directory],
                                     cwd=Path(__file__).resolve().parents[1],
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 1, process.stderr)
            text = (Path(directory)/"dss.log").read_text(encoding="utf-8")
            for message in ("PID=", "callback failure", "worker failure", "main failure"):
                self.assertIn(message, text)
            self.assertIn("File", (Path(directory)/"native-crash.log").read_text(encoding="utf-8"))
