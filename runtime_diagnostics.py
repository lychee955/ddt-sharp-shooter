"""Persistent runtime errors, including failures without a visible console."""

import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading

from logger import logger


_fault_file = None


def report_callback_exception(exc_type, exc_value, traceback):
    logger.error("界面回调异常", exc_info=(exc_type, exc_value, traceback))


def setup_diagnostics(directory=None):
    global _fault_file
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "dss.log"
    if not any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(log_path.resolve())
               for handler in logger.handlers):
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if _fault_file is None:
        # Keep the descriptor alive: native fault handling writes directly to it.
        _fault_file = (directory / "native-crash.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    sys.excepthook = lambda kind, value, tb: logger.critical("主线程未处理异常", exc_info=(kind, value, tb))
    threading.excepthook = lambda args: logger.critical(
        "后台线程 %s 未处理异常", args.thread.name if args.thread else "未知",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    logger.info("辅助启动，PID=%s，日志目录=%s", os.getpid(), directory)
