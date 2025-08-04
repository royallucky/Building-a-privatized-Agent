# log_config.py
import os
import logging
import queue
import logging.handlers
from datetime import datetime

# ✅ 模块级全局变量（天然生命周期伴随主程序）
log_queue = queue.Queue()
_global_task_logger = None
_listener = None


def get_log_file_path(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"{date_str}.log")


def setup_async_logger(name="agent", level=logging.INFO):
    global _global_task_logger, _listener

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    file_handler = logging.FileHandler(get_log_file_path(), encoding="utf-8")
    file_handler.setFormatter(formatter)

    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(queue_handler)

    # ✅ 仅创建一次 listener
    if not _listener:
        _listener = logging.handlers.QueueListener(
            log_queue, file_handler, respect_handler_level=True
        )
        _listener.start()

    # ✅ 注入全局变量，供其他模块访问
    _global_task_logger = logger
    return logger


def get_global_logger():
    return _global_task_logger


def stop_async_logger():
    global _listener
    if _listener:
        _listener.stop()
