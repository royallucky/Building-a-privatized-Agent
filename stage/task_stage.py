from datetime import datetime

from logs.logger_config import get_global_logger


class TaskState:
    def __init__(self, task_id):
        self.task_id = task_id
        self.status = "Not Started"
        self.error_message = None
        self.start_time = None
        self.end_time = None
        self.retry_count = 0
        self.max_retries = 3

    def update_status(self, status, error_message=None):
        self.status = status
        self.error_message = error_message

        now = self._now_str()
        if status == "Running":
            self.start_time = now
        elif status in ("Completed", "Failed"):
            self.end_time = now

        logger = get_global_logger()
        if logger:
            msg = f"[Task {self.task_id}] 状态 -> {status}"
            if error_message:
                msg += f" | 错误: {error_message}"
            if status == "Failed":
                logger.error(msg)
            else:
                logger.info(msg)

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
