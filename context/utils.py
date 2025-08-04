import threading
import time
import random


class TaskContext:
    def __init__(self, task_id, user_id, product, start_date, end_date):
        """
        初始化任务上下文对象，用于管理任务的基本信息和状态。

        :param task_id: 任务ID
        :param user_id: 用户ID
        :param product: 产品类型
        :param start_date: 任务开始日期
        :param end_date: 任务结束日期
        """
        self.task_id = task_id  # 任务ID
        self.user_id = user_id  # 用户ID
        self.product = product  # 产品类型
        self.start_date = start_date  # 开始日期
        self.end_date = end_date  # 结束日期
        self.raw_data_path = None  # 原始数据路径
        self.cleaned_data = None  # 清洗后的数据
        self.model_result = None  # 模型分析结果
        self.report_url = None  # 报告生成的URL
        self.status = "Not Started"  # 任务状态，初始为未开始
        self.error_message = None  # 错误信息，初始为空

    def update_status(self, status, error_message=None):
        """
        更新任务的状态，并记录任务的错误信息（如果有的话）。

        :param status: 任务的新状态（如"Running"、"Completed"、"Failed"）
        :param error_message: 任务失败时的错误信息
        """
        self.status = status  # 更新任务状态
        self.error_message = error_message  # 如果有错误，更新错误信息
        if status == "Completed":
            self.end_time = self.get_current_time()  # 如果任务完成，记录结束时间
        elif status == "Running":
            self.start_time = self.get_current_time()  # 如果任务正在运行，记录开始时间

    def get_current_time(self):
        """
        获取当前时间，格式为"YYYY-MM-DD HH:MM:SS"。

        :return: 当前时间字符串
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def set_raw_data(self, path):
        """
        设置原始数据的路径。

        :param path: 原始数据的路径
        """
        self.raw_data_path = path

    def set_cleaned_data(self, data):
        """
        设置清洗后的数据。

        :param data: 清洗后的数据
        """
        self.cleaned_data = data

    def set_model_result(self, result):
        """
        设置模型分析结果。

        :param result: 模型分析的结果
        """
        self.model_result = result

    def set_report_url(self, url):
        """
        设置报告的URL。

        :param url: 生成的报告URL
        """
        self.report_url = url


class TaskScheduler:
    def __init__(self):
        """
        初始化任务调度器，负责任务的管理与调度。
        """
        self.tasks = {}  # 存储任务和其上下文的字典
        self.lock = threading.Lock()  # 线程锁，确保线程安全

    def add_task(self, task_id, task_context):
        """
        将任务和其上下文添加到调度器中。

        :param task_id: 任务ID
        :param task_context: 任务的上下文对象（TaskContext）
        """
        with self.lock:
            self.tasks[task_id] = task_context  # 将任务ID和其上下文添加到字典中

    def execute_task(self, task_id, task_function):
        """
        执行指定的任务，并更新任务的状态。

        :param task_id: 任务ID
        :param task_function: 执行任务的函数
        """
        task_context = self.tasks.get(task_id)  # 获取任务的上下文
        if not task_context:
            print(f"任务 {task_id} 不存在")  # 如果任务不存在，输出提示信息
            return

        task_context.update_status("Running")  # 更新任务状态为“运行中”
        print(f"任务 {task_id} 开始执行")

        try:
            task_function(task_context)  # 执行任务函数，传递上下文
            task_context.update_status("Completed")  # 如果任务执行成功，更新状态为“已完成”
            print(f"任务 {task_id} 执行完成")
        except Exception as e:
            task_context.update_status("Failed", str(e))  # 如果任务执行失败，更新状态为“失败”
            print(f"任务 {task_id} 执行失败，错误：{e}")  # 输出错误信息

    def resume_task(self, task_id):
        """
        恢复一个失败的任务，从失败节点继续执行。

        :param task_id: 任务ID
        """
        task_context = self.tasks.get(task_id)  # 获取任务的上下文
        if task_context and task_context.status == "Failed":  # 如果任务存在并且状态为“失败”
            print(f"任务 {task_id} 正在恢复从失败节点...")
            self.execute_task(task_id, task_context)  # 从失败的节点开始恢复执行任务
