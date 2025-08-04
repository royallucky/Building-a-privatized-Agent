import time

from stage.task_stage import TaskState


class TaskScheduler:
    def __init__(self):
        self.tasks = {}

    def add_task(self, task_id, task_state: TaskState):
        self.tasks[task_id] = task_state

    def execute_task(self, task_id, task_function):
        task_state = self.tasks.get(task_id)
        if not task_state:
            print(f"任务 {task_id} 不存在")
            return

        task_state.update_status("Running")  # 更新任务状态为“运行中”
        print(f"任务 {task_id} 开始执行")

        while task_state.retry_count < task_state.max_retries:
            try:
                task_function()  # 执行任务
                task_state.update_status("Completed")  # 任务完成，更新状态
                print(f"任务 {task_id} 执行完成")
                return
            except Exception as e:
                task_state.retry_count += 1
                task_state.error_message = str(e)
                print(f"任务 {task_id} 执行失败，错误：{e}，第{task_state.retry_count}次重试")
                time.sleep(0.1)  # 等待后重试
        print(f"任务 {task_id} 达到最大重试次数，失败")

    def resume_task(self, task_id):
        task_state = self.tasks.get(task_id)
        if task_state and task_state.status == "Failed":
            print(f"任务 {task_id} 正在恢复从失败节点...")
            self.execute_task(task_id, task_state)  # 从失败节点继续执行任务


if __name__ == '__main__':
    # 任务定义
    def load_data():
        print("加载数据...")


    def clean_data():
        print("清洗数据...")


    def run_model():
        print("运行模型进行分析...")


    async def generate_report():
        print("生成报告...")


    async def send_email():
        print("发送邮件...")

    def generate_task_planning():
        # 模拟LLM生成的任务规划，任务的顺序和内容可能会动态变化
        return [
            {"task_id": "task_001", "name": "数据清洗", "function": clean_data},
            {"task_id": "task_002", "name": "模型训练", "function": run_model},
            {"task_id": "task_003", "name": "报告生成", "function": generate_report},
            {"task_id": "task_004", "name": "发送邮件", "function": send_email}
        ]

    # 创建任务调度器
    scheduler = TaskScheduler()
    # 获取LLM生成的任务规划
    task_planning = generate_task_planning()

    # 动态生成任务状态并添加到任务调度器
    for task in task_planning:
        task_state = TaskState(task["task_id"])  # 为每个任务创建状态对象
        scheduler.add_task(task["task_id"], task_state)  # 将任务状态对象添加到调度器中

    # 执行任务
    for task in task_planning:
        task_function = task["function"]
        scheduler.execute_task(task["task_id"], lambda: task_function())

    # 模拟任务恢复
    scheduler.resume_task("task_001")  # 例如从任务1开始恢复执行

