import random
import threading
import time

from context.utils import TaskContext, TaskScheduler


def task_cleaning(context: TaskContext):
    print(f"任务 {context.task_id} 清洗数据...")
    time.sleep(random.uniform(0.5, 1.5))  # 模拟清洗过程
    context.set_cleaned_data(f"Cleaned data for {context.product}")
    print(f"任务 {context.task_id} 清洗数据完成")


def task_modeling(context: TaskContext):
    print(f"任务 {context.task_id} 开始模型推理...")
    time.sleep(random.uniform(0.5, 2.0))  # 模拟模型推理
    context.set_model_result(f"Model results for {context.product}")
    print(f"任务 {context.task_id} 模型推理完成")


def task_reporting(context: TaskContext):
    print(f"任务 {context.task_id} 生成报告...")
    time.sleep(1)  # 模拟报告生成
    context.set_report_url(f"Report URL for {context.task_id}")
    print(f"任务 {context.task_id} 报告生成完成")


# 创建任务调度器
scheduler = TaskScheduler()

# 定义任务规划
task_planning = [
    {"task_id": "task_001", "name": "数据清洗", "function": task_cleaning},
    {"task_id": "task_002", "name": "模型推理", "function": task_modeling},
    {"task_id": "task_003", "name": "报告生成", "function": task_reporting},
]

# 动态创建任务状态对象并将其添加到调度器
for task in task_planning:
    task_context = TaskContext(task["task_id"], "u001", "MicroLED", "2025-06-25", "2025-07-01")
    scheduler.add_task(task["task_id"], task_context)


# 执行任务（并发）
def run_concurrent_tasks():
    threads = []
    for task in task_planning:
        task_function = task["function"]
        task_id = task["task_id"]
        thread = threading.Thread(target=scheduler.execute_task, args=(task_id, task_function))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


# 执行任务
run_concurrent_tasks()

# 模拟任务恢复
scheduler.resume_task("task_001")  # 从任务1恢复执行
