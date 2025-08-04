import atexit
import json

from logs.logger_config import setup_async_logger
from stage.task_stage import TaskState
from task.build_task import Builder
from task.ollama_model import LLMModel_1

atexit.register(setup_async_logger)


class AnalysAgent:
    def __init__(self):
        # 日志记录
        self.Agent_logger = setup_async_logger("Agent Logger")
        self.Agent_logger.info("日志模块启动成功")
        # 任务解析
        self.Agent_builder = Builder(logger=self.Agent_logger)
        self.Agent_logger.info("任务解析器启动成功")
        # 任务规划
        self.Agent_planner = LLMModel_1()
        self.Agent_planner.update_tools(Tools=self.Agent_builder.tool_registry)
        self.Agent_logger.info("LLM加载&任务规划启动成功")

    def register_tool(self, name, description):
        if self.Agent_builder.register_tool(name, description):
            self.Agent_logger.info(f"Tool ({name}) 注册成功")
        else:
            self.Agent_logger.warning(f"Tool ({name}) 注册失败")

    def run(self):
        plans = self.Agent_planner.run(user_goal="读取数据库中的销售数据，清洗掉异常值，画出分布图并保存结果")
        self.Agent_logger.info(f"LLM生成的任务规划: \n{plans}")
        try:
            plans = json.loads(plans)
        except json.JSONDecodeError as e:
            self.Agent_logger.error(f"解析任务规划的 JSON 格式解析失败: {e}")
            plans = []
        self.Agent_logger.info(f"解析得到标准 JSON 格式的任务规划: \n{plans}")

        task = TaskState(task_id="Task-001")
        # 更新状态自动记录日志
        task.update_status("Running")
        self.Agent_builder.run(task_planning=plans, business_id=1)
        task.update_status("Completed")


if __name__ == '__main__':
    AA = AnalysAgent()
    AA.run()
