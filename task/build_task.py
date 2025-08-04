import os
import json
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel
from langchain_core.tools import Tool

from config.tool_map import TOOL_FUNC_MAP


class Builder:
    def __init__(self, logger, registry_file="./config/tool_registry.json"):
        self.logger = logger
        self.registry_file = registry_file
        self.tool_registry = {}

        # 自动加载工具注册表
        self.load_registry()

    def load_registry(self):
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

        with open(self.registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for name, meta in data.items():
            func = TOOL_FUNC_MAP.get(name)
            if func:
                self.tool_registry[name] = Tool(
                    name=meta["name"],
                    description=meta.get("description", ""),
                    func=func
                )
            else:
                self.logger.warning(f"Tool '{name}' 未定义或未导入")

    def register_tool(self, name, description):
        try:
            """注册元信息并写入注册文件（函数仍需写死在代码里）"""
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data[name] = {
                "name": name,
                "description": description
            }

            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            return False

    def get_tool(self, name):
        return self.tool_registry.get(name)

    def run(self, task_planning, business_id):
        """
        :param task_planning: sample
            [
                {"task_id": "load_data", "tool_name": "load_data", "type": "serial"},
                {"task_id": "clean_data", "tool_name": "clean_data", "type": "serial"},
                {"task_id": "run_model", "tool_name": "run_model", "type": "serial"},
                {"task_id": "generate_charts", "tool_name": {
                    "trend": "draw_trend",
                    "distribution": "draw_distribution"
                }, "type": "parallel"},
                {"task_id": "combine_results", "tool_name": "combine_outputs", "type": "serial"}
            ]
        :param business_id
        :return:
        """
        if len(task_planning) < 2:
            self.logger.error("任务规划为空或任务步骤数量低于2")
            return

        # 构建任务流
        flow = RunnableSequence(
            RunnableLambda(self.get_tool(task_planning[0]['tool_name']).func)
            | RunnableLambda(self.get_tool(task_planning[1]['tool_name']).func)
        )

        for task in task_planning[2:]:
            if task["type"] == "parallel":
                parallel_tasks = {
                    key: RunnableLambda(self.get_tool(value).func)
                    for key, value in task["tool_name"].items()
                }
                flow = flow | RunnableParallel(parallel_tasks)
            else:
                flow = flow | RunnableLambda(self.get_tool(task["tool_name"]).func)

        msg = f"[{business_id}]"
        self.logger.info(msg + "开始执行流程...")
        output = flow.invoke({"raw_data": []})
        self.logger.info(msg + f"流程结束，全流程: {output}")
