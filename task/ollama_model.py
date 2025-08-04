import json

from langchain_community.chat_models import ChatOllama


class LLMModel_1:
    def __init__(self):
        self.LLM = ChatOllama(
            model="llama3.2",               # 换成你的模型
            temperature=0.3,
            system="你是严谨的软件工程助手，所有规划以 JSON 数组格式输出"
        )
        self.TOOLS = None

    def update_tools(self, Tools):
        self.TOOLS = Tools

    def run(self, user_goal: str) -> list:
        """
        只生成任务规划，不执行，用于观测 LLM 输出
        """
        # 假设 self.TOOLS 是一个字典 {tool_name: Tool}
        tool_descriptions = [
            f"- {tool.name}: {tool.description}" for tool in self.TOOLS.values()
        ]
        tool_section = "\n".join(tool_descriptions)

        plan_prompt = (
            "你是一个 Task Planner，负责将用户的任务目标拆解成可以用工具执行的步骤。\n\n"
            "以下是当前系统中可用的工具，每个工具包含名称和用途：\n"
            f"{tool_section}\n\n"
            "## 用户任务目标：\n"
            f"{user_goal}\n\n"
            "请将整个任务拆解为多个步骤，每一步都指定一个工具名称，并写明该步骤的作用以及任务是并发还是串行。\n"
            "输出格式为 JSON 数组，每个元素格式如下：\n"
            "{'id': 1, 'tool_name': 'load_data', 'description': '加载原始数据', 'type': 'serial'}\n\n"
            "请严格按照 JSON 数组格式输出。仅输出 JSON 数据，不输出其他内容"
        )
        return self.LLM.invoke(plan_prompt).content

