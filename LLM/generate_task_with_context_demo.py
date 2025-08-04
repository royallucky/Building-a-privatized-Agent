"""
示例：基于LangChain和自定义上下文管理，动态生成、补全和调整数据分析任务规划，并驱动后端执行。
本示例包含完整注释，适合初学者理解和业务快速扩展。
"""

import os
import json
import re
from typing import Union, Literal, List

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, RootModel, ValidationError

# ========= 1. 业务工具函数与注册 =========
def load_data(context):
    """【模拟】加载数据。实际业务应替换为数据读取/入库等逻辑"""
    print("加载数据...（此处为模拟）")
    context["raw_data"] = [1, 2, 3]
    return context

def clean_data(context):
    """【模拟】数据清洗。实际业务应实现自己的数据预处理逻辑"""
    print(f"清洗数据: {context.get('raw_data')}")
    context["cleaned_data"] = [x for x in context["raw_data"] if x > 1]
    context["model_result"] = context["cleaned_data"]
    return context

def run_model(context):
    """【模拟】模型推理。实际业务应调用模型接口/AI服务"""
    print(f"运行模型: 输入数据 {context.get('cleaned_data')}")
    context["model_result"] = sum(context["cleaned_data"])
    return context

def draw_trend(context):
    """【模拟】绘制趋势图。实际业务应接入绘图或BI接口"""
    print(f"绘制趋势图: 基于结果 {context.get('model_result')}")
    context["trend"] = f"trend_chart_based_on_{context['model_result']}"
    return context

def draw_distribution(context):
    """【模拟】绘制分布图。实际业务应接入绘图或BI接口"""
    print(f"绘制分布图: 基于结果 {context.get('model_result')}")
    context["distribution"] = f"distribution_chart_based_on_{context['model_result']}"
    return context

def combine_outputs(context):
    """【模拟】结果合并。实际业务可自定义汇总逻辑"""
    print(f"合并输出数据: {context}")
    context["final_report"] = {
        "trend": context.get("trend"),
        "distribution": context.get("distribution")
    }
    return context

# 工具注册表：小白理解——把每个可自动执行的步骤登记到这个字典即可
registry = {
    "load_data": [load_data, "加载数据"],
    "clean_data": [clean_data, "清洗数据"],
    "run_model": [run_model, "缺陷检测模型推理"],
    "draw_trend": [draw_trend, "生成缺陷趋势图"],
    "draw_distribution": [draw_distribution, "生成分布图"],
    "combine_outputs": [combine_outputs, "合并数据并输出结果"]
}

# ========= 2. Prompt模板定义（链式提示生成/调整任务规划） =========
context_template = """
你是智能任务规划Agent，善于理解历史上下文、任务状态及用户约束，能够根据对话历史动态补全和调整任务流程。

【工具列表】
{tool_registry}

【对话历史】
{chat_history}

【中间状态】
{intermediate_state}

【当前已生成任务规划】
{current_plan}

【当前用户请求】
{current_user_input}

【约束规则】
- 任务规划需避免重复步骤
- 保证上下文一致性
- 遵循业务流程规范
- 输出为标准JSON结构，字段包括 id、tool_name、description、type

【输出要求】
- 基于【对话历史】、【中间状态】与【当前已生成任务规划】，对【当前用户请求】进行正确的任务规划。
- 输出【完成调整后的最新完整任务规划】的JSON内容，不包含无关文本。

请基于以上信息，补全或调整任务规划。
"""

# ========= 3. JSON工具函数 =========
def extract_json(content: str) -> str:
    """提取 LLM 返回的 JSON 数组片段。只保留最后一个数组，兼容冗余输出。"""
    matches = re.findall(r'\[.*?\]', content, re.DOTALL)
    if matches:
        return matches[-1]
    else:
        raise ValueError("未找到有效的 JSON 数据")

# ========= 4. 任务规划的数据结构（基于 Pydantic） =========
class PlanNode(BaseModel):
    id: int
    tool_name: Union[str, dict]
    description: str
    type: Literal["serial", "parallel"]

class Plan(RootModel):
    root: List[PlanNode]

def safe_parse(json_str: str) -> List[PlanNode]:
    """解析并验证 LLM 输出的 JSON 结构，保证类型安全。"""
    try:
        parsed_data = json.loads(json_str)
        return Plan.model_validate(parsed_data).root
    except ValidationError as e:
        raise RuntimeError(f"LLM 输出格式错误: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 解析失败: {e}")

# ========= 5. 生成上下文链式提示 =========
def build_context_prompt(
        chat_history: List[str],
        intermediate_state: str,
        current_plan: str,
        current_user_input: str
    ) -> str:
    """组装完整上下文Prompt，方便LLM理解全部信息并生成下一步。"""
    context_prompt = PromptTemplate(
        input_variables=["tool_registry", "chat_history", "intermediate_state", "current_plan", "current_user_input"],
        template=context_template
    )
    prompt = context_prompt.format(
        tool_registry=str("\n".join([f"- {k}：{v[-1]}" for k, v in registry.items()])),
        chat_history="\n".join(chat_history),
        intermediate_state=intermediate_state,
        current_plan=current_plan,
        current_user_input=current_user_input
    )
    return prompt

# ========= 6. 简单缓存类（可以理解为“会话变量”） =========
class SimpleCache:
    """小白也能理解的缓存方案：用于保存每个session的任务规划结果"""
    def __init__(self):
        self.data = {}
    def get(self, k):
        return self.data.get(k, [])
    def set(self, k, v):
        self.data[k] = v

# ========= 7. 任务规划与驱动执行Agent（核心） =========
class ContextPlanner:
    """
    负责：
    1. 根据历史和当前请求，动态生成/补全/调整任务规划（LLM链式思维）
    2. 缓存规划结果
    3. 自动记录对话历史，便于溯源
    """
    def __init__(self, llm, cache, memory):
        self.llm = llm
        self.cache = cache
        self.memory = memory

    def _extract_chat_history(self) -> List[str]:
        """提取本轮会话的对话历史，格式化为字符串列表"""
        messages = self.memory.chat_memory.messages
        history = []
        for m in messages:
            if m.type == "human":
                history.append(f"用户：{m.content}")
            elif m.type == "ai":
                history.append(f"助手：{m.content}")
        return history

    def generate_plan(self, session_id, intermediate_state, current_plan, user_input):
        # 记录本轮用户输入
        self.memory.save_context({"input": user_input}, {"output": ""})

        # 组装完整链式Prompt
        chat_history = self._extract_chat_history()
        prompt = build_context_prompt(chat_history, intermediate_state, current_plan, user_input)

        print("\n===== 当前Prompt内容（供Debug） =====\n")
        print(prompt)
        print("\n===== LLM输出 =====\n")
        # 用 LLM 生成最新规划
        response = self.llm.invoke(prompt).content
        print(response)

        # 解析、缓存
        new_plan = safe_parse(extract_json(response))
        self.cache.set(session_id, new_plan)

        # 回写本轮LLM输出，确保内存完整（便于历史追踪）
        self.memory.chat_memory.messages[-1].content = response
        return new_plan

# ========= 8. 主流程/测试用例 =========
if __name__ == '__main__':
    # 初始化LangChain会话内存
    memory = ConversationBufferMemory(return_messages=True)
    # 初始化LLM，实际业务可切换不同模型
    test_llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
    )
    # 初始化缓存与Agent
    cache = SimpleCache()
    planner = ContextPlanner(llm=test_llm, cache=cache, memory=memory)

    session_id = "abc001"

    # 预置历史：可根据业务实际情况初始化部分历史
    memory.save_context({"input": "我要一个原始的缺陷趋势数据"}, {"output": """
    json
    [
      {
        "id": 1,
        "tool_name": "load_data",
        "description": "加载A产品检测数据",
        "type": "serial"
      },
      {
        "id": 2,
        "tool_name": "run_model",
        "description": "运行缺陷检测模型",
        "type": "serial"
      }
    ]

    是否需要添加报表功能？"""})
    # 新的用户请求（实际项目可来自API、UI等）
    user_input = "请把非法数据清洗掉、并且给我生成趋势图和报告"

    # 当前任务状态和历史规划（json格式，便于追溯）
    intermediate_state = "state='进行中'"
    current_plan = json.dumps([
        {"id": 1, "tool_name": "load_data", "description": "加载A产品检测数据", "type": "serial"},
        {"id": 2, "tool_name": "run_model", "description": "运行缺陷检测模型", "type": "serial"}
    ], ensure_ascii=False, indent=2)

    # 调用链式Agent生成并调整任务规划
    final_plan = planner.generate_plan(session_id, intermediate_state, current_plan, user_input)

    print("\n=== 生成的最新任务规划（对象结构） ===")
    for n in final_plan:
        print(n)

    # 下面演示“自动驱动函数流”——即根据LLM规划，自动调用注册工具
    print("\n=== 自动调用各环节业务函数，演示执行 ===")
    context_test = {}
    for node in final_plan:
        # 支持串行/并行（示例只用串行），实际业务可拓展
        if isinstance(node.tool_name, dict):
            # 并行场景下依次执行每个工具
            for tname in node.tool_name.keys():
                func = registry[tname][0]
                context_test = func(context_test)
        else:
            func = registry[node.tool_name][0]
            context_test = func(context_test)
    print("\n=== 最终执行上下文 ===")
    print(context_test)

    # 可选：对话历史持久化，方便复盘
    history_data = [
        {
            "type": msg.type,
            "content": msg.content
        }
        for msg in memory.chat_memory.messages
    ]
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    print("已保存对话历史到 chat_history.json")

