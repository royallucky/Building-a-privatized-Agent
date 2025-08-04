import json
from typing import Literal, List

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from pydantic import BaseModel, ValidationError, RootModel

import re

instruction_template = """你是 Task Planner，负责把用户的自然语言需求转成可执行任务规划。

【约束】
1. 任务规划必须满足字段：id、tool、description、type。
2. tool 字段必须存在于下面的【工具列表】中。
3. 输出必须是【合法JSON】，不包含多余文本。

【工具列表】
- load_data：加载原始数据
- clean_data：数据清洗与标准化
- run_model：缺陷检测模型推理
- draw_trend：生成缺陷趋势图
- send_report：发送报告邮件

【示例输入】
请统计并汇总近 7 天 A 产品的缺陷趋势

【示例输出】
[
  {"id": 1, "tool_name": "load_data",  "description": "加载 A 产品近 7 天检测数据", "type": "serial"},
  {"id": 2, "tool_name": "clean_data", "description": "清洗检测数据", "type": "serial"},
  {"id": 3, "tool_name": "run_model", "description": "缺陷推理并统计", "type": "serial"},
  {"id": 4, "tool_name": "draw_trend", "description": "绘制缺陷趋势图", "type": "serial"},
  {"id": 5, "tool_name": "send_report", "description": "邮件发送分析报告", "type": "serial"}
]

【正式任务】
{{user_input}}"""


# 假设这些函数是执行某些任务的实际功能
def load_data(context):
    print("加载数据...")
    context["raw_data"] = [1, 2, 3]
    return context

def clean_data(context):
    print(f"清洗数据: {context['raw_data']}")
    context["cleaned_data"] = [x for x in context["raw_data"] if x > 1]
    context["model_result"] = context["cleaned_data"]
    return context

def run_model(context):
    print(f"运行模型: 输入数据 {context['cleaned_data']}")
    context["model_result"] = sum(context["cleaned_data"])
    return context

def draw_trend(context):
    print(f"绘制趋势图: 基于结果 {context['model_result']}")
    context["trend"] = f"trend_chart_based_on_{context['model_result']}"
    return context

def draw_distribution(context):
    print(f"绘制分布图: 基于结果 {context['model_result']}")
    context["distribution"] = f"distribution_chart_based_on_{context['model_result']}"
    return context

def combine_outputs(context):
    print(f"合并输出数据: {context}")
    context["final_report"] = {
        "trend": context["trend"],
        "distribution": context["distribution"]
    }
    return context

# 创建 registry 数据，将工具名称和对应函数映射
registry = {
    "load_data": load_data,
    "clean_data": clean_data,
    "run_model": run_model,
    "draw_trend": draw_trend,
    "draw_distribution": draw_distribution,
    "combine_outputs": combine_outputs
}

# 测试打印 registry，验证工具是否正确注册
print(registry)


def extract_json(content: str) -> str:
    """
    使用正则裁剪LLM输出中的无关文本，保留第一个有效的JSON片段。
    """
    # 匹配第一个以 "[" 开始，以 "]" 结束的 JSON 数据
    match = re.search(r'(\[.*\])', content, re.DOTALL)
    if match:
        return match.group(1)  # 返回匹配的 JSON 片段
    else:
        raise ValueError("未找到有效的 JSON 数据")


# 定义任务节点数据模型
class PlanNode(BaseModel):
    id: int  # 任务ID
    tool_name: str  # 工具名称
    description: str  # 任务描述
    type: Literal["serial", "parallel"]  # 任务类型，串行或并行

# 定义任务规划数据结构，继承自 RootModel
class Plan(RootModel):
    root: List[PlanNode]


def safe_parse(json_str: str) -> List[PlanNode]:
    """
    安全地解析任务规划 JSON 数据，并进行校验。
    如果校验失败，抛出异常并提示错误。
    """
    try:
        # 使用 json.loads() 将 JSON 字符串转换为 Python 对象（列表）
        parsed_data = json.loads(json_str)
        print(parsed_data)
        # 使用 Pydantic 的 model_validate 校验数据结构
        return Plan.model_validate(parsed_data).root  # 返回任务节点列表
    except ValidationError as e:
        # 如果校验失败，抛出异常并提供详细的错误信息
        raise RuntimeError(f"LLM 输出格式错误: {e}")
    except json.JSONDecodeError as e:
        # 如果 JSON 格式有问题，抛出解析错误
        raise RuntimeError(f"JSON 解析失败: {e}")


def build_flow(plan: List[PlanNode]):
    """
    根据任务规划节点生成执行流。
    - 当 type 为 parallel 时，将同级任务聚合到 RunnableParallel；
    - 根据 id 和依赖关系动态构建 DAG。
    """
    runnables = []  # 存储最终的任务流对象（串行或并行）
    parallel_tasks = {}  # 临时存储一批并行任务（用字典收集）

    for idx, node in enumerate(plan):
        if node.type == "parallel":
            # 支持工具名为 dict 或 str 两种格式
            if isinstance(node.tool_name, dict):
                tool_keys = list(node.tool_name.values())  # 如果是 dict，取所有 value（如多个并行子任务）
            else:
                tool_keys = [node.tool_name]  # 如果是 str，只转换为单元素列表

            # 为每一个并行工具创建 RunnableLambda，并放入 parallel_tasks
            for i, key in enumerate(tool_keys):
                run_func = registry[key]  # 根据工具名获取具体的函数
                current_runnable = RunnableLambda(run_func)
                # 以 key 和下标作为唯一标识，存入并行任务字典
                parallel_tasks[f"{key}_{i}"] = current_runnable

            # 判断是不是并行任务的末尾（即下一个不是 parallel 或已到结尾）
            is_last = idx == len(plan) - 1
            next_is_not_parallel = not (not is_last and plan[idx + 1].type == "parallel")
            if is_last or next_is_not_parallel:
                # 遇到并行组结尾，将这批并行任务整体加入 runnables
                runnables.append(RunnableParallel(parallel_tasks))
                parallel_tasks = {}  # 清空，为下一个并行组做准备

        else:  # 如果是串行任务
            # 如果前面还积压了并行任务，先将它们加进 runnables
            if parallel_tasks:
                runnables.append(RunnableParallel(parallel_tasks))
                parallel_tasks = {}

            # 直接处理串行任务
            run_func = registry[node.tool_name]
            current_runnable = RunnableLambda(run_func)
            runnables.append(current_runnable)

    # 返回一个按顺序执行的任务流
    return RunnableSequence(*runnables)


class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.template = instruction_template

    def _build_prompt(self, user_input: str) -> str:
        return self.template.replace("{{user_input}}", user_input)

    def generate_plan(self, user_input: str):
        prompt = self._build_prompt(user_input)
        print(prompt)
        print("*" * 100)
        response = self.llm.invoke(prompt).content
        print(response)
        print("*" * 100)
        json_part = extract_json(response)  # 自定义提取函数
        # 解析并校验 JSON 数据
        try:
            plan = safe_parse(json_part)
            print("任务规划校验通过，任务节点：", plan)
            print("*" * 100)
            flow = build_flow(plan=plan)
            flow.invoke({"raw_data": []})  # 此处由于load_data模拟了生成数据，因此需要在输入设置占位的空数据
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    test_llm = ChatOllama(
        model="llama3.2",  # 换成你的模型
        temperature=0.3,
        system="你是严谨的软件工程助手，所有规划以 JSON 数组格式输出"
    )
    PL = Planner(llm=test_llm)
    PL.generate_plan(user_input="读取数据库中的销售数据，清洗掉异常值，画出分布图并保存结果")
