import json
import re
from typing import List, Literal, Union

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from pydantic import ValidationError, BaseModel, RootModel

samples_template = """你是 Task Planner，负责将业务请求转为可执行任务规划。
【字段说明】
- id：任务编号，按顺序递增
- tool_name：工具名称，串行为字符串，并行为字典（如{"a":"a", "b":"b"}）
- description：任务描述
- type：任务类型，仅能为 "serial" 或 "parallel"

【工具列表】
{{tool_registry}}

【示例 1】
请求：分析近 7 天 A 产品缺陷趋势
输出：
[
  {"id": 1, "tool_name": "load_data", "description": "加载 A 产品近 7 天检测数据", "type": "serial"},
  {"id": 2, "tool_name": "run_model", "description": "运行缺陷检测模型", "type": "serial"},
  {"id": 3, "tool_name": "draw_trend", "description": "绘制缺陷趋势图", "type": "serial"}
]

【示例 2】
请求：清洗并分析所有 C 产品的良率数据
输出：
[
  {"id": 1, "tool_name": "load_data", "description": "加载 C 产品良率数据", "type": "serial"},
  {"id": 2, "tool_name": "clean_data", "description": "清洗并标准化数据", "type": "serial"},
  {"id": 3, "tool_name": {"run_model": "run_model"}, "description": "统计分析良率波动", "type": "parallel"}
]

【示例 3】
请求：生成近一月检测数据报告并发送邮件
输出：
[
  {"id": 1, "tool_name": "load_data", "description": "加载数据", "type": "serial"},
  {"id": 2, "tool_name": "clean_data", "description": "清洗数据", "type": "serial"},
  {"id": 3, "tool_name": 
    {"draw_distribution": "draw_distribution", 
     "draw_trend": "draw_trend"}, 
    "description": "绘制图表", "type": "parallel"},
  {"id": 4, "tool_name": "combine_outputs", "description": "发送报告", "type": "serial"}
]

【输出要求】
- 仅输出一个合法的 JSON 数组，不要有任何多余文本、注释或解释
- 每个任务都要包含 id、tool_name、description、type 字段
- tool_name 可为字符串（串行）或字典（并行）
- type 仅能为 "serial" 或 "parallel"
- 输出格式必须与上方示例完全一致

【用户请求】
{{user_input}}

【输出】
（请严格按照示例格式，仅输出 JSON 数组，不要任何其他内容。）
"""

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
    "load_data": [load_data, "加载数据"],
    "clean_data": [clean_data, "清洗数据"],
    "run_model": [run_model, "缺陷检测模型推理"],
    "draw_trend": [draw_trend, "生成缺陷趋势图"],
    "draw_distribution": [draw_distribution, "生成分布图"],
    "combine_outputs": [combine_outputs, "合并数据并输出结果"]
}

# 测试打印 registry，验证工具是否正确注册
print(registry)


def extract_json(content: str) -> str:
    """
    使用正则裁剪LLM输出中的无关文本，保留最后一个有效的JSON片段。
    """
    # 匹配所有以 "[" 开始，以 "]" 结束的 JSON 数据
    matches = re.findall(r'\[.*?\]', content, re.DOTALL)
    if matches:
        return matches[-1]  # 返回最后一个匹配的 JSON 片段
    else:
        raise ValueError("未找到有效的 JSON 数据")


# 定义任务节点数据模型
class PlanNode(BaseModel):
    id: int  # 任务ID
    tool_name: Union[str, dict]  # 工具名称
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
    - 连续的 parallel 节点作为一组并行，遇到 serial 时自动flush；
    - 每个 serial 节点单独执行。
    :param plan: List[PlanNode]，任务节点列表
    :return: RunnableSequence，任务执行流
    """
    runnables = []            # 存储最终要执行的任务流
    parallel_tasks = {}       # 临时存储一组并行任务

    for idx, node in enumerate(plan):
        # 判断当前节点是不是并行任务
        if node.type == "parallel":
            # 处理工具名支持 dict（如 {"draw_trend":"draw_trend"}）和 str 两种格式
            if isinstance(node.tool_name, dict):
                tool_key = list(node.tool_name.values())  # 取所有 value（多个并行子任务）
            else:
                tool_key = [node.tool_name]              # 如果是 str，转为长度为1的列表

            # 为每个并行工具创建对应的执行单元
            for i in range(len(tool_key)):
                run_func = registry[tool_key[i]][0]      # 从注册表拿到对应的函数
                current_runnable = RunnableLambda(run_func)
                # 存入字典，key带上下标便于区分
                parallel_tasks[f"{tool_key}_{i}"] = current_runnable

            # 检查：并行组是否结束（后面不是 parallel 或到最后一个节点）
            is_last = idx == len(plan) - 1
            next_is_not_parallel = not (not is_last and plan[idx + 1].type == "parallel")
            if is_last or next_is_not_parallel:
                # 并行组到头了，把当前所有并行任务打包放进主流
                runnables.append(RunnableParallel(parallel_tasks))
                parallel_tasks = {}  # 清空，准备下一组并行

        else:
            # 遇到串行节点，先把前面的并行组（如果有）加入主流
            if parallel_tasks:
                runnables.append(RunnableParallel(parallel_tasks))
                parallel_tasks = {}

            # 直接将串行节点转为RunnableLambda并加入
            run_func = registry[node.tool_name][0]
            current_runnable = RunnableLambda(run_func)
            runnables.append(current_runnable)

    # 返回所有组装好的任务流（串行/并行自动按顺序组织）
    return RunnableSequence(*runnables)


class Planner:
    def __init__(self, llm):
        self.llm = llm
        self.template = samples_template

    def _build_prompt(self, user_input: str) -> str:
        self.template = self.template.replace("{{tool_registry}}", str("\n".join([f"- {k}：{v[-1]}" for k, v in registry.items()])))
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
        temperature=0.3
    )
    PL = Planner(llm=test_llm)
    PL.generate_plan(user_input="读取数据库中的销售数据，清洗掉异常值，画出分布图并输出结果")
