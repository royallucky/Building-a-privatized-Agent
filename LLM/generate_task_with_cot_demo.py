"""
本示例代码演示如何用 LangChain 实现“任务规划-参数确认-模块执行-对话追踪”的完整流程。
适合 AI 业务小白参考。代码中有模拟对话、模拟数据的部分，实际项目中可替换为真实逻辑。
"""

import re
import json

# ================= 1. LangChain/LLM 基础依赖导入 =================
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# ================= 2. 工具/一级功能模块注册 =================
def load_data(context):
    """
    【模拟】数据加载模块：实际项目请替换为真实的数据读取代码
    """
    print(f"[模块] 加载数据源: {context['filepath']}")
    # 这里用的是模拟数据，实际请用 pd.read_csv、数据库读取等
    context["raw_data"] = [1, 2, 3, 4, 5]
    return context

def analysis_data(context):
    """
    【模拟】数据分析模块：实际项目请替换为真实的数据分析/模型推理代码
    """
    print(f"[模块] 数据分析: {context['raw_data']}")
    avg = sum(context["raw_data"]) / len(context["raw_data"])
    context["analysis_result"] = {"avg": avg}
    return context

# 注册功能模块，实际业务可继续添加新模块
registry = {
    "LoadData": [load_data, "加载数据用的一级功能模块"],
    "AnalysisData": [analysis_data, "分析数据用的一级功能模块"],
    # "CleanData": [...], "DrawData": [...], ...
}

# ================= 3. LangChain 对话历史内存 =================
memory = ConversationBufferMemory(memory_key="chat_history")

# ================= 4. 任务拆解 Prompt =================
# 让 LLM 直接输出分步骤目标及其功能模块归属
task_goal_prompt = """
你是一个智能任务规划Agent，负责根据用户的需求拆解任务目标并生成具体的步骤。
根据需求拆解任务目标，并用JSON格式返回任务目标列表。每个任务目标应包含目标名称、描述和一级功能模块名称。

用户需求：{user_request}

【一级功能模块】
- LoadData: 加载数据用的一级功能模块
- CleanData: 清洗数据用的一级功能模块
- AnalysisData: 专门对数据进行分析的一级功能模块
- DrawData: 绘制图、表的一级功能模块
- Report: 专门写报告的一级功能模块
- SendInfo: 对外发送报告的一级功能模块

【字段说明】
- target_name: 任务目标的名称
- description: 任务目标的描述
- model_name: 对应【一级功能模块】中的模块名称

【约束规则】
输出为JSON内容，不包含无关文本
"""

# ================= 5. 各功能模块的参数确认 Prompt =================
# 实际开发时可将每个模块的 prompt 拆成独立 py 文件进行复用和维护
load_data_prompt = """
你现在负责确定加载数据任务需要的所有参数，请根据上下文确认参数需求，并用JSON返回所有参数的名称和描述。
【工具功能】{tool_desc}
【任务目标描述】{task_goal}
【上下文历史】{chat_history}

输出要求：
- 以JSON对象返回所有需要的参数名称和参数说明，不包含无关文本
"""

analysis_data_prompt = """
你现在负责确定分析数据任务需要的所有参数，请根据上下文确认参数需求，并用JSON返回所有参数的名称和描述。
【工具功能】{tool_desc}
【任务目标描述】{task_goal}
【上下文历史】{chat_history}

输出要求：
- 以JSON对象返回所有需要的参数名称和参数说明，不包含无关文本
"""

# ================= 6. LLM JSON 输出解析辅助函数 =================
def extract_json(content: str, first=False) -> str:
    """
    【通用工具】正则提取 LLM 回复中的 JSON 数组，兼容 LLM 输出带前后无关文本的情况
    """
    if not first:
        matches = re.findall(r'\[.*?\]', content, re.DOTALL)
        if matches:
            return matches[-1]
        else:
            raise ValueError("未找到有效的 JSON 数据")
    else:
        start = content.find("[")
        if start == -1:
            raise ValueError("未找到 [ 起始标志")
        bracket_count = 0
        for i in range(start, len(content)):
            if content[i] == "[":
                bracket_count += 1
            elif content[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return content[start:i + 1]
        raise ValueError("JSON 数组未闭合，无法提取")

# ================= 7. 任务目标拆解函数 =================
def decompose_task(user_request):
    """
    调用 LLM 拆解复杂需求为结构化子任务，每个任务绑定功能模块
    """
    prompt = task_goal_prompt.format(user_request=user_request)
    # 用 LLMChain 封装 LLM 调用
    chain = LLMChain(
        llm=ChatOllama(model="llama3.2", temperature=0.3),
        prompt=PromptTemplate.from_template(prompt)
    )
    return chain.invoke({})['text']

# ================= 8. 功能模块参数交互（可对接真实/模拟 LLM） =================
def confirm_module_params(module_name, tool_desc, task_goal, chat_history):
    """
    与 LLM 对话确认功能模块需要的参数结构；
    实际业务中可接前端表单、API、后续多轮 LLM 交互等方式完善
    """
    # 根据模块名动态选择不同的参数确认 prompt
    if module_name == "LoadData":
        prompt_tmpl = load_data_prompt
    elif module_name == "AnalysisData":
        prompt_tmpl = analysis_data_prompt
    else:
        raise NotImplementedError(f"{module_name} 参数确认未实现")

    prompt = PromptTemplate(
        input_variables=["tool_desc", "task_goal", "chat_history"],
        template=prompt_tmpl
    )
    chain = LLMChain(
        llm=ChatOllama(model="llama3.2", temperature=0.2),
        prompt=prompt
    )
    # 用 LLMChain.invoke 统一注入参数
    response = chain.invoke({
        "tool_desc": tool_desc,
        "task_goal": task_goal,
        "chat_history": chat_history
    })['text']
    # 提取并解析 JSON 对象
    match = re.search(r'{[\s\S]*}', response)
    if match:
        return json.loads(match.group())
    else:
        raise ValueError("未找到有效参数JSON")

# ================= 9. 主流程入口（任务规划与执行） =================
def interact_and_generate_plan(user_request):
    """
    主流程入口：1. 拆解任务目标 2. LLM参数确认 3. 实际执行 4. 全流程对话追踪
    """
    # -------- (1) 任务目标拆解 --------
    task_goals_json_ = decompose_task(user_request)
    task_goals_json = extract_json(task_goals_json_, first=False)
    print("[拆解结果]:", task_goals_json)
    memory.save_context({"input": user_request}, {"output": task_goals_json_})
    task_goals = json.loads(task_goals_json)

    # -------- (2) 依次处理每个一级功能模块 --------
    context = {}  # 承载数据流的上下文
    for goal in task_goals:
        module_name = goal["model_name"]
        tool_func, tool_desc = registry.get(module_name, (None, None))
        if tool_func is None:
            print(f"未注册模块 {module_name}，跳过")
            continue

        # 【重要】模拟对话历史，实际可接真实对话/用户输入
        chat_history = json.dumps([
            {"role": msg.type, "content": msg.content}
            for msg in memory.chat_memory.messages
        ], ensure_ascii=False)

        # -------- (2.1) 参数确认环节（可对接UI/多轮LLM） --------
        param_info = confirm_module_params(
            module_name, tool_desc, goal["description"], chat_history
        )
        print(f"\n[模块:{module_name}] 需要参数：", param_info)

        # -------- (2.2) 参数赋值（【模拟】此处只做演示，实际请替换成前端/真实输入） --------
        if module_name == "LoadData":
            context["filepath"] = "demo_sales.csv"  # 业务场景下请用真实路径
        elif module_name == "AnalysisData":
            # AnalysisData 要依赖 raw_data（由上一步输出），无需手动赋值
            pass

        # -------- (2.3) 工具/模块实际调用 --------
        context = tool_func(context)

        # -------- (2.4) 记录每步对话/操作到内存 --------
        memory.save_context(
            {"input": f"{module_name}参数确认"},
            {"output": f"参数：{json.dumps(param_info, ensure_ascii=False)}，执行结果：{context}"}
        )

    return context

# ================= 10. 流程调试/演示入口 =================
if __name__ == "__main__":
    # ========== 【业务模拟】可换成自己的业务需求 ==========
    user_request = "请加载本地的销售数据，并分析数据的平均值。"
    final_context = interact_and_generate_plan(user_request)

    print("\n[最终流程执行结果]:")
    print(json.dumps(final_context, ensure_ascii=False, indent=2))

    # ========== 【模拟】保存所有对话历史，方便溯源 ==========
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
