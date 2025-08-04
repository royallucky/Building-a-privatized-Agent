from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
from langchain.tools import Tool

# 定义插件函数
def load_data(context):
    print("加载数据...")
    # 模拟从外部获取原始数据
    context["raw_data"] = [1, 2, 3]
    return context

def clean_data(context):
    print(f"清洗数据: {context['raw_data']}")
    # 清洗数据，过滤掉小于或等于1的数据
    context["cleaned_data"] = [x for x in context["raw_data"] if x > 1]
    return context

def run_model(context):
    print(f"运行模型: 输入数据 {context['cleaned_data']}")
    # 简单的模型运行逻辑，计算清洗后的数据和
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
    # 将不同的图表结果合并到上下文中
    context["final_report"] = {
        "trend": context["trend"],
        "distribution": context["distribution"]
    }
    return context

# 创建Tool注册功能
tool_registry = {}

def register_tool(name, description, func):
    tool_registry[name] = Tool(
        name=name,
        description=description,
        func=func
    )

def get_tool(name):
    return tool_registry.get(name)

# 注册所有插件工具
register_tool("load_data", "加载原始数据", load_data)
register_tool("clean_data", "清洗数据", clean_data)
register_tool("run_model", "运行模型", run_model)
register_tool("draw_trend", "生成趋势图", draw_trend)
register_tool("draw_distribution", "生成分布图", draw_distribution)
register_tool("combine_outputs", "合并输出结果", combine_outputs)

print(tool_registry)

# 模拟LLM生成的任务规划（动态化）
def generate_task_planning(con):
    """
    模拟LLM根据上下文生成任务规划
    """
    if con.get("task_type") == "simple":
        return [
            {"task_id": "load_data", "tool_name": "load_data", "type": "serial"},
            {"task_id": "clean_data", "tool_name": "clean_data", "type": "serial"},
            {"task_id": "run_model", "tool_name": "run_model", "type": "serial"},
            {"task_id": "combine_results", "tool_name": "combine_outputs", "type": "serial"}
        ]
    elif con.get("task_type") == "complex":
        return [
            {"task_id": "load_data", "tool_name": "load_data", "type": "serial"},
            {"task_id": "clean_data", "tool_name": "clean_data", "type": "serial"},
            {"task_id": "run_model", "tool_name": "run_model", "type": "serial"},
            {"task_id": "generate_charts", "tool_name": {
                "trend": "draw_trend",
                "distribution": "draw_distribution"
            }, "type": "parallel"},
            {"task_id": "combine_results", "tool_name": "combine_outputs", "type": "serial"}
        ]
    elif con.get("task_type") == "mixed":
        return [
            {"task_id": "load_data", "tool_name": "load_data", "type": "serial"},
            {"task_id": "clean_data", "tool_name": "clean_data", "type": "serial"},
            {"task_id": "generate_charts", "tool_name": {
                "trend": "draw_trend",
                "distribution": "draw_distribution"
            }, "type": "parallel"},
            {"task_id": "run_model", "tool_name": "run_model", "type": "serial"},
            {"task_id": "combine_results", "tool_name": "combine_outputs", "type": "serial"}
        ]
    else:
        return []

# 模拟LLM根据不同的配置生成任务规划
con = {"task_type": "complex"}  # 可以设置为"simple"、"complex"或"mixed"来改变任务流
# 根据上下文生成动态的任务规划
task_planning = generate_task_planning(con)

# 确保任务规划不为空且至少有2个步骤
if len(task_planning) < 2:
    print("任务规划为空或任务步骤不足")
else:
    # 创建任务流
    flow = RunnableSequence(
        RunnableLambda(get_tool(task_planning[0]['tool_name']).func)  # 加载数据
        | RunnableLambda(get_tool(task_planning[1]['tool_name']).func)
    )

    # 动态构建任务流
    for task in task_planning[2:]:
        if task["type"] == "parallel":  # 如果是并行任务
            # 将并行任务按需创建
            parallel_tasks = {
                key: RunnableLambda(get_tool(value).func)
                for key, value in task["tool_name"].items()
            }
            flow = flow | RunnableParallel(parallel_tasks)  # 并行任务
        else:  # 顺序执行任务
            flow = flow | RunnableLambda(get_tool(task["tool_name"]).func)

    # 执行流程
    print("开始执行流程...\n")
    output = flow.invoke({"raw_data": []})  # 传递初始上下文
    print("\n流程执行完毕，输出结果:")
    print(output)
