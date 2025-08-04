# 测试用的插件函数
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


# 函数注册字典：仅在内存中
TOOL_FUNC_MAP = {
    "load_data": load_data,
    "clean_data": clean_data,
    "run_model": run_model,
}
