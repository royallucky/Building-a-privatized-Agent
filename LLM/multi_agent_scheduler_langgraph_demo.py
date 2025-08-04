import operator
import time
from typing import TypedDict, List, Dict, Annotated
from functools import partial
from langgraph.graph import StateGraph, END

# ===================
# 1. Agent与注册表
# ===================

class Agent:
    def __init__(self, agent_id: str, name: str = "UnnamedAgent", capabilities: list = None, config: dict = None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities if capabilities else []
        self.config = config if config else {}
        self.is_registered = False

    def register(self, registry):
        if not self.is_registered:
            registry.register_agent(self)
            self.is_registered = True

    def describe(self):
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "config": self.config
        }

    def execute(self, task_context):
        print(f"{self.name} 开始执行任务")
        timeout = self.config.get("timeout", 1)
        # 模拟业务处理
        if self.name == "DataAnalysisAgent":
            time.sleep(timeout)
        task_context['status'] = "success"
        task_context['outputs'] = f"{self.name} 处理完成输出"
        print(f"{self.name} 任务完成")
        return task_context

class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register_agent(self, agent: Agent):
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent ID '{agent.agent_id}' 已存在，不能重复注册。")
        self._agents[agent.agent_id] = agent
        print(f"已注册Agent: {agent.describe()}")

    def get_agent(self, agent_id):
        return self._agents.get(agent_id)

    def find_by_capability(self, capability: str):
        return [a for a in self._agents.values() if capability in a.capabilities]

    def list_agents(self):
        return list(self._agents.values())

# 业务Agent实现
class DataCleaningAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-001",
            name="DataCleaningAgent",
            capabilities=["clean_data", "preprocessing"],
            config={"timeout": 1, "priority": 1}
        )

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-002",
            name="DataAnalysisAgent",
            capabilities=["analyze_data", "statistics", "feature_extraction"],
            config={"timeout": 1, "priority": 1}
        )

class ReportAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-003",
            name="ReportAgent",
            capabilities=["generate_report", "export_pdf", "summarize"],
            config={"timeout": 1, "priority": 1}
        )

class SendAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-004",
            name="SendAgent",
            capabilities=["send_report", "email", "wechat"],
            config={"timeout": 1, "priority": 1}
        )

# ================
# 2. DAG结构定义
# ================

dag_tasks = [
    {"id": "t1", "type": "clean_data", "payload": "数据A", "next": ["t2", "t3"], "agent_type": "clean_data"},
    {"id": "t2", "type": "analyze_data", "payload": "数据B", "next": ["t4"], "agent_type": "analyze_data"},
    {"id": "t3", "type": "generate_report", "payload": "数据C", "next": ["t4"], "agent_type": "generate_report"},
    {"id": "t4", "type": "send_report", "payload": "数据D", "next": [], "agent_type": "send_report"}
]

task_map = {task["id"]: task for task in dag_tasks}
dependencies: Dict[str, set] = {}
for task in dag_tasks:
    for next_task in task["next"]:
        dependencies.setdefault(next_task, set()).add(task["id"])

# 注册所有Agent
registry = AgentRegistry()
clean_agent = DataCleaningAgent()
analyze_agent = DataAnalysisAgent()
report_agent = ReportAgent()
send_agent = SendAgent()
for agent in [clean_agent, analyze_agent, report_agent, send_agent]:
    agent.register(registry)

# 定义“type”到Agent的能力的映射
type2cap = {
    "clean_data": "clean_data",
    "analyze_data": "analyze_data",
    "generate_report": "generate_report",
    "send_report": "send_report"
}

# =====================
# 3. 状态定义
# =====================
class TaskState(TypedDict):
    completed_tasks: Annotated[List[str], operator.add]
    pending_tasks: Annotated[List[str], operator.add]
    task_results: Annotated[Dict[str, str], operator.or_]
    dag_definition: Dict[str, set]

# =====================
# 4. 节点业务实现（核心Agent驱动）
# =====================

def task_router(state: TaskState, task_id: str):
    if task_id in state["completed_tasks"]:
        return {}
    task = task_map[task_id]
    agent_cap = type2cap[task["type"]]
    # 通过能力找Agent
    agents = registry.find_by_capability(agent_cap)
    if not agents:
        raise ValueError(f"没有可执行 {task['type']} 能力的Agent")
    agent = agents[0]  # 这里简单选第一个，有多Agent可以做复杂调度
    print(f"\n 调用Agent: {agent.name} 执行任务: {task_id} ({task['type']})")
    # 执行真实的Agent
    ctx = {
        "task_id": task_id,
        "payload": task["payload"],
        "status": "pending",
        "outputs": None
    }
    ctx = agent.execute(ctx)
    result = ctx["outputs"]  # 这里直接用Agent输出
    updated_results = state["task_results"].copy()
    updated_results[task_id] = result
    updated_completed = state["completed_tasks"] + [task_id]
    return {
        "task_results": updated_results,
        "completed_tasks": updated_completed,
        "pending_tasks": [],
    }

# ==========================================
# 5. 全局依赖补全节点
# ==========================================
def refresh_pending_tasks(state: TaskState):
    completed_set = set(state["completed_tasks"])
    pending_set = set(state["pending_tasks"])
    add_pending = []
    for t in task_map:
        if t in completed_set or t in pending_set:
            continue
        pre_deps = state["dag_definition"].get(t, set())
        if pre_deps and pre_deps.issubset(completed_set):
            add_pending.append(t)
    if add_pending:
        print(f"\n[依赖扫描] 新可调度任务：{add_pending}")
    return {"pending_tasks": add_pending}

# ===============================
# 6. 起始节点、调度节点
# ===============================
def start_node(state: TaskState):
    return {
        "task_results": {},
        "completed_tasks": [],
        "pending_tasks": ["t1"],
        "dag_definition": dependencies
    }

def decide_next(state: TaskState):
    ready = [t for t in set(state["pending_tasks"]) if t not in state["completed_tasks"]]
    return ready if ready else END

# ===============================
# 7. 构建LangGraph DAG
# ===============================
workflow = StateGraph(TaskState)
for task in dag_tasks:
    workflow.add_node(task["id"], partial(task_router, task_id=task["id"]))
workflow.set_entry_point("start")
workflow.add_node("start", start_node)
workflow.add_node("dispatch_tasks", lambda state: state)
workflow.add_node("refresh_pending_tasks", refresh_pending_tasks)
workflow.add_edge("start", "dispatch_tasks")
workflow.add_conditional_edges("dispatch_tasks", decide_next)
for task in dag_tasks:
    workflow.add_edge(task["id"], "refresh_pending_tasks")
workflow.add_edge("refresh_pending_tasks", "dispatch_tasks")

app = workflow.compile()

# ===============================
# 8. 可视化DAG结构（Graphviz）
# ===============================
def visualize_dag(dag_tasks):
    try:
        from graphviz import Digraph
        dot = Digraph(comment='DAG执行流程', format='png')
        for task in dag_tasks:
            label = f"{task['id']}\\n{task['type']}"
            dot.node(task['id'], label)
        for task in dag_tasks:
            for next_task in task['next']:
                dot.edge(task['id'], next_task)
        dot.render('dag_example', view=False)
    except Exception as e:
        print("可视化失败：", e)

if __name__ == "__main__":
    visualize_dag(dag_tasks)
    initial_state: TaskState = {
        "task_results": {},
        "completed_tasks": [],
        "pending_tasks": [],
        "dag_definition": dependencies
    }
    # print("\n=== DAG任务执行过程 ===\n")
    # for step in app.stream(initial_state):
    #     for node, result in step.items():
    #         if node == "dispatch_tasks":
    #             print(f"\n[调度中心] 当前待执行: {result['pending_tasks']}")
    #         elif node == "refresh_pending_tasks":
    #             pass
    #         elif node not in ("__end__", "start"):
    #             print(f"节点 {node} 完成: {result['task_results'][node]}")
    #             print(f"已完成: {result['completed_tasks']}")
    #             print(f"待处理: {result['pending_tasks']}")
    #             print("-" * 40)
    final_state = app.invoke(initial_state)
    print("\n=== 所有任务最终结果 ===")
    for tid, res in final_state["task_results"].items():
        print(f"{tid}: {res}")
