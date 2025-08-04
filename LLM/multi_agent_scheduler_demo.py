import multiprocessing
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


# 用于追踪任务的上下文结构体，包含任务ID、状态、日志、输出等信息
class TaskContext:
    def __init__(self, task_type, payload):
        self.task_id = str(uuid.uuid4())  # 生成唯一任务ID
        self.task_type = task_type  # 任务类型，例如：clean_data
        self.payload = payload  # 实际任务携带的数据
        self.start_time = None  # 任务开始时间
        self.end_time = None  # 任务结束时间
        self.status = "pending"  # 当前状态：pending / success / failed / timeout
        self.retries = 0  # 当前重试次数
        self.logs = []  # 日志记录
        self.outputs = None  # 最终输出结果

    # 添加日志并打印（用于调试和追踪）
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        print(f"[{self.task_type}][{self.task_id}] {message}")

# 通用的Agent类，每个Agent代表一个功能模块（可继承扩展）
class Agent:
    """
    通用Agent基类，实现Agent注册信息、能力描述、执行策略和核心接口规范。
    便于系统自动注册、统一管理和灵活调度。
    """

    def __init__(self,
                 agent_id: str,
                 name: str = "UnnamedAgent",
                 capabilities: list = None,
                 config: dict = None):
        """
        :param agent_id: 唯一标识，系统内不允许重复
        :param name: Agent名称（人类可读）
        :param capabilities: 能力标签列表，描述Agent擅长处理的任务类型
        :param config: 执行策略与配置信息，如超时、优先级、最大并发数等
        """
        self.agent_id = agent_id  # 必须唯一
        self.name = name
        self.capabilities = capabilities if capabilities else []
        self.config = config if config else {}
        # 是否已注册到Agent注册表（由系统管理）
        self.is_registered = False

    def register(self, registry):
        """
        注册到全局Agent注册表（registry为外部统一管理的注册表对象）
        """
        if not self.is_registered:
            registry.register_agent(self)
            self.is_registered = True

    def describe(self):
        """
        返回本Agent的注册元信息，便于系统发现和调度
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "config": self.config
        }

    def execute(self, task_context, stop_event):
        """
        Agent的核心执行方法。需要子类实现实际业务逻辑。
        通用执行流程包含日志、超时响应、中断检测等机制。
        """
        task_context.log(f"{self.name} 开始执行任务")

        # 演示：模拟耗时任务（实际业务请在子类重写本方法）
        timeout = self.config.get("timeout", 5)
        if self.name == "ReportAgent":
            for i in range(timeout * 10):
                if stop_event.is_set():
                    task_context.log(f"{self.name} 收到停止信号，终止任务")
                    task_context.status = "stopped"
                    return
                time.sleep(0.1)  # 模拟耗时

        task_context.status = "success"
        task_context.outputs = f"{self.name} 处理完成输出"
        task_context.log(f"{self.name} 任务完成")

# Agent注册表
class AgentRegistry:
    """
    Agent注册表。负责管理所有已注册的Agent实例。
    支持按ID、能力等属性检索。
    """
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
        # 支持按能力标签查找Agent
        return [a for a in self._agents.values() if capability in a.capabilities]

    def list_agents(self):
        return list(self._agents.values())

# 以下三个类为业务具体的功能Agent（清洗、分析、生成报告）
class DataCleaningAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-001",                       # 唯一标识
            name="DataCleaningAgent",                       # Agent名称
            capabilities=["clean_data", "preprocessing"],    # 能力标签
            config={
                "timeout": 5,                      # 单任务最大执行时长（秒）
                "max_concurrent_tasks": 2,         # 最大并发数
                "priority": 1,                     # 优先级（数值越大优先级越高）
                "max_retries": 2,                  # 最大重试次数
                "retry_interval": 3,               # 重试间隔（秒）
                "resource_limits": {"cpu": 2, "memory": "1G"},  # 资源限制
                "execution_mode": "parallel",      # 串行或并行
                "health_check": True,              # 启用健康检查
                "logging": "INFO",                 # 日志级别
                "status_callback": None            # 状态回调
            }# 执行策略/参数
        )


class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-002",  # 唯一标识
            name="DataAnalysisAgent",  # Agent名称
            capabilities=["analyze_data", "statistics", "feature_extraction"],  # 能力标签
            config={
                "timeout": 5,  # 单任务最大执行时长（秒）
                "max_concurrent_tasks": 2,  # 最大并发数
                "priority": 1,  # 优先级（数值越大优先级越高）
                "max_retries": 2,  # 最大重试次数
                "retry_interval": 3,  # 重试间隔（秒）
                "resource_limits": {"cpu": 2, "memory": "1G"},  # 资源限制
                "execution_mode": "parallel",  # 串行或并行
                "health_check": True,  # 启用健康检查
                "logging": "INFO",  # 日志级别
                "status_callback": None  # 状态回调
            }  # 执行策略/参数
        )


class ReportAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="dataclean-003",  # 唯一标识
            name="ReportAgent",  # Agent名称
            capabilities=["generate_report", "export_pdf", "summarize"],  # 能力标签
            config={
                "timeout": 5,  # 单任务最大执行时长（秒）
                "max_concurrent_tasks": 2,  # 最大并发数
                "priority": 1,  # 优先级（数值越大优先级越高）
                "max_retries": 2,  # 最大重试次数
                "retry_interval": 3,  # 重试间隔（秒）
                "resource_limits": {"cpu": 2, "memory": "1G"},  # 资源限制
                "execution_mode": "parallel",  # 串行或并行
                "health_check": True,  # 启用健康检查
                "logging": "INFO",  # 日志级别
                "status_callback": None  # 状态回调
            }  # 执行策略/参数
        )


# 多进程执行函数，必须是模块级函数，不能写在类或函数内部
def wrapped_execute(agent, task_context, stop_event, shared_status):
    agent.execute(task_context, stop_event)
    shared_status["status"] = task_context.status
    shared_status["output"] = task_context.outputs


# 控制Agent的包装器，负责执行、超时控制、重试等功能
class AgentControlWrapper:
    def __init__(self, agent, timeout=5, max_retries=3):
        self.agent = agent
        self.timeout = timeout
        self.max_retries = max_retries

    def start(self, task_context):
        retries = 0
        while retries < self.max_retries:
            task_context.retries = retries
            task_context.start_time = datetime.now()
            task_context.log(f"{self.agent.name} 第 {retries + 1} 次尝试执行")

            # 使用共享变量与事件标志进行多进程状态通信
            with multiprocessing.Manager() as manager:
                shared_status = manager.dict()
                shared_status["status"] = "pending"
                shared_status["output"] = None
                stop_event = multiprocessing.Event()

                process = multiprocessing.Process(
                    target=wrapped_execute,
                    args=(self.agent, task_context, stop_event, shared_status)
                )
                process.start()
                process.join(self.timeout)

                if process.is_alive():
                    # 超时则终止子进程
                    stop_event.set()
                    process.terminate()
                    process.join()
                    task_context.status = "timeout"
                    task_context.log(f"{self.agent.name} 执行超时（>{self.timeout}s），终止进程")
                    retries += 1
                elif shared_status["status"] == "success":
                    task_context.status = "success"
                    task_context.outputs = shared_status["output"]
                    task_context.end_time = datetime.now()
                    return task_context
                else:
                    # 任务执行失败，尝试重试
                    task_context.status = shared_status["status"]
                    task_context.log(f"{self.agent.name} 任务未成功，尝试重试")
                    retries += 1

        # 超过最大重试次数仍失败
        task_context.status = "failed"
        task_context.end_time = datetime.now()
        task_context.log(f"{self.agent.name} 重试 {self.max_retries} 次仍失败")
        return task_context


class DAGContext:
    """
    管理 DAG（有向无环图）中所有 TaskContext。
    负责解析任务依赖关系，追踪每个任务的执行状态，支持可视化 DAG 图。
    """

    def __init__(self, dag_definition):
        # 节点字典：任务ID -> TaskContext（用于记录执行状态、日志等）
        self.nodes = {}

        # 正向边：当前任务ID -> 下游任务ID列表
        self.edges = {}

        # 反向边：当前任务ID -> 所有前置任务ID列表（用于判断是否可执行）
        self.reverse_edges = {}

        # 原始的DAG结构定义（外部传入）
        self.dag_definition = dag_definition

        # 解析DAG结构
        self._parse_dag(dag_definition)

    def _parse_dag(self, dag_def):
        """
        解析DAG定义列表，建立任务节点、依赖边等信息。
        """
        for node in dag_def:
            tid = node['id']  # 当前任务ID
            task_type = node['type']  # 当前任务类型
            payload = node.get('payload')  # 输入数据

            # 为每个节点生成一个任务上下文对象
            self.nodes[tid] = TaskContext(task_type, payload)

            # 建立正向依赖（当前任务 -> 后继任务）
            self.edges[tid] = node.get('next', [])

            # 建立反向依赖（后继任务 <- 当前任务）
            for succ in node.get('next', []):
                self.reverse_edges.setdefault(succ, []).append(tid)

    def get_ready_tasks(self):
        """
        找出当前所有可以执行的任务（即所有依赖任务都已成功完成的任务）。
        """
        ready = []
        for tid, ctx in self.nodes.items():
            if ctx.status == 'pending':  # 如果任务还未执行
                deps = self.reverse_edges.get(tid, [])  # 获取其所有前置任务
                # 如果所有前置任务都已成功完成，则可以执行
                if all(self.nodes[d].status == 'success' for d in deps):
                    ready.append(tid)
        return ready

    def is_all_done(self):
        """
        检查所有任务是否都已经完成（不论成功还是失败）。
        """
        return all(ctx.status in ['success', 'failed', 'timeout'] for ctx in self.nodes.values())

    def get_summary(self):
        """
        获取所有任务的执行摘要信息，适用于后续分析或生成报表。
        """
        return [{
            "Task ID": ctx.task_id,
            "Type": ctx.task_type,
            "Status": ctx.status,
            "Retries": ctx.retries,
            "Start": ctx.start_time,
            "End": ctx.end_time,
            "Output": ctx.outputs
        } for ctx in self.nodes.values()]

    def visualize(self, with_status=True, save_path=None):
        """
        可视化DAG任务图，展示任务之间的依赖关系。
        可选显示每个任务的当前状态，并支持导出为PNG图片。

        :param with_status: 是否在图中展示任务的执行状态（如 success、pending）
        :param save_path: 如果指定了路径，则将图像保存到文件，否则直接显示
        """
        G = nx.DiGraph()  # 创建一个有向图

        # 添加任务节点
        for tid, ctx in self.nodes.items():
            label = f"{tid}\n{ctx.task_type}"
            if with_status:
                label += f"\n[{ctx.status}]"  # 可选显示状态
            G.add_node(tid, label=label)

        # 添加任务之间的依赖边（箭头）
        for from_tid, to_list in self.edges.items():
            for to_tid in to_list:
                G.add_edge(from_tid, to_tid)

        # 使用 spring 布局自动生成图坐标
        pos = nx.spring_layout(G)
        node_labels = nx.get_node_attributes(G, 'label')

        # 绘图配置
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=2000, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        plt.title("DAG Task Flow")

        # 如果提供了路径就保存图像，否则弹出图窗展示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[DAG可视化] 已保存至 {save_path}")
        else:
            plt.show()



# 路由器，负责任务类型匹配到对应的Agent
class Router:
    """
    路由器负责任务类型与Agent的动态匹配与调度。
    支持基于AgentRegistry的统一注册与检索，兼容能力标签/类型调度与异常保护。
    """

    def __init__(self, registry: AgentRegistry = None, agent_control_cls=AgentControlWrapper, control_timeout=3):
        # 允许外部注入AgentRegistry，方便测试与系统集成
        self.registry = registry if registry else AgentRegistry()

        # 注册/初始化系统Agent
        self._init_agents()

        # Agent控制包装表，按能力类型动态分发
        self.route_table = {}
        self.agent_control_cls = agent_control_cls
        self.control_timeout = control_timeout

        self._init_route_table()

    def _init_agents(self):
        """
        可统一在此注册/初始化全部业务相关Agent，支持后续自动扩展和能力发现。
        """
        # 如需动态加载配置，可用配置文件+反射机制自动加载
        DC_Agent = DataCleaningAgent()
        DA_Agent = DataAnalysisAgent()
        R_Agent = ReportAgent()
        DC_Agent.register(self.registry)
        DA_Agent.register(self.registry)
        R_Agent.register(self.registry)

    def _init_route_table(self):
        """
        初始化路由表。此处采用能力标签/类型名绑定，实际可支持动态映射或能力优先级分派。
        """
        # 可按需扩展为支持能力标签或任务类型列表绑定
        for agent in self.registry.list_agents():
            for cap in agent.capabilities:
                # 如任务类型和能力标签一致，则直接绑定（也可用优先级筛选最佳Agent）
                if cap not in self.route_table:
                    self.route_table[cap] = self.agent_control_cls(agent, timeout=self.control_timeout)
        # 兼容直接类型名映射
        # self.route_table["clean_data"] = self.agent_control_cls(self.registry.get_agent("dataclean-001"), timeout=self.control_timeout)
        # self.route_table["analyze_data"] = self.agent_control_cls(self.registry.get_agent("dataclean-002"), timeout=self.control_timeout)
        # self.route_table["generate_report"] = self.agent_control_cls(self.registry.get_agent("dataclean-003"), timeout=self.control_timeout)

    def route(self, task):
        """
        路由与调度入口。根据任务类型/能力分派Agent，自动控制、监控与错误处理。
        【对应章节 9.2.1与章节 9.2.2 的入口】
        """
        task_type = task.get("type")
        task_context = TaskContext(task_type, task.get("payload"))
        agent_ctrl = self.route_table.get(task_type)

        if not agent_ctrl:
            # 若未找到完全匹配的类型，可考虑尝试能力标签分派、模糊匹配或返回异常
            print(f"[Router] 未找到匹配的Agent类型：{task_type}")
            task_context.status = "not_found"
            task_context.log("任务未被分派，缺少合适Agent")
            return task_context

        try:
            result_context = agent_ctrl.start(task_context)
            return result_context
        except Exception as e:
            task_context.status = "router_error"
            task_context.log(f"调度或执行异常: {e}")
            return task_context

    def summary(self):
        """
        打印已注册Agent与能力的映射表，便于系统调试与运维追踪。
        """
        print("当前路由映射表：")
        for task_type, agent_ctrl in self.route_table.items():
            print(f"任务类型/能力: {task_type} -> Agent: {agent_ctrl.agent.name} (ID: {agent_ctrl.agent.agent_id})")

    def route_dag(self, dag_def):
        """
        执行整个 DAG 的调度执行逻辑。
        每轮查找所有 ready 的 task 节点并调度执行，直到 DAG 执行完毕。
        """
        dag_ctx = DAGContext(dag_def)
        dag_ctx.visualize()
        while not dag_ctx.is_all_done():
            ready_tasks = dag_ctx.get_ready_tasks()
            if not ready_tasks:
                print("[Router] 无可调度任务，可能存在循环依赖或前序任务失败")
                break

            # 并发调度所有 ready 的任务
            futures = []
            with ThreadPoolExecutor(max_workers=len(ready_tasks)) as executor:
                for tid in ready_tasks:
                    ctx = dag_ctx.nodes[tid]
                    agent_ctrl = self.route_table.get(ctx.task_type)
                    if agent_ctrl:
                        # 异步提交每个Agent执行任务
                        futures.append(executor.submit(agent_ctrl.start, ctx))
                    else:
                        ctx.status = "not_found"
                        ctx.log("未找到匹配Agent")

                # 等待所有并发任务执行完成，并更新 DAG 状态
                for future in as_completed(futures):
                    result_ctx = future.result()
                    dag_ctx.nodes[result_ctx.task_id] = result_ctx

        return dag_ctx


# 主程序入口：执行任务调度、追踪状态，并保存执行日志
if __name__ == "__main__":
    """ 对应章节 9.2.1 与 章节 9.2.2 的用法示例 """
    # multiprocessing.freeze_support()  # Windows 平台必须加上这行
    #
    # # 模拟任务列表（实际应来自任务队列、LLM响应等）
    # tasks = [
    #     {"type": "clean_data", "payload": "数据A"},
    #     {"type": "analyze_data", "payload": "数据B"},
    #     {"type": "generate_report", "payload": "数据C"}
    # ]
    #
    # router = Router()
    # results = [router.route(task) for task in tasks]
    #
    # # 汇总每个任务的执行结果并导出CSV，便于分析与追踪
    # summary = pd.DataFrame([{
    #     "Task ID": r.task_id,
    #     "Type": r.task_type,
    #     "Status": r.status,
    #     "Retries": r.retries,
    #     "Start": r.start_time,
    #     "End": r.end_time,
    #     "Output": r.outputs
    # } for r in results if r])
    #
    # summary.to_csv("./summary.csv")

    """ 对应章节 9.2.3 的用法示例 """
    multiprocessing.freeze_support()  # Windows 平台必须加上这行

    # 将独立任务包装为“虚拟 DAG”（无依赖）
    dag_tasks = [
        {"id": "t1", "type": "clean_data", "payload": "数据A", "next": ["t3", "t2"]},
        {"id": "t2", "type": "analyze_data", "payload": "数据B", "next": []},
        {"id": "t3", "type": "generate_report", "payload": "数据C", "next": []}
    ]

    # 路由器初始化
    router = Router()

    # 执行 DAG 调度
    dag_ctx = router.route_dag(dag_tasks)

    # 汇总结果写入 CSV 文件
    summary = pd.DataFrame(dag_ctx.get_summary())
    summary.to_csv("./summary.csv", index=False)

