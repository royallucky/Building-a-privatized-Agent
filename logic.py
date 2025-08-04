import networkx as nx
import matplotlib.pyplot as plt

# 推荐用graphviz_layout
try:
    pos = nx.nx_agraph.graphviz_layout
except AttributeError:
    pos = nx.nx_pydot.graphviz_layout  # 部分环境需用pydot
except Exception:
    pos = None

G = nx.DiGraph()

# 添加节点
nodes = [
    "start", "dispatch_tasks", "t1", "t2", "t3",
    "refresh_pending_tasks", "END"
]
G.add_nodes_from(nodes)

# 添加边
edges = [
    ("start", "dispatch_tasks"),
    ("dispatch_tasks", "t1"),
    ("dispatch_tasks", "t2"),
    ("dispatch_tasks", "t3"),
    ("t1", "refresh_pending_tasks"),
    ("t2", "refresh_pending_tasks"),
    ("t3", "refresh_pending_tasks"),
    ("refresh_pending_tasks", "dispatch_tasks"),
    ("dispatch_tasks", "END")  # 结束条件
]
G.add_edges_from(edges)

# 层次化布局（需安装graphviz和pygraphviz/pydot）
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
except:
    pos = nx.spring_layout(G, seed=42)  # 兜底，不美观但能用

plt.figure(figsize=(10, 6))
nx.draw(
    G, pos,
    with_labels=True,
    node_color='#d0ebff',
    node_size=2500,
    font_size=12,
    font_weight='bold',
    edge_color='#9db4c0',
    arrows=True,
    arrowsize=20,
    linewidths=1.8
)
plt.title("DAG 工作流调度流程图（层次化美观版）")
plt.tight_layout()
plt.show()
