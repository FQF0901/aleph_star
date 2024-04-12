from typing import Dict, List, Optional, Tuple

# 定义一个类来表示树结构中的节点
class Node:
    def __init__(self, action_ix: int, parent: Optional['Node'], id: int):
        self.action_ix = action_ix  # 指向该状态的动作索引
        self.parent = parent  # 对父节点的引用，可为空以处理根节点
        self.children: Dict[int, Node] = {}  # 字典，存储按动作索引索引的子节点
        self.id = id  # 节点的唯一标识符

# 定义一个类来表示堆中的单元格
class HeapCell:
    def __init__(self, is_used: bool, score: float, action_ix: int, parent_id: int):
        self.is_used = is_used  # 指示单元格是否被使用
        self.score = score  # 与单元格相关联的分数
        self.action_ix = action_ix  # 动作的索引
        self.parent_id = parent_id  # 父节点的标识符

# 定义一个类来表示堆数据结构
class Heap:
    def __init__(self):
        self.cells: List[HeapCell] = []  # 用于存储堆单元格的列表
        self.total_used = 0  # 堆中已使用单元格的总数

# 定义一个类来表示树
class Tree:
    def __init__(self, env, heap):
        self.env = env  # 环境对象
        self.heap = heap  # 堆数据结构
        self.nodes: List[Node] = []  # 存储树中节点的列表
        self.sensors: List[SENSOR] = []  # 存储传感器数据的列表
        self.children_qs: List[List[float]] = []  # 存储子节点的 Q 值的列表
        self.states: List[STATE] = []  # 存储状态的列表
        self.accumulated_rewards: List[float] = []  # 存储累积奖励的列表
        self.dones: List[bool] = []  # 存储指示是否完成一集的标志的列表
