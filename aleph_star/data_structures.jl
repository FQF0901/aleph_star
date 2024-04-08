# a Node contains the minimum needed to hold the tree structure: 定义一个 Node 结构体来表示树结构中的节点
struct Node{ACTIONC}
    action_ix::Int32    # action index leading to this state: 指向该状态的动作索引
    parent::Nullable{Node{ACTIONC}} # 对父节点的引用，可为空以处理根节点
    children::Dict{Int32, Node{ACTIONC}}    # 字典，存储按动作索引索引的子节点
    id::Int32   # 节点的唯一标识符
    # Node 结构体的构造函数
    function Node(
                  action_ix::Int32,
                  parent::Nullable{Node{ACTIONC}},
                  id::Int32) where ACTIONC
        children = Dict{Int32, Node{ACTIONC}}()  # 初始化一个空字典来存储子节点
        return new{ACTIONC}(action_ix, parent, children, id)    # 创建并返回一个新的 Node 实例
    end
end

# 定义一个结构体来表示堆中的单元格
struct HeapCell
    is_used::Bool # because we cannot efficiently pop a random element from a heap: 指示单元格是否被使用
    score::Float32  # 与单元格相关联的分数
    action_ix::Int32    # 动作的索引
    parent_id::Int32    # 父节点的标识符
end

# 定义一个可变结构体来表示堆数据结构
mutable struct Heap
    cells::Vector{HeapCell} # 用于存储堆单元格的向量
    total_used::Int64   # 堆中已使用单元格的总数
    # Heap 结构体的构造函数
    Heap() = new(Vector{HeapCell}(), 0) # 初始化一个空的堆单元格向量和总使用单元格数为零
end

# The tree contains the nodes (describing the structure)
# and any additional data per node
# 定义一个结构体来表示树
# STATE: 状态类型, SENSOR: 传感器类型, ENV: 环境类型, ACTIONC: 动作类型
struct Tree{STATE, SENSOR, ENV, ACTIONC}
    env::ENV                                # 环境对象
    heap::Heap                              # 堆数据结构
    # these 6 vectors are indexed by node.id (SOA style)
    nodes::Vector{Node{ACTIONC}}            # 存储树中节点的向量
    sensors::Vector{SENSOR}                 # 存储传感器数据的向量
    children_qs::Vector{Vector{Float32}}    # 存储子节点的 Q 值的向量
    states::Vector{STATE}                   # 存储状态的向量
    accumulated_rewards::Vector{Float32}    # 存储累积奖励的向量
    dones::Vector{Bool}                     # 存储指示是否完成一集的标志的向量
end
