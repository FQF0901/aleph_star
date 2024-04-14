# 检查节点是否为根节点
is_root(node) = isnull(node.parent)

# 检查节点是否为叶节点
is_leaf(node) = length(node.children) == 0

# 检查节点的所有子节点是否都已探索过
all_children_explored(node::Node{ACTIONC}) where ACTIONC =
    length(node.children) == ACTIONC

# 检查节点的所有子节点是否都已完成
all_children_done(tree, node) =
    all_children_explored(node) && all(tree.dones[c.id] for c in values(node.children))

# 获取节点的层数，即从根节点到该节点的路径长度
function get_rank(node)
    rank = 1
    while true
        isnull(node.parent) && break    # 如果节点的父节点为空，退出循环
        node = get(node.parent)         # 获取节点的父节点
        rank += 1                       # 排名加一
    end
    rank
end

# 获取树中具有最佳奖励的叶节点
get_best_leaf(tree, gamma) = tree.nodes[argmax(tree.accumulated_rewards + gamma*[maximum(qs) for qs in tree.children_qs])]

# 获取树的排名，即最佳叶节点的排名
get_tree_rank(tree, gamma) = get_rank(get_best_leaf(tree, gamma))

# 获取树中节点的累积奖励的最大值
get_accumulated_reward(tree) = maximum(tree.accumulated_rewards)

# 计算树中每个节点被访问的次数
function calc_visitedc(tree, maxval=-1)
    # calculate number of times each node was visited
    # iterate in reverse, so parents have all children
    # already updated when we get to them
    # 计算每个节点被访问的次数
    # 逆序迭代，这样当我们到达父节点时，所有子节点都已经更新
    visitedc = zeros(Int32, length(tree.nodes))
    for nix in length(tree.nodes):-1:1
        node = tree.nodes[nix]
        for ch in values(node.children)
            visitedc[nix] += visitedc[ch.id]
            if maxval > 0 && visitedc[nix] > maxval
                visitedc[nix] = maxval
            end
        end
        visitedc[nix] += 1
    end
    visitedc
end

# 计算树中节点动作的访问次数
function calc_action_visitedc(tree, visited_threshold=-1, nonexplored_value=0)
    visitedc = calc_visitedc(tree, visited_threshold)
    avc = Vector{Int32}[]
    for node in tree.nodes
        _avc = Int32(nonexplored_value) * ones(Int32, length(tree.children_qs[1]))
        for (ix, ch) in node.children
            _avc[ix] = visitedc[ch.id]
        end
        push!(avc, _avc)
    end
    avc
end

# backprop_weighted_q!函数用于更新树中节点的Q值，采用加权平均的方式计算Q值，并通过反向传播的方式更新父节点的Q值。
function backprop_weighted_q!(tree, gamma, visited_threshold=-1)
    avisitedc = calc_action_visitedc(tree, visited_threshold)
    # calculate weighted Qs, root is done separately
    # iterae in reverse, so parents have all children
    # already updated when we get to them
    # 计算加权Q值，根节点单独处理
    # 逆序迭代，这样父节点在到达它们时已经更新完所有子节点
    for nix in length(tree.nodes):-1:2
        node = tree.nodes[nix]
        # 检查节点的所有子节点是否都已完成
        tree.dones[node.id] = all_children_done(tree, node)
        # update Q at parent
        # 更新父节点的Q值
        parent = get(node.parent)
        # 计算当前节点相对于其父节点的奖励：因为是action value，因此要基于父状态
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        mn = if is_leaf(node)
            mean(tree.children_qs[nix])
        else
            sum(avisitedc[nix] ./ sum(avisitedc[nix] .* tree.children_qs[nix]))
        end
        # 更新父节点的Q值
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma*mn
    end
    # update root
    # 更新根节点的完成状态
    tree.dones[1] = all_children_done(tree, tree.nodes[1])
end

# backprop_max_q!函数用于更新树中节点的Q值，采用最大Q值的方式更新父节点的Q值
function backprop_max_q!(tree, gamma)
    # backprop everybody in reverse, so parents have all children
    # already updated when we get to them
    # 逆序迭代，这样父节点在到达它们时已经更新完所有子节点
    for nix in length(tree.nodes):-1:2
        node = tree.nodes[nix]
        parent = get(node.parent)
        # 检查节点的所有子节点是否都已完成
        tree.dones[node.id] = all_children_done(tree, node)        
        # update Q at parent
        # 更新父节点的Q值
        mx::Float32 = maximum(tree.children_qs[node.id])
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        # 更新父节点的Q值
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma*mx
    end
    # update root
    # 更新根节点的完成状态
    tree.dones[1] = all_children_done(tree, tree.nodes[1])
end

# expand!函数用于扩展树中的节点，根据环境返回的奖励和状态，创建新的节点，并更新父节点的属性和Q值
function expand!(heap, tree, parent_node::Node{ACTIONC}, w, action_ix, env, gamma) where ACTIONC
    actual_action = action_ix_to_action(env, action_ix)
    new_sim_state, reward, done = sim!(env, tree.states[parent_node.id], actual_action)
    new_sensors = get_sensors(env, new_sim_state)
    children_qs = if done
        zeros(Float32, ACTIONC)
    else
        network_predict(env, w,new_sensors)
    end
    id::Int32 = length(tree.states) + 1
    new_node = Node(Int32(action_ix), Nullable(parent_node), id)
    parent_node.children[action_ix] = new_node
    accumulated_reward::Float32 = tree.accumulated_rewards[parent_node.id] + Float32(reward)
    push!(tree.nodes, new_node)
    push!(tree.sensors, new_sensors)
    push!(tree.children_qs, children_qs)
    push!(tree.states, new_sim_state)
    push!(tree.accumulated_rewards, accumulated_reward)
    push!(tree.dones, done)
    if !done
        for (aix, q) in enumerate(children_qs)
            score::Float32 = accumulated_reward + Float32(gamma)*q
            push!(heap, Int32(aix), new_node.id, score)
        end
    end
end

function build_tree(w, env, root_state, stepc, epsilon, gamma).
    # build the root
    # 构建树的根节点
    root_sensors = get_sensors(env, root_state) # 获取根节点的传感器数据
    root_children_qs = network_predict(env, w, root_sensors)    # 使用神经网络预测根节点的所有可能动作的Q值
    ACTIONC = length(root_children_qs)  # 动作数量
    null_parent = Nullable{Node{ACTIONC}}() # 根节点的父节点为空
    root_action_ix::Int32 = -1 # nonsensical action: 根节点的动作索引设为-1，表示无意义的动作
    root_id::Int32 = 1  # 根节点的ID为1
    root_accumulated_reward::Float32 = 0.0f0    # 根节点的累积奖励初始化为0
    root_done::Bool = false # 根节点的完成状态初始化为false
    root = Node(root_action_ix, null_parent, root_id)   # 创建根节点

    # build the tree
    # 构建树的结构
    nodes = Node{ACTIONC}[] # 存储树中的所有节点
    sensors = typeof(root_sensors)[]    # 存储树中每个节点的传感器数据
    children_qs = Vector{Float32}[] # 存储树中每个节点的子节点的Q值
    states = typeof(root_state)[]   # 存储树中每个节点的状态数据
    accumulated_rewards = Float32[] # 存储树中每个节点的累积奖励
    dones = Bool[]  # 存储树中每个节点的完成状态
    heap = Heap()   # 创建堆数据结构用于管理节点的选择顺序
    tree = Tree(env, heap, nodes, sensors, children_qs, states, accumulated_rewards, dones) # 创建树对象

    # update the tree with root
    # 将根节点添加到树中
    push!(tree.nodes, root)
    push!(tree.sensors, root_sensors)
    push!(tree.children_qs, root_children_qs)
    push!(tree.states, root_state)
    push!(tree.accumulated_rewards, root_accumulated_reward)
    push!(tree.dones, root_done)
    for (aix, q) in enumerate(tree.children_qs[1])
        score::Float32 = Float32(gamma)*q   # 计算根节点每个动作的得分，用于堆的初始化
        push!(tree.heap, Int32(aix), root.id, score)    # 将动作索引、父节点ID和得分添加到堆中
    end

    # add new nodes in a loop
    # 循环添加新节点
    for i in 1:stepc
        if i % 200 == 0
            print(".")  # 每200步打印一个点，用于显示进度
            flush(stdout)   # 刷新标准输出缓冲区
        end

        # choose parent and action
        # 选择父节点和动作
        tree.dones[1] && return tree # root is done: 如果根节点已完成，返回树对象
        length(heap) == 0 && return tree    # 如果堆为空，返回树对象
        action_ix, parent_id = if rand()>epsilon
            pop_max!(heap)  # 从堆中选择得分最高的节点和对应的动作
        else
            pop_rand!(heap) # 从堆中随机选择一个节点和对应的动作
        end
        @assert !tree.dones[parent_id]  # 断言所选父节点未完成
        parent = tree.nodes[parent_id]  # 获取所选父节点
        @assert !haskey(parent.children, action_ix) # 断言所选父节点没有对应动作的子节点

        # add a new node to the tree
        # 添加新节点到树中
        expand!(heap, tree, parent, w, action_ix, env, gamma)   # 扩展树，添加新节点
    end
    println()   # 打印换行，用于输出美化
    return tree # 返回构建好的树对象
end
