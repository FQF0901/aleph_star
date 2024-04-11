# 检查节点是否为根节点
def is_root(node):
    return node.parent is None

# 检查节点是否为叶节点
def is_leaf(node):
    return len(node.children) == 0

# 检查节点的所有子节点是否都已探索过
def all_children_explored(node, action_count):
    return len(node.children) == action_count

# 检查节点的所有子节点是否都已完成
def all_children_done(tree, node):
    return all_children_explored(node) and all(tree.dones[c.id] for c in node.children.values())

# 获取节点的层数，即从根节点到该节点的路径长度
def get_rank(node):
    rank = 1
    while True:
        if node.parent is None:
            break  # 如果节点的父节点为空，退出循环
        node = node.parent  # 获取节点的父节点
        rank += 1  # 排名加一
    return rank

# 获取树中具有最佳奖励的叶节点
def get_best_leaf(tree, gamma):
    accumulated_rewards_plus_gamma = tree.accumulated_rewards + gamma * [max(qs) for qs in tree.children_qs]
    best_leaf_index = max(range(len(tree.nodes)), key=lambda i: accumulated_rewards_plus_gamma[i])
    return tree.nodes[best_leaf_index]

# 获取树的排名，即最佳叶节点的排名
def get_tree_rank(tree, gamma):
    best_leaf = get_best_leaf(tree, gamma)
    return get_rank(best_leaf)

# 获取树中节点的累积奖励的最大值
def get_accumulated_reward(tree):
    return max(tree.accumulated_rewards)

# 计算树中每个节点被访问的次数
def calc_visitedc(tree, maxval=-1):
    # 计算每个节点被访问的次数
    # 逆序迭代，这样当我们到达父节点时，所有子节点都已经更新
    visitedc = [0] * len(tree.nodes)
    for nix in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[nix]
        for ch in node.children.values():
            visitedc[nix] += visitedc[ch.id]
            if maxval > 0 and visitedc[nix] > maxval:
                visitedc[nix] = maxval
        visitedc[nix] += 1
    return visitedc

# 计算树中节点动作的访问次数
def calc_action_visitedc(tree, visited_threshold=-1, nonexplored_value=0):
    visitedc = calc_visitedc(tree, visited_threshold)
    avc = []
    for node in tree.nodes:
        _avc = [nonexplored_value] * len(tree.children_qs[0])
        for ix, ch in node.children.items():
            _avc[ix] = visitedc[ch.id]
        avc.append(_avc)
    return avc

# backprop_weighted_q!函数用于更新树中节点的Q值，采用加权平均的方式计算Q值，并通过反向传播的方式更新父节点的Q值。
def backprop_weighted_q(tree, gamma, visited_threshold=-1):
    avisitedc = calc_action_visitedc(tree, visited_threshold)
    # calculate weighted Qs, root is done separately
    # iterae in reverse, so parents have all children
    # already updated when we get to them
    # 计算加权Q值，根节点单独处理
    # 逆序迭代，这样父节点在到达它们时已经更新完所有子节点
    for nix in reversed(range(len(tree.nodes))):
        node = tree.nodes[nix]
        # 检查节点的所有子节点是否都已完成
        tree.dones[node.id] = all_children_done(tree, node)
        # update Q at parent
        # 更新父节点的Q值
        parent = node.parent
        # 计算节点奖励
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        if is_leaf(node):
            mn = np.mean(tree.children_qs[nix])
        else:
            mn = np.sum(avisitedc[nix] * tree.children_qs[nix]) / np.sum(avisitedc[nix])
        # 更新父节点的Q值
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma * mn
    # update root
    # 更新根节点的完成状态
    tree.dones[0] = all_children_done(tree, tree.nodes[0])

# backprop_max_q!函数用于更新树中节点的Q值，采用最大Q值的方式更新父节点的Q值
def backprop_max_q(tree, gamma):
    # backprop everybody in reverse, so parents have all children
    # already updated when we get to them
    # 逆序迭代，这样父节点在到达它们时已经更新完所有子节点
    for nix in reversed(range(len(tree.nodes))):
        node = tree.nodes[nix]
        parent = node.parent
        # 检查节点的所有子节点是否都已完成
        tree.dones[node.id] = all_children_done(tree, node)        
        # update Q at parent
        # 更新父节点的Q值
        mx = np.max(tree.children_qs[node.id])
        node_reward = tree.accumulated_rewards[node.id] - tree.accumulated_rewards[parent.id]
        # 更新父节点的Q值
        tree.children_qs[parent.id][node.action_ix] = node_reward + gamma * mx
    # update root
    # 更新根节点的完成状态
    tree.dones[0] = all_children_done(tree, tree.nodes[0])

# expand!函数用于扩展树中的节点，根据环境返回的奖励和状态，创建新的节点，并更新父节点的属性和Q值
def expand(heap, tree, parent_node, w, action_ix, env, gamma):
    actual_action = action_ix_to_action(env, action_ix)
    new_sim_state, reward, done = sim(env, tree.states[parent_node.id], actual_action)
    new_sensors = get_sensors(env, new_sim_state)
    children_qs = np.zeros(len(tree.children_qs[0]), dtype=np.float32) if done else network_predict(env, w, new_sensors)
    id_val = len(tree.states) + 1
    new_node = Node(action_ix, parent_node, id_val)
    parent_node.children[action_ix] = new_node
    accumulated_reward = tree.accumulated_rewards[parent_node.id] + float(reward)
    tree.nodes.append(new_node)
    tree.sensors.append(new_sensors)
    tree.children_qs.append(children_qs)
    tree.states.append(new_sim_state)
    tree.accumulated_rewards.append(accumulated_reward)
    tree.dones.append(done)
    if not done:
        for aix, q in enumerate(children_qs):
            score = accumulated_reward + gamma * q
            heap.append((aix, new_node.id, score))

def build_tree(w, env, root_state, stepc, epsilon, gamma):
    # build the root
    root_sensors = get_sensors(env, root_state)
    root_children_qs = network_predict(env, w, root_sensors)
    ACTIONC = len(root_children_qs)
    null_parent = None
    root_action_ix = -1
    root_id = 1
    root_accumulated_reward = 0.0
    root_done = False
    root = Node(root_action_ix, null_parent, root_id)

    # build the tree
    nodes = []
    sensors = []
    children_qs = []
    states = []
    accumulated_rewards = []
    dones = []
    heap = []
    tree = Tree(env, heap, nodes, sensors, children_qs, states, accumulated_rewards, dones)

    # update the tree with root
    tree.nodes.append(root)
    tree.sensors.append(root_sensors)
    tree.children_qs.append(root_children_qs)
    tree.states.append(root_state)
    tree.accumulated_rewards.append(root_accumulated_reward)
    tree.dones.append(root_done)
    for aix, q in enumerate(tree.children_qs[0]):
        score = float(gamma) * q
        tree.heap.append((aix, root_id, score))

    # add new nodes in a loop
    for i in range(1, stepc + 1):
        if i % 200 == 0:
            print(".", end="")
            # Flush the output to show progress
            import sys
            sys.stdout.flush()

        if tree.dones[0]:
            return tree
        if len(tree.heap) == 0:
            return tree

        action_ix, parent_id = tree.heap.pop(0 if np.random.rand() > epsilon else np.random.randint(len(tree.heap)))
        assert not tree.dones[parent_id]
        parent = tree.nodes[parent_id]
        assert action_ix not in parent.children

        expand(tree.heap, tree, parent, w, action_ix, env, gamma)

    print()
    return tree