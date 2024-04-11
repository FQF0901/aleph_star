class DemoTrainingAlephStar:
    def __init__(self):
        # network weights
        self.w = []

        # tree params
        self.stepc = 0
        self.epsilon = 0
        self.MIN_EPSILON = 0
        self.fac = 0
        self.gamma = 0

        # training params
        self.LR = 0
        self.batchsize = 0
        self.train_epochs = 0
        self.max_states = 0
        self.train_when_min = 0

        # for validation
        self.val_safety = 0

        # experience buffer
        self.sensors = []
        self.qs = []
        self.priorities = []

        # for stats
        self.rewards = []
        self.ranks = []
        self.num_of_done_leafs = []
        self.num_of_leafs = []
        self.num_of_car_crash = []
        self.num_of_wall_crash = []
        self.num_of_vel_crash = []
        self.tree_mean_speed = []
        self.tree_std_speed = []
        self.val_mean_speed = []
        self.val_std_speed = []
        self.val_steps = []
        self.val_rewards = []
        self.avg_window = 0
        self.weighted_nodes_threshold = 0

    @classmethod
    def initialize(cls):
        dtas = cls()
        dtas.w = cls.initialize_weights(actionc_steer * actionc_accel)
        dtas.stepc = 5500
        dtas.epsilon = 0.5
        dtas.MIN_EPSILON = 0.01
        dtas.fac = 0.995
        dtas.gamma = 0.98
        dtas.LR = 0.01
        dtas.batchsize = 64
        dtas.train_epochs = 20
        dtas.max_states = 80000
        dtas.train_when_min = 50000
        dtas.val_safety = 1.0
        dtas.avg_window = 50
        dtas.weighted_nodes_threshold = 200
        return dtas

def traindtas(dtas, iters, fname):
    # 将模型参数转换为KnetArray类型
    w = list(map(KnetArray, dtas.w))
    # 迭代训练
    for i in range(1, iters+1):
        #### Accumulating tree ###################################################
        # 初始化环境和状态
        
        print("--------------------------------------------------------------- i=" + str(i))
        sys.stdout.flush()
        
        state, env = initialize_simple_road()
        # 构建搜索树
        tree = build_tree(w, env, state, dtas.stepc, dtas.epsilon, dtas.gamma)
        # 反向传播更新Q值
        backprop_weighted_q(tree, dtas.gamma, dtas.weighted_nodes_threshold)
        
        # Gather some stats
        # 收集一些统计信息
        dtas.num_of_car_crash.append(0)
        dtas.num_of_wall_crash.append(0)
        dtas.num_of_vel_crash.append(0)
        # Some crash stats
        # 统计碰撞次数
        for node in tree.nodes:
            if not is_leaf(node):
                continue
            sim_state = tree.states[node.id]
            yolo_x = env.splx(sim_state.yolocar_dist)
            yolo_y = env.sply(sim_state.yolocar_dist)    
            _, result = my_calc_reward(getcar(sim_state), env.coll, yolo_x, yolo_y, SAFETY)
            if result == 1:
                dtas.num_of_vel_crash[-1] += 1
            elif result == 2:
                dtas.num_of_wall_crash[-1] += 1
            elif result == 3:
                dtas.num_of_car_crash[-1] += 1
            assert is_leaf(node)
            # This is possible due to floating point issues
            # assert result == 0 and not node.done
        # Calculate mean speed
        # 计算平均速度
        treev = []
        n = get_best_leaf(tree, dtas.gamma)
        times_passed = tree.states[n.id].times_passed_yolocar
        while not is_root(n):
            sim_state = tree.states[n.id]
            treev.append(sim_state.speed)
            n = get(n.parent)
        max_rank = get_tree_rank(tree, dtas.gamma)
        dtas.ranks.append(max_rank)
        dtas.tree_mean_speed.append(mean(treev))
        dtas.tree_std_speed.append(std(treev))
        dtas.rewards.append(get_accumulated_reward(tree))
        dtas.num_of_done_leafs.append(sum([tree.dones[n.id] for n in tree.nodes if is_leaf(n)]))
        dtas.num_of_leafs.append(sum([is_leaf(n) for n in tree.nodes]))
        # These could be slightly different due to floating point issues:
        # assert num_of_car_crash[-1] + num_of_vel_crash[-1] + num_of_wall_crash[-1] == num_of_done_leafs[-1]
        
        ixs = [i for i, n in enumerate(tree.nodes) if not is_leaf(n) or tree.dones[n.id]]
        dtas.qs.extend(tree.children_qs[ixs])
        dtas.sensors.extend(tree.sensors[ixs])
        # 计算优先级
        f = 1.0 if len(dtas.priorities) == 0 else median(dtas.priorities)
        dtas.priorities.extend([f * np.ones(len(ixs), dtype=np.float32)])
        # If the length of priorities exceeds the maximum number of states, truncate
        if len(dtas.qs) > dtas.max_states:
            dtas.qs = dtas.qs[-dtas.max_states:]
            dtas.sensors = dtas.sensors[-dtas.max_states:]
            dtas.priorities = dtas.priorities[-dtas.max_states:]

        # Training #######################################################
        # 训练部分
        if len(dtas.qs) >= dtas.train_when_min:
            N = round(len(ixs) * dtas.train_epochs / dtas.batchsize)
            print("\n---- training for " + str(N) + " iterations")
            sys.stdout.flush()
            w, ll = trainit(N, w, dtas.sensors, dtas.qs, dtas.LR, dtas.batchsize, dtas.priorities)
            dtas.w = list(map(list, w))
            # 更新epsilon
            if dtas.epsilon > dtas.MIN_EPSILON:
                dtas.epsilon *= dtas.fac

        # Validating #####################################################
        valv = []
        valr = 0.0
        spd = np.mean(dtas.tree_mean_speed[-dtas.avg_window:])
        if np.isnan(spd):
            spd = 0.8
        state, val_env = initialize_simple_road(spd)
        for _ in range(dtas.stepc):
            valv.append(state.speed)
            _sensors = get_sensors(val_env, state)
            _vqs = list(network_predict(val_env, w, _sensors))
            action = action_ix_to_action(val_env, np.argmax(_vqs))
            state, reward, done = sim(val_env, state, action, 5, dtas.val_safety)
            valr += reward
            if done:
                break
        vpassed = state.times_passed_yolocar
        dtas.val_rewards.append(valr)
        dtas.val_mean_speed.append(np.mean(valv))
        dtas.val_std_speed.append(np.std(valv))
        dtas.val_steps.append(len(valv))

        # Reporting #####################################################
        print()
        print("      eps   = " + str(round(dtas.epsilon, 4)) + "    gamma=" + str(round(dtas.gamma, 4)))
        print(" tree_rnk   = " + str(max_rank) + "    mvel    = " + str(round(np.mean(treev), 4)) + "    svel = " + str(round(np.std(treev), 4)))
        print("val_steps   = " + str(len(valv)) + "    mvel    = " + str(round(np.mean(valv), 4)) + "    svel = " + str(round(np.std(valv), 4)))
        print("tree_passed = " + str(times_passed) + "    vpassed = " + str(vpassed))
        sys.stdout.flush()

        if i % 10 == 0:
            IJulia.clear_output()

        if i % 100 == 0:
            save(fname, dtas)

def plotdtas(dtas):
    # Plot max tree reward vs. network rewards
    p1 = plt.plot(moving_avg(dtas.rewards, dtas.avg_window), label="max tree reward")
    plt.plot(moving_avg(dtas.val_rewards, dtas.avg_window), lw=3, label="network rewards")

    # Plot max tree rank (percentage) vs. network rank, done percentage, and leaf percentage
    p2 = plt.plot(moving_avg(dtas.ranks / dtas.stepc, dtas.avg_window), label="max tree rank (perc.)")
    plt.plot(moving_avg(dtas.val_steps / dtas.stepc, dtas.avg_window), lw=3, label="network rank")
    plt.plot(moving_avg(dtas.num_of_done_leafs / dtas.stepc, dtas.avg_window), label="done perc.")
    plt.plot(moving_avg(dtas.num_of_leafs / dtas.stepc, dtas.avg_window), label="leafs perc.")

    # Plot velocity crash percentage for velocity, wall, and car crashes
    p3 = plt.plot(moving_avg(dtas.num_of_vel_crash / dtas.num_of_done_leafs, dtas.avg_window), label="vel crash perc.", lw=3)
    plt.plot(moving_avg(dtas.num_of_wall_crash / dtas.num_of_done_leafs, dtas.avg_window), label="wall crash perc.", lw=3)
    plt.plot(moving_avg(dtas.num_of_car_crash / dtas.num_of_done_leafs, dtas.avg_window), label="car crash perc.", lw=3)

    # Plot tree mean speed, tree std speed, network std speed, and network mean speed
    p4 = plt.plot(moving_avg(dtas.tree_mean_speed, dtas.avg_window), label="tree mean speed")
    plt.plot(moving_avg(dtas.tree_std_speed, dtas.avg_window), label="tree std speed")
    plt.plot(moving_avg(dtas.val_std_speed, dtas.avg_window), label="network std speed")
    plt.plot(moving_avg(dtas.val_mean_speed, dtas.avg_window), label="network mean speed")

    # Combine all plots into one figure with a layout of 4 rows and 1 column
    plt.subplots_adjust(hspace=0.5)
    plt.show()