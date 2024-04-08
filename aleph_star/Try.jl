function traindtas(dtas::DemoTrainingAlephStar, iters::Integer, fname)
    w = map(KnetArray, dtas.w)
    for i in 1:iters
        #### Accumulating tree ###################################################
        
        println("--------------------------------------------------------------- i="*string(i))
        flush(stdout)
        
        state, env = initialize_simple_road();
        tree = build_tree(w, env, state, dtas.stepc, dtas.epsilon, dtas.gamma)
        backprop_weighted_q!(tree, dtas.gamma, dtas.weighted_nodes_threshold)
        
        # gather some stats
        push!(dtas.num_of_car_crash, 0)
        push!(dtas.num_of_wall_crash, 0)
        push!(dtas.num_of_vel_crash, 0)
        # some crash stats
        for node in tree.nodes
            !is_leaf(node) && continue
            sim_state = tree.states[node.id]
            yolo_x = env.splx(sim_state.yolocar_dist)
            yolo_y = env.sply(sim_state.yolocar_dist)    
            _, result = my_calc_reward(getcar(sim_state), env.coll, yolo_x, yolo_y, SAFETY)
            if result==1
                dtas.num_of_vel_crash[end] += 1
            elseif result==2
                dtas.num_of_wall_crash[end] += 1
            elseif result==3
                dtas.num_of_car_crash[end] += 1
            end
            @assert is_leaf(node)
            # this is possible due to floating point issues
            # result==0 && @assert !node.done
        end
        # calculate mean speed
        treev = Float64[]
        n = get_best_leaf(tree, dtas.gamma)
        times_passed = tree.states[n.id].times_passed_yolocar
        while !is_root(n)
            sim_state = tree.states[n.id]
            push!(treev, sim_state.speed)
            n = get(n.parent)
        end
        max_rank = get_tree_rank(tree, dtas.gamma)
        push!(dtas.ranks, max_rank)
        push!(dtas.tree_mean_speed, mean(treev))
        push!(dtas.tree_std_speed, std(treev))
        push!(dtas.rewards, get_accumulated_reward(tree))
        push!(dtas.num_of_done_leafs, sum([tree.dones[n.id] for n in tree.nodes if is_leaf(n)]))
        push!(dtas.num_of_leafs, sum([is_leaf(n) for n in tree.nodes]))
        # these could be slightly different due to floating point issues:
        # @assert num_of_car_crash[end] + num_of_vel_crash[end] + num_of_wall_crash[end] == num_of_done_leafs[end]
        
        ixs = findall([!is_leaf(n) || tree.dones[n.id] for n in tree.nodes])
        append!(dtas.qs, tree.children_qs[ixs])
        append!(dtas.sensors, tree.sensors[ixs])
        f = if length(dtas.priorities)==0
            1.0
        else 
            median(dtas.priorities)
        end
        append!(dtas.priorities, f.*ones(Float32, length(ixs)))
        if length(dtas.qs) > dtas.max_states
            dtas.qs = dtas.qs[(end-dtas.max_states+1):end]
            dtas.sensors = dtas.sensors[(end-dtas.max_states+1):end]
            dtas.priorities = dtas.priorities[(end-dtas.max_states+1):end]
        end
        
        #### Training #######################################################
                            
        if length(dtas.qs) >= dtas.train_when_min
            N = round(Int64, length(ixs) * dtas.train_epochs / dtas.batchsize)
            println("\n---- training for "*string(N)*" iterations"); flush(stdout)
            w,ll = trainit(N, w, dtas.sensors, dtas.qs, dtas.LR, dtas.batchsize, dtas.priorities)
            dtas.w = map(Array, w)
            if dtas.epsilon > dtas.MIN_EPSILON
                dtas.epsilon *= dtas.fac
            end
        end                  
    end
end