"""
Module for adding multi-GPU support to tree TDVP algorithms

This module enhances the treeTDVP functionality by distributing computations across multiple GPUs.
"""

"""
    parallelize_tree_sweep!(A::TreeNetwork, M::TreeNetwork) -> Vector{Int}

Analyze the tree structure and assign task IDs to nodes for parallel processing on multiple GPUs.
Returns a vector of task IDs corresponding to each node in the tree.
"""
function parallelize_tree_sweep!(A::TreeNetwork, M::TreeNetwork)
    if !@isdefined(MULTI_GPU) || !MULTI_GPU
        # If multi-GPU not enabled, assign all nodes to task ID 1
        return ones(Int, length(A))
    end
    
    # Create task assignments for each node
    task_ids = zeros(Int, length(A))
    
    # Use a breadth-first approach to assign tasks
    # This ensures that siblings are assigned to different GPUs when possible
    queue = [findheadnode(A)]
    current_task = 1
    
    while !isempty(queue)
        node = popfirst!(queue)
        task_ids[node] = current_task
        current_task = current_task % length(GPU_DEVICES) + 1
        
        # Add children to the queue
        append!(queue, A.tree[node].children)
    end
    
    return task_ids
end

"""
    multi_gpu_tdvp1sweep!(dt, A::TreeNetwork, M::TreeNetwork, task_ids::Vector{Int}, F=nothing; verbose=false, kwargs...)

Run the tdvp1sweep! algorithm with multi-GPU support. Each node in the tree is processed on its
assigned GPU according to the task_ids vector.
"""
function multi_gpu_tdvp1sweep!(dt, A::TreeNetwork, M::TreeNetwork, task_ids::Vector{Int}, F=nothing; verbose=false, kwargs...)
    hn = findheadnode(A)
    if hn != findheadnode(M)
        setheadnode!(M, hn)
    end
    
    # Initialize environments
    F = initenvs(A, M, F)
    F0 = fill!(similar(M[1], (1,1,1)), 1)
    
    # Check if we're actually using multiple GPUs
    if !@isdefined(MULTI_GPU) || !MULTI_GPU
        # Fall back to regular implementation but with task_id parameter
        for id in PostOrder(A.tree)
            A, F = tdvp1sweep_node!(dt, A, M, F, id, F0, task_id=task_ids[id], verbose=verbose, kwargs...)
        end
        return A, F
    end
    
    # Process tree with multi-GPU support
    # We use the same traversal order as regular TDVP, but execute on different GPUs
    for id in PostOrder(A.tree)
        task_id = task_ids[id]
        gpu_id = GPUUtils.get_gpu_for_task(task_id)
        
        if verbose
            println("Processing node $id on GPU $gpu_id (task ID: $task_id)")
        end
        
        # Execute this node's operations on its assigned GPU
        A, F = GPUUtils.with_device(() -> 
            tdvp1sweep_node!(dt, A, M, F, id, F0, task_id=task_id, verbose=verbose, kwargs...),
            gpu_id)
        
        # Synchronize after each node to ensure data consistency
        GPUUtils.sync_all_gpus()
    end
    
    return A, F
end

"""
    tdvp1sweep_node!(dt, A, M, F, id, F0; task_id, verbose) -> (A, F)

Process a single node in the TDVP algorithm. This is a helper for the multi-GPU implementation.
"""
function tdvp1sweep_node!(dt, A::TreeNetwork, M::TreeNetwork, F, id, F0; task_id=nothing, verbose=false, kwargs...)
    children = A.tree[id].children
    nc = length(children)
    
    if isempty(children)
        # Leaf node - simplest case
        AC = A[id]
        # Single site evolution
        if id == findheadnode(A)
            # Head node with no children
            AC, info = exponentiate(x->applyH1(x, M[id], F0, F0), -im*dt, AC; ishermitian=true, task_id=task_id)
        else
            # Regular leaf node
            par = A.tree[id].parent
            pdir = findbond(A.tree[par], id)
            AC, info = exponentiate(x->applyH1(x, M[id], F[par], F0), -im*dt, AC; ishermitian=true, task_id=task_id)
        end
        
        A[id] = AC
        return A, F
    end
    
    # Non-leaf node
    AC = A[id]
    
    # Forward half step
    if id == findheadnode(A)
        # Head node
        AC, info = exponentiate(x->applyH1(x, M[id], F0, F[children]...), -im*dt/2, AC; 
                               ishermitian=true, task_id=task_id)
    else
        # Regular node
        par = A.tree[id].parent
        AC, info = exponentiate(x->applyH1(x, M[id], F[par], F[children]...), -im*dt/2, AC; 
                               ishermitian=true, task_id=task_id)
    end
    
    # Process each child
    for (i, child) in enumerate(children)
        # Extract C
        AL, C = QR(AC, i+1)
        
        # Update environment
        otherchildren = filter(x->x!=child, children)
        if id == findheadnode(A)
            F[id] = updateleftenv(AL, M[id], i, F0, F[otherchildren]...)
        else
            par = A.tree[id].parent
            F[id] = updateleftenv(AL, M[id], i, F[par], F[otherchildren]...)
        end
        
        # Evolve C backwards
        F_child = F[child]
        F_id = F[id]
        C, info = exponentiate(x->applyH0(x, F_id, F_child), im*dt/2, C; 
                              ishermitian=true, task_id=task_id)
        
        # Contract C with child
        A[child] = contractC!(A[child], C, 1)
        
        # Process child subtree (happens in the main loop)
        
        # Extract C from child after it's been processed
        AR, C = QR(A[child], 1)
        A[child] = AR
        F[child] = updaterightenv(AR, M[child], F[A.tree[child].children]...)
        
        # Evolve C backwards
        C, info = exponentiate(x->applyH0(x, F[child], F[id]), im*dt/2, C; 
                              ishermitian=true, task_id=task_id)
        
        # Contract with node
        AC = contractC!(AL, C, i+1)
    end
    
    # Final half step
    if id == findheadnode(A)
        AC, info = exponentiate(x->applyH1(x, M[id], F0, F[children]...), -im*dt/2, AC; 
                               ishermitian=true, task_id=task_id)
    else
        par = A.tree[id].parent
        AC, info = exponentiate(x->applyH1(x, M[id], F[par], F[children]...), -im*dt/2, AC; 
                               ishermitian=true, task_id=task_id)
    end
    
    A[id] = AC
    return A, F
end