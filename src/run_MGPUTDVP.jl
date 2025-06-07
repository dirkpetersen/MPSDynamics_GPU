"""
    run_MGPUTDVP(dt, tmax, A0, M, file=nothing; kwargs...)

Simulate time evolution using TDVP1 algorithm with multi-GPU acceleration.

# Arguments
- `dt`: Time step
- `tmax`: Maximum simulation time
- `A0`: Initial state (TreeNetwork)
- `M`: MPO Hamiltonian (TreeNetwork)
- `file`: Optional filename to save results

# Keywords
- `obs=[]`: List of observables to measure
- `verbose=false`: Print progress information
- `kwargs...`: Additional arguments passed to tdvp1sweep!

# Returns
- `t`: Time points
- `A`: Final state
- `res`: Measurement results for observables
"""
function run_MGPUTDVP(dt, tmax, A0, M, file=nothing; 
                    obs=[], verbose=false, remake=false, kwargs...)
    
    if !@isdefined(MULTI_GPU) || !MULTI_GPU
        @warn "Multi-GPU support not enabled. Run with GPU MULTI_GPU arguments for acceleration."
        return run_1TDVP(dt, tmax, A0, M, file; obs=obs, verbose=verbose, remake=remake, kwargs...)
    end
    
    # Setup initial state
    A = deepcopy(A0)
    nsteps = round(Int, abs(tmax / dt))
    t = [n*dt for n in 0:nsteps]
    
    if file !== nothing && isfile(file) && !remake
        println("Loading from $file")
        d = load(file)
        A = d["A"]
        res = d["res"]
        t0 = d["t"][end]
        i0 = findfirst(x->x==t0, t) + 1
    else
        # Initialize results storage
        res = initres(t, obs, A)
        i0 = 2
        storeres!(res, 1, A, obs)
    end
    
    # Initialize TDVP
    hn = findheadnode(A)
    setheadnode!(M, hn)
    mpsmixednorm!(A, hn)
    F = nothing
    
    # Analyze tree and assign parallel tasks
    task_ids = parallelize_tree_sweep!(A, M)
    if verbose
        println("Task assignments:")
        for (node, task) in enumerate(task_ids)
            gpu_id = GPUUtils.get_gpu_for_task(task)
            println("  Node $node -> Task $task (GPU $gpu_id)")
        end
    end
    
    for i in i0:nsteps+1
        ti = t[i]
        
        # Print progress information
        if verbose || (i-1) % 10 == 0
            @printf("t = %0.1f / %0.1f\n", ti, tmax)
        end
        
        # Forward time step with multi-GPU parallelization
        A, F = multi_gpu_tdvp1sweep!(dt, A, M, task_ids, F; verbose=verbose, kwargs...)
        
        # Store results
        storeres!(res, i, A, obs)
        
        # Save intermediate results if file is specified
        if file !== nothing
            save(file, Dict([("A", A), ("t", t[1:i]), ("res", res)]))
        end
    end
    
    return t, A, res
end

export run_MGPUTDVP