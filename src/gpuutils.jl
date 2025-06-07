module GPUUtils

using CUDA
using LinearAlgebra

# Check if NCCL is available
const nccl_available = try
    using NCCL
    true
catch
    false
end

export distribute_to_gpus, gather_from_gpus, sync_all_gpus
export get_gpu_for_task, with_device

"""
    get_gpu_for_task(task_id)

Select a GPU device for the given task using round-robin assignment.
"""
function get_gpu_for_task(task_id::Int)
    if !@isdefined(Main.MPSDynamics.MULTI_GPU) || !Main.MPSDynamics.MULTI_GPU
        return 0  # Default to first GPU if multi-GPU not enabled
    end
    
    devices = Main.MPSDynamics.GPU_DEVICES
    return devices[1 + (task_id - 1) % length(devices)]
end

"""
    with_device(f, device_id)

Execute function f on the specified GPU device, then return to the original device.
"""
function with_device(f::Function, device_id::Int)
    # Save the current device
    current_device = CUDA.device().handle
    
    try
        # Set the requested device
        CUDA.device!(device_id)
        
        # Execute the function
        return f()
    finally
        # Return to the original device
        CUDA.device!(current_device)
    end
end

"""
    distribute_to_gpus(data, task_ids)

Distribute data array across multiple GPUs based on task IDs.
Returns a dictionary mapping task ID to data portion on that GPU.
"""
function distribute_to_gpus(data::AbstractArray, task_ids::Vector{Int})
    if !@isdefined(Main.MPSDynamics.MULTI_GPU) || !Main.MPSDynamics.MULTI_GPU
        # Just return the data on the current device
        return Dict(task_ids[1] => CuArray(data))
    end
    
    # Count tasks per GPU
    task_to_gpu = Dict(task_id => get_gpu_for_task(task_id) for task_id in task_ids)
    gpu_task_counts = Dict{Int, Int}()
    for gpu_id in values(task_to_gpu)
        gpu_task_counts[gpu_id] = get(gpu_task_counts, gpu_id, 0) + 1
    end
    
    # Calculate how to split the data
    total_tasks = length(task_ids)
    result = Dict{Int, CuArray}()
    
    # Distribute data according to task assignment
    for task_id in task_ids
        gpu_id = task_to_gpu[task_id]
        
        # Execute on the target GPU
        result[task_id] = with_device(() -> begin
            # For simplicity, we're just copying the full data to each GPU
            # In a real implementation, you would split the data more intelligently
            return CuArray(data)
        end, gpu_id)
    end
    
    return result
end

"""
    gather_from_gpus(distributed_data, task_ids; reduce_op=+)

Gather data from multiple GPUs and combine using the specified reduction operation.
If NCCL is available, uses collective operations for efficiency; otherwise falls back
to a manual gather and reduce approach.
"""
function gather_from_gpus(distributed_data::Dict{Int, <:CuArray}, task_ids::Vector{Int}; 
                          reduce_op=+)
    if !@isdefined(Main.MPSDynamics.MULTI_GPU) || !Main.MPSDynamics.MULTI_GPU || length(task_ids) <= 1
        # If not multi-GPU or only one task, just return that data
        return distributed_data[task_ids[1]]
    end
    
    # Get the first array to determine size and type
    first_array = distributed_data[task_ids[1]]
    result = similar(first_array)
    
    # Check if NCCL is available for efficient collective operations
    if nccl_available && @isdefined(Main.MPSDynamics.NCCL_COMM)
        # Use NCCL to efficiently combine data across GPUs
        comm = Main.MPSDynamics.NCCL_COMM
        
        # This is a simplified version - in reality, you would need to 
        # handle different reduction operations and data types properly
        if reduce_op == +
            NCCL.allreduce!(first_array, result, NCCL.sum, comm)
        elseif reduce_op == *
            NCCL.allreduce!(first_array, result, NCCL.prod, comm)
        elseif reduce_op == max
            NCCL.allreduce!(first_array, result, NCCL.max, comm)
        elseif reduce_op == min
            NCCL.allreduce!(first_array, result, NCCL.min, comm)
        else
            error("Unsupported reduction operation")
        end
    else
        # Fallback implementation without NCCL
        # Copy first array to result
        CUDA.copyto!(result, first_array)
        
        # Manually gather and reduce from other GPUs
        for task_id in task_ids[2:end]
            other_array = distributed_data[task_id]
            # Move to CPU for reduction if they're on different devices
            if reduce_op == +
                result .+= other_array
            elseif reduce_op == *
                result .*= other_array
            elseif reduce_op == max
                result .= max.(result, other_array)
            elseif reduce_op == min
                result .= min.(result, other_array)
            else
                error("Unsupported reduction operation")
            end
        end
    end
    
    return result
end

"""
    sync_all_gpus()

Synchronize all GPUs to ensure pending operations are complete.
"""
function sync_all_gpus()
    if !@isdefined(Main.MPSDynamics.MULTI_GPU) || !Main.MPSDynamics.MULTI_GPU
        # Just synchronize the current device
        CUDA.synchronize()
        return
    end
    
    # Synchronize all devices
    for dev in Main.MPSDynamics.GPU_DEVICES
        CUDA.device!(dev)
        CUDA.synchronize()
    end
    
    # Return to the first device
    CUDA.device!(Main.MPSDynamics.GPU_DEVICES[1])
end

end # module