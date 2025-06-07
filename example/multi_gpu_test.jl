#!/usr/bin/env julia

# Example script for running a TDVP simulation with multi-GPU support
# Usage: julia multi_gpu_test.jl GPU MULTI_GPU
# This will use all available GPUs on the system

using MPSDynamics

# Parameters
dt = 0.1            # time step
tmax = 10.0         # total simulation time
nsites = 10         # number of sites in the system
localdim = 2        # local dimension (2 for spin-1/2)
J = 1.0             # coupling strength
h = 0.5             # magnetic field strength

# Create a spin chain Hamiltonian (Transverse Ising model)
sites = [siteinds("S=1/2", nsites)]
ampo = AutoMPO()
for j=1:nsites-1
    ampo += J, "Sx", j, "Sx", j+1
end
for j=1:nsites
    ampo += h, "Sz", j
end
H = MPO(ampo, sites)

# Convert to TreeNetwork representation
M = TreeNetwork(ITensor.(H))

# Create initial state (all spins up)
psi = MPS([ITensor([1.0, 0.0], s) for s in sites[1]])
A = TreeNetwork(psi)

# Print GPU information
if MPSDynamics.GPU
    println("Running with GPU acceleration")
    if MPSDynamics.MULTI_GPU
        println("Multi-GPU mode enabled")
        println("Available GPUs: $(length(MPSDynamics.GPU_DEVICES))")
        for (i, dev_id) in enumerate(MPSDynamics.GPU_DEVICES)
            MPSDynamics.CUDA.device!(dev_id)
            println("  GPU $i: $(MPSDynamics.CUDA.name())")
        end
    else
        println("Using single GPU: $(MPSDynamics.CUDA.name())")
    end
else
    println("Running on CPU (no GPU acceleration)")
end

# Define observables to measure
obs = [
    ("Sz_1", OpSum([("Sz", [1])]), 1),
    ("Sz_middle", OpSum([("Sz", [div(nsites, 2)])]), 1),
    ("Sz_N", OpSum([("Sz", [nsites])]), 1),
    ("TotalSz", OpSum([("Sz", [j]) for j in 1:nsites]), 1),
]

# Run simulation
if MPSDynamics.MULTI_GPU
    println("\nRunning multi-GPU TDVP simulation...")
    t, A_final, results = run_MGPUTDVP(dt, tmax, A, M; obs=obs, verbose=true)
else
    println("\nRunning standard TDVP simulation...")
    t, A_final, results = run_1TDVP(dt, tmax, A, M; obs=obs, verbose=true)
end

# Print results
println("\nSimulation complete!")
println("Final time: $(t[end])")
println("Observable values at final time:")
for (label, _) in obs
    println("  $label: $(results[label][end])")
end

# Compute and print runtime performance
println("\nPerformance summary:")
if MPSDynamics.MULTI_GPU
    println("Multi-GPU TDVP with $(length(MPSDynamics.GPU_DEVICES)) GPUs")
else
    if MPSDynamics.GPU
        println("Single-GPU TDVP")
    else
        println("CPU TDVP")
    end
end