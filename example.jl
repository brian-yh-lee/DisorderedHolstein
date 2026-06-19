"""
This script `example.jl` demonstrates:
1. Running a small 2D disordered Holstein simulation with fixed parameters
2. Loading results from disk
3. Plotting optical conductivity and spectral functions
"""

include("DisorderedHolstein.jl")
using .DisorderedHolstein, Plots, Statistics, JLD2 

## 1. Basic simulation run
# System parameters
lx, ly, lz = 8, 8, 0           # 8×8 2D square lattice
t, tp = 1.0, -0.3              # n.n. and n.n.n. hopping 
K = 1.0                        # phonon spring constant
Δ = 0.0                        # onsite shift
ρf = 0.103                     # free electron DOS estimate
λ = 0.05                       # dimensionless electron-phonon coupling
T = 0.1                        # temperature 
α = 0.0                        # magnetic field
ν = 1.4                        # filling ν ∈ [0, 2]
γ_seed = 1                     # RNG seed for coupling disorder 

# Monte Carlo parameters
s_tot = 5000                   # total steps (including burn-in)
s_burn = 1000                  # burn-in steps
η = 0.1                        # Lorentzian broadening for Dirac distributions

# Run simulation
println("Running simulation with parameters:")
println("  Lattice: $(lx) × $(ly) (2D)")
println("  Hopping: t = $t, t' = $tp")
println("  Coupling: λ = $λ, ρf = $ρf")
println("  Temperature: T = $T (β = $(1/T))")
println("  Filling: ν = $ν")
println("")

simulate(lx, ly, lz, t, tp, K, Δ, ρf, λ, T, α, ν, s_tot, s_burn;
         θs = zeros(3),
         folder = ".",
         η = η,
         seed_name = false, γ_seed = γ_seed)


## 2. Loading results
# Construct filename (same as what simulate() saved)
filename = DisorderedHolstein.encode((lx, ly, lz, t, tp, K, Δ, ρf, λ, T, α, ν, s_tot, s_burn, γ_seed))
filepath = joinpath(".", filename)

# Load data
data = jldopen(filepath, "r")
pars = data["pars"]
diagnostics = data["diagnostics"]
observables = data["observables"]
close(data)

# Extract key results
accept_ratio = diagnostics.accept_ratio
n_mean = observables.n_mean
n_var = observables.n_var
Dx = observables.Dx
Dy = observables.Dy

println("Simulation diagnostics:")
println("  Acceptance ratio: $(round(accept_ratio, digits=3)) (target ≈ 0.574)")
println("  Final filling: ν = $(round(n_mean, digits=4)) (target: $ν)")

## 3. Plotting optical conductivity
omega_op = observables.omega_op
sigma_x = observables.op_cond_fast_x
sigma_y = observables.op_cond_fast_y
dc_x = observables.dc_cond_fast_x
dc_y = observables.dc_cond_fast_y
p_sigma = plot(omega_op, sigma_x, label="σ_x(ω)", linewidth=2, xlabel="ω", ylabel="σ(ω)", legend=:topright)
plot!(p_sigma, omega_op, sigma_y, label="σ_y(ω)", linewidth=2)
title!(p_sigma, "Optical Conductivity")
display(p_sigma)
