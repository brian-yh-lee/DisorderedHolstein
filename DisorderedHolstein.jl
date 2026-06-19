"""
Metropolis-adjusted Langevin algorithm + exact diagonalization solver for disordered Holstein polaron physics on 2D/3D lattices.
Requires: Base.Threads, JLD2, Random, LinearAlgebra, Statistics, Roots, Distributions, SparseArrays, DSP.

The basic Hamiltonian is H = -∑tᵢⱼcᵢ⁺cⱼ + ∑ᵢγᵢ(nᵢ-ν)xᵢ + ∑ᵢΔᵢnᵢ with periodic boundary conditions where cᵢ⁺/cᵢ are electron operators, 
nᵢ = cᵢ⁺cᵢ is the occupation, xᵢ are classical phonon fields, and γᵢ are random site-dependent electron-phonon couplings drawn from 
Uniform(-√(3w), √(3w)) with w = λ/ρf.

The important parameters of the code are detailed below. An example using this module is provided in `example.jl`. 

Lattice
- `lx::Int`: Linear system size in x-direction. Must be > 0.
- `ly::Int`: Linear system size in y-direction. Must be > 0.
- `lz::Int`: Linear system size in z-direction. Set to 0 for 2D lattices.

Tight-binding
- `t::Float64`: Nearest-neighbor hopping amplitude.
- `tp::Float64`: Next-nearest-neighbor hopping amplitude.

Phonon
- `K::Float64`: Phonon spring constant.
- `Δ::Float64`: Onsite energy shift.

Electron-phonon coupling
- `γ::Vector{Float64}`: Site-resolved electron-phonon coupling strengths; will be drawn from disorder distribution with RMS value γ_rms = √(λ/ρf).
- `ρf::Float64`: Free-electron density of states per spin; defined with factor of 1/(lx * ly * lz).
   For a 2D square lattice: ρf ≈ 2/πt (this is approximate; the exact value depends on the band structure).
- `λ::Float64`: Dimensionless electron-phonon coupling strength λ = ρf * γ² / K.

Thermodynamics
- `T::Float64`: Temperature.
- `β::Float64`: Inverse temperature (β = 1/T in units where Boltzmann constant kB = 1).
- `ν::Float64`: Target electron filling fraction. Must be in [0, 2] for spin-1/2 electrons. The algorithm adjusts μ to maintain this filling on average.

Magnetic Field
- `α::Float64`: Magnetic flux per plaquette in units of the flux quantum Φ₀ = h/e. Set to 0 for zero field.

Monte Carlo Parameters
- `τ::Float64`: Step size for discretized Langevin dynamics used to propose new phonon configurations in the Monte Carlo. 
- `s_tot::Int`: Total number of MALA steps (burn-in + production). 
- `s_burn::Int`: Number of burn-in steps discarded before measurements. The algorithm tunes step size τ and chemical potential μ during burn-in.

Keyword Arguments
- `θs::Vector{Float64}`: Twisted boundary condition phases. 
- `x0::Vector{Float64}`: Initial phonon configuration. 
- `η::Float64`: Broadening parameter for Dirac distributions.
"""


module DisorderedHolstein

export 
    tb_ham, 
    vol,
    simulate, 
    malaed, 
    encode, 
    fermi, 
    dfde, 
    N_site, 
    N_tot, 
    A_ind,
    find_nodal_antinodal,
    gaussian, 
    lorentzian, 
    op_cond_fast,
    op_cond_weight, 
    J_site, 
    tau_lattice,
    tau,
    E_basis,
    γ2λ, 
    λ2γ,
    broaden_spectrum,
    bloch_2d

using Base.Threads, JLD2, Random, LinearAlgebra, Statistics, Roots, Distributions, SparseArrays, DSP

"""

Code for timing expensive functions in full simulation 

"""

const TIMING = false     # set this to ``false" and comment out ``using TimerOutputs" to remove dependency on TimerOutputs
# using TimerOutputs
if TIMING
    using TimerOutputs
    const to = TimerOutput()
else
    macro timeit(args...)
        esc(args[end])
    end
    const to = nothing
end

""" 

Helper functions 

"""

# Turn parameters into a filename 
function encode(parameters)
    p = join(parameters, "_")
    return "malaed__$(p)"
end

# Returns total number of sites, accounting for 1D, 2D, or 3D.
function vol(lx, ly, lz)
    return Int(max(lx, 1) * max(ly, 1) * max(lz, 1))
end

function gaussian(x, μ, σ; w = 1.0)
    C = 1 / sqrt(2π)
    z = (x - μ) / σ
    return w * C * exp(-0.5 * z^2) / σ
end

function lorentzian(x, η)
    # Normalized Lorentzian with broadening η
    return (1 / π) .* η ./ (x.^2 .+ η.^2)
end

# Numerically stable Fermi function. Takes in both energy AND chemical potential
function fermi(λ, β, μ)
    z = β .* (λ .- μ)
    return (1/2) .* (1 .- tanh.(z ./ 2))
end

# Numerically stable derivative of Fermi function w.r.t. energy: ∂f(ϵ, β, μ)/∂ϵ
function dfde(λ, β, μ)
    z = β .* (λ .- μ)
    a = -(β / 4) .* sech.(z ./ 2) .^ 2
    return a
end

# Numerically stable log(1 + exp(x))
function log1pexp(x)
    if x > 0
        return x + log1p(exp(-x))
    else
        return log1p(exp(x))
    end
end

# Convert physical coupling γ to dimensionless coupling λ in units of K = 1 
function γ2λ(γ, ρf) 
    return  ρf * γ^2 
end

# Convert dimensionless coupling λ to physical coupling γ in units K = 1 
function λ2γ(λ, ρf)
    return sqrt(λ / ρf)
end

# Approximately finds k-point closest to the nodal and antinodal points. 
function find_nodal_antinodal(lx, ly, t, tp, μ0; e_tol = 0.3)
    ϵ(kx, ky) = -2t * (cos(kx) + cos(ky)) - 4tp * cos(kx) * cos(ky)

    kxs = [2π * (nx <= lx÷2 ? nx : nx - lx) / lx for nx in 0:lx-1]
    kys = [2π * (ny <= ly÷2 ? ny : ny - ly) / ly for ny in 0:ly-1]

    # Nodal: kx = ky > 0, closest to Fermi surface
    nodal_k = nothing
    nodal_dist = Inf
    for nx in 0:lx-1
        kx = 2π * (nx <= lx÷2 ? nx : nx - lx) / lx
        ky = 2π * (nx <= ly÷2 ? nx : nx - ly) / ly
        if kx <= 0
            continue
        end
        d = abs(ϵ(kx, ky) - μ0)
        if d < nodal_dist
            nodal_dist = d
            nodal_k = (kx, ky)
        end
    end

    # Antinodal: upper right quadrant, within e_tol of Fermi surface, closest to (π, 0)
    antinodal_k = nothing
    antinodal_dist = Inf
    for kx in kxs, ky in kys
        if kx <= 0 || ky < 0
            continue
        end
        if abs(ϵ(kx, ky) - μ0) > e_tol
            continue
        end
        d = sqrt((kx - π)^2 + ky^2)
        if d < antinodal_dist
            antinodal_dist = d
            antinodal_k = (kx, ky)
        end
    end

    if isnothing(nodal_k)
        # Find discrete point closest to (π/2, π/2)
        min_d = Inf
        for kx in kxs, ky in kys
            d = sqrt((kx - π/2)^2 + (ky - π/2)^2)
            if d < min_d
                min_d = d
                nodal_k = (kx, ky)
            end
        end
    end

    if isnothing(antinodal_k)
        # Find discrete point closest to (π, 0)
        min_d = Inf
        for kx in kxs, ky in kys
            d = sqrt((kx - π)^2 + ky^2)
            if d < min_d
                min_d = d
                antinodal_k = (kx, ky)
            end
        end
    end

    return nodal_k, antinodal_k
end

# Broadens functions of energy using a gaussian kernel
function broaden_spectrum(E_centers, spectrum, σ)       # used to smoothen density of states and localization length 
    smoothed = zeros(eltype(spectrum), length(spectrum))
    for i in eachindex(E_centers)
        # Gaussian kernel centered at E_centers[i]
        kernel = @. exp(-0.5 * ((E_centers - E_centers[i]) / σ)^2)
        
        # Normalize the kernel to preserve the integral (total number of states)
        kernel ./= sum(kernel)
        
        # Convolve
        smoothed[i] = sum(spectrum .* kernel)
    end
    return smoothed
end

"""

The Hamiltonian

"""

# Tight-binding Hamiltonian with nearest neighbor hopping t, next nearest neighbor hopping tp 
# on an lx * ly * lz lattice with twist boundary conditions θs and magnetic flux per plaquette α.
function tb_ham(t, tp, lx, ly, lz; θs = fill(0.0, 3), α = 0.0)
    # α = B⋅a²: magnetic flux per plaquette in units of flux quantum Φ₀ = h/e
    # Landau gauge: A = (0, Bx, 0)
    # PBC constraint: α * lx * ly must be an integer
    if !iszero(α) && abs(α * lx * ly - round(α * lx * ly)) > 1e-10
        error("Landau gauge + PBC requires α * lx * ly is an integer.")
    end

    dims = lz == 0 ? (lx, ly) : (lx, ly, lz)
    dx_range = -1:1
    dy_range = -1:1
    dz_range = lz == 0 ? (0:0) : (-1:1)
    v = prod(dims)
    ls = [lx, ly, lz == 0 ? 1 : lz]
    ϕs = θs ./ ls
    H = zeros(ComplexF64, v, v)

    for i in 1:v
        coords = Tuple(CartesianIndices(dims)[i])
        x_i = coords[1] - 1      # 0-indexed x-coordinate for Peierls phase

        for dx in dx_range, dy in dy_range, dz in dz_range
            dist_sq = dx^2 + dy^2 + dz^2
            if dist_sq == 0 || dist_sq > 2
                continue
            end
            amp = 0.0
            if dist_sq == 1
                amp = -t
            elseif dist_sq == 2
                amp = -tp
            end

            cx_n = mod1(coords[1] + dx, lx)
            cy_n = mod1(coords[2] + dy, ly)
            cz_n = lz == 0 ? 1 : mod1(coords[3] + dz, lz)
            if lz == 0
                nn = LinearIndices((lx, ly))[cx_n, cy_n]
            else
                nn = LinearIndices((lx, ly, lz))[cx_n, cy_n, cz_n]
            end

            if nn < i
                phase_val = 0.0

                # Twisted boundary condition phases (unchanged)
                phase_val += dx * ϕs[1]
                phase_val += dy * ϕs[2]
                if lz > 0
                    phase_val += dz * ϕs[3]
                end

                # Peierls phase in Landau gauge A = (0, Bx, 0)
                # For hop i → nn with displacement (dx, dy):
                #   φ = 2π α dy (x_i + dx/2)
                # Works for both NN (dx or dy = 0) and NNN (|dx|=|dy|=1)
                phase_val += 2π * α * dy * (x_i + dx / 2.0)

                H[i, nn] += amp * exp(-im * phase_val)
            end
        end
    end
    return H + H'
end

""" 

Optical conductivity calculation 

"""

function J_site(lx, ly, lz, dir, t, tp; θs, α)
    v = vol(lx, ly, lz)
    H = tb_ham(t, tp, lx, ly, lz; θs = θs, α = α)   # hopping matrix with phases

    # x, y, z component of site i
    xs = [((i-1) % lx) + 1 for i in 1:v]
    ys = [((i-1) ÷ lx) % ly + 1 for i in 1:v]
    zs = [((i-1) ÷ (lx*ly)) + 1 for i in 1:v]

    # pick the coordinate difference depending on direction
    if dir == [1,0,0]
        coord = xs
        l = lx
    elseif dir == [0,1,0]
        coord = ys
        l = ly
    elseif dir == [0,0,1]
        coord = zs
        l = lz
    else
        error("dir must be [1,0,0], [0,1,0], or [0,0,1]")
    end

    R = coord .- coord'                 # build matrix Rᵢⱼ = (rᵢ - rⱼ)
    R .= R .- l .* round.(R ./ l)       # takes care of periodic boundary conditions 

    return -im * H .* R 
end

# Change basis of any operator from site basis to energy 
function E_basis(ϕ, A)
    return ϕ' * (A * ϕ)
end

function op_cond_weight(lx, ly, lz, λ, β, μ)
    v = vol(lx, ly, lz)
    f = fermi.(λ, Ref(β), Ref(μ))
    df = dfde.(λ, Ref(β), Ref(μ))
    Δf = f .- f'
    Δλ = λ .- λ'
    for i in 1:v
        for j in 1:v 
            # handling degeneracies: the diagonal contributions will vanish anyway since |Jₐₐ| = 0
            if abs(Δλ[i, j]) < 1e-10
                Δλ[i, j] = 1.0
                Δf[i, j] = df[i]     
            end 
        end
    end
    F = Δf ./ Δλ
    return F
end

function op_cond_fast(ω, λ, ϕ, β, μ, lx, ly, lz, Jsite, Je_buf, JΦ_buf, Jsq_buf; η)
    v = vol(lx, ly, lz)
    C = -2π / v
    mul!(JΦ_buf, Jsite, ϕ)
    mul!(Je_buf, ϕ', JΦ_buf)
    @. Jsq_buf = abs2(Je_buf)
    f = fermi.(λ, β, μ)
    df = dfde.(λ, β, μ)
    
    ω_step = step(ω)
    ω_lo = first(ω)
    n_ω = length(ω)
    
    λ_min, λ_max = extrema(λ)
    Δλ_max_abs = λ_max - λ_min
    
    k_min = floor(Int, (-Δλ_max_abs - ω_lo) / ω_step)
    k_max = ceil(Int, (Δλ_max_abs - ω_lo) / ω_step)
    n_hist = k_max - k_min + 1
    σ_hist = zeros(n_hist)
    
    @inbounds for a in 1:v, b in 1:v
        Δλ = λ[a] - λ[b]
        w = abs(Δλ) < 1e-10 ? df[a] : (f[a] - f[b]) / Δλ
        coeff = C * Jsq_buf[a, b] * w
        
        k = round(Int, (Δλ - ω_lo) / ω_step) - k_min + 1
        if 1 <= k <= n_hist
            σ_hist[k] += coeff
        end
    end

    half_width = ceil(Int, 20η / ω_step)
    kernel_xs = (-half_width:half_width) .* ω_step
    kernel = @. (η / π) / (kernel_xs^2 + η^2)
    
    σ_conv = DSP.conv(σ_hist, kernel) 
    
    offset = half_width - k_min
    return σ_conv[offset+1 : offset+n_ω]
end

"""

Electronic observables 

"""

# spin-1/2 occupation number on site i for every site i ∈ Λ
function N_site(λ, ϕ, β, μ)
    v = length(λ)
    n = zeros(v)
    @inbounds for j in 1:v
        fj = 2.0 * fermi(λ[j], β, μ)
        for i in 1:v
            n[i] += abs2(ϕ[i, j]) * fj
        end
    end
    return n
end

# spin-1/2 occupation number on site i 
function N_site_select(λ, ϕ, β, μ, sites)
    v = length(λ)
    n = zeros(length(sites))
    @inbounds for j in 1:v
        fj = 2.0 * fermi(λ[j], β, μ)
        for (k, i) in enumerate(sites)
            n[k] += abs2(ϕ[i, j]) * fj
        end
    end
    return n
end

# spin-1/2 total occupation number 
function N_tot(λ, β, μ)
    return 2 * sum(fermi.(λ, Ref(β), Ref(μ)))   # factor of 2 for spin-1/2
end

# inverse participation ratio 
function ipr(ϕ)
    return vec(sum(abs2.(ϕ).^2, dims=1))
end

function ϵ0(kx, ky, t, tp)
    return -2t * (cos(kx) + cos(ky)) - 4tp * cos(kx) * cos(ky)
end

# Construct 2D quasimomentum eigenstate |kx, ky⟩
function bloch_2d(kx, ky, lx, ly, lz)
    v = Int(lx * ly)
    k_state = zeros(ComplexF64, v)
    for x in 1:lx
        for y in 1:ly
            i = x + (y - 1) * lx 
            k_state[i] = exp(im * (x * kx + y * ky)) 
        end
    end
    return k_state / sqrt(v)
end

# Spectral function at single k-point (kx, ky)
# ϕs: (lx * ly) × (lx * ly) matrix whose columns are eigenstates
# Es: (lx * ly)-vector; each entry is a vector of eigenvalues
# η: same broadening as is used in the MALA + ED

function A_ind(ω, ϕ, Es, ϕk; η)
    K = abs2.(ϕ' * ϕk)
    inv_π = 1 / π
    η2 = η^2
    v = length(Es)
    A = zeros(length(ω))
    @inbounds for a in 1:v
        Ka = K[a]
        Ea = Es[a]
        for k in eachindex(ω)
            x = ω[k] - Ea
            A[k] += Ka * inv_π * η / (x^2 + η2)
        end
    end
    return A
end

function tau_lattice(lx, ly, lz, dir, t, tp; θs, α)
    v = vol(lx, ly, lz)
    H = tb_ham(t, tp, lx, ly, lz; θs = θs, α = α)   # hopping matrix with phases

    # x, y, z component of site i
    xs = [((i-1) % lx) + 1 for i in 1:v]
    ys = [((i-1) ÷ lx) % ly + 1 for i in 1:v]
    zs = [((i-1) ÷ (lx*ly)) + 1 for i in 1:v]

    # pick the coordinate difference depending on direction
    if dir == [1,0,0]
        coord = xs
        l = lx
    elseif dir == [0,1,0]
        coord = ys
        l = ly
    elseif dir == [0,0,1]
        coord = zs
        l = lz
    else
        error("dir must be [1,0,0], [0,1,0], or [0,0,1]")
    end

    R = coord .- coord'                 # build matrix Rᵢⱼ = (rᵢ - rⱼ)
    R .= R .- l .* round.(R ./ l)       # takes care of periodic boundary conditions 

    return - R .* R .* H
end

function tau(E, ϕ, β, μ, tau_space, tau_buf, tauΦ_buf)
    mul!(tauΦ_buf, tau_space, ϕ)
    tau_diag = real.(vec(sum(conj.(ϕ) .* tauΦ_buf, dims=1)))
    return 2 * sum(fermi.(E, β, μ) .* tau_diag)
end

"""

MALA + ED algorithm

"""

# Log electronic partition function
function logZe(λ, β, μ)
    return 2 * sum(log1pexp.(-β .* (λ .- μ)))  # factor of 2 for spin-1/2           
end

# Log target distribution logP
function logP(x, K, γ, λ, β, μ, n0s)
    q = γ .* n0s ./ K            
    q[K .== 0.0] .= 0.0                        # handle case where Kᵢ = 0.0
    return -β * sum(K .* (x .- q).^2) / 2 + logZe(λ, β, μ)
end

# Gradient of log target distribution 
function ∇logP(x, K, γ, λ, ϕ, β, μ, n0s)
    return - β .* (K .* x .+ γ .* N_site(λ, ϕ, β, μ) .- γ .* n0s)
end

# Single MALA step with flip, taking in simulation parameters and eigen from previous step. 
function mala_step(x, K, γ, Δ, H0, H_work, n0s, ν, β, τ, E_current, ϕ_current, μ; mc_rng)
    # Computing logP[x], ∇logP[x], and x'
    logP_current = logP(x, K, γ, E_current, β, μ, n0s)
    ∇logP_current = ∇logP(x, K, γ, E_current, ϕ_current, β, μ, n0s)
    ξ = randn(mc_rng, length(x))
    x_propose = x .+ (τ .* ∇logP_current) .+ (sqrt(2 * τ) .* ξ)

    # Solving H[x']
    H_work .= H0
    @inbounds for i in eachindex(x_propose)
        H_work[i, i] += γ[i] * x_propose[i] + Δ[i]
    end
    @timeit to "ED" E_propose, ϕ_propose = eigen!(Hermitian(H_work))

    # Computing logP[x'] and ∇logP[x']
    logP_propose = logP(x_propose, K, γ, E_propose, β, μ, n0s)
    ∇logP_propose = ∇logP(x_propose, K, γ, E_propose, ϕ_propose, β, μ, n0s)

    # Computing logQ[x → x'] and logQ[x' → x]
    logQ_forward = -sum((x_propose .- x .- τ .* ∇logP_current).^2) / (4 * τ)
    logQ_backward = -sum((x .- x_propose .- τ .* ∇logP_propose).^2) / (4 * τ)

    # Accept/reject
    log_accept_ratio = (logP_propose - logP_current + logQ_backward - logQ_forward)

    if log(rand(mc_rng)) < log_accept_ratio
        return x_propose, E_propose, ϕ_propose, μ, true
    else
        return x, E_current, ϕ_current, μ, false
    end
end

"""

Simulation

"""

# MALA + ED simulation 
function malaed(lx, ly, lz, t, tp, K, γ, Δ, ρf, β, α, ν, s_tot, s_burn; θs, x0, η, mc_rng)
    v = vol(lx, ly, lz)
    d = (lx > 1) + (ly > 1) + (lz > 1)
    K = fill(K, v) 
    Δ = fill(Δ, v)
    x = x0     
    H_work = zeros(ComplexF64, v, v)

    # Solving the free Hamiltonian
    H0 = tb_ham(t, tp, lx, ly, lz; θs = θs, α = α)
    E0, ϕ0 = eigen(Hermitian(H0))

    # Solving for the chemical potential μ to match target filling ν for the free Hamiltonian
    function ΔN_tot_0(μ)
        return N_tot(E0, β, μ) - ν * v
    end
    min_E0, max_E0 = minimum(E0), maximum(E0)
    bracket = (min_E0 - 2/β, max_E0 + 2/β)
    μ0 = find_zero(ΔN_tot_0, bracket, Bisection())

    # Site-resolved particle density for the free Hamiltonian
    n0s = N_site(E0, ϕ0, β, μ0)

    # Solving the full Hamiltonian
    H = H0 + Diagonal(γ .* x) + Diagonal(Δ) 
    E, ϕ = eigen(Hermitian(H))

    # Initialize μ and τ
    μ = μ0
    τ = 0.01 

    # Computing total number of walking steps 
    s = Int(s_tot - s_burn)

    # Defining parameters for tuning time step during burn-in
    a_adapt_window = 100
    n_adapt_window = 500
    a_target = 0.574
    a_tol = 0.05
    n_tol = 0.005
    a_burn = 0
    n_mean = Inf
    E_burn = Vector{Vector{Float64}}()

    @timeit to "tuning" println("Tuning...")
    while a_burn <= a_target - a_tol || a_burn >= a_target + a_tol || abs(ν - n_mean) > n_tol
        a_tot = 0
        adapt_accept_count = 0

        for i in 1:s_burn
            x, E, ϕ, _, a = mala_step(x, K, γ, Δ, H0, H_work, n0s, ν, β, τ, E, ϕ, μ; mc_rng)
            adapt_accept_count += a
            a_tot += a

            # Update τ
            if i % a_adapt_window == 0
                push!(E_burn, E)
                ratio = adapt_accept_count / a_adapt_window
                τ *= exp(0.1 * (ratio - a_target))
                adapt_accept_count = 0
            end

            # Update μ via clamped gradient descent
            if i % n_adapt_window == 0
                n_vals = N_tot.(E_burn, Ref(β), Ref(μ)) ./ v
                n_mean = mean(n_vals)
                n_var = var(n_vals)
                κ = β * n_var * v
                δn = abs(n_mean - ν)
                if δn > 0.1
                    μ_bound = 0.2
                elseif δn > 0.01
                    μ_bound = 0.05
                else
                    μ_bound = 0.01
                end
                Δμ = clamp((ν - n_mean) / κ, -μ_bound, μ_bound)
                μ += Δμ
                empty!(E_burn)
            end
        end

        a_burn = a_tot / s_burn
        if a_target - a_tol <= a_burn <= a_target + a_tol && abs(ν - n_mean) <= n_tol
        else
            println("Failed: β = $(β), Δα = $(a_burn - a_target), Δn = $(ν - n_mean), μ = $(μ), τ = $(τ)")
        end
    end

    # Monte Carlo diagnostics 
    x_max_ind = argmax(abs.(γ))
    xmax_trace = Vector{Float64}(undef, s_tot)

    @timeit to "burning" println("Burning...")
    for i in 1:s_burn
        x, E, ϕ, _, _ = mala_step(x, K, γ, Δ, H0, H_work, n0s, ν, β, τ, E, ϕ, μ; mc_rng)
        xmax_trace[i] = x[x_max_ind]
    end

    # Basic data (unthinned)
    λ_sites = γ.^2 .* ρf
    q_bins = -15.0:0.1:15.0
    q_hist = zeros(Int, v, length(q_bins) - 1)
    q_step = step(q_bins)
    q_lo = first(q_bins)
    data_μ = Vector{Float64}(undef, s)
    data_n = Vector{Float64}(undef, s)
    sorted_idx = sortperm(λ_sites)
    idx_weak = sorted_idx[ceil(Int, length(λ_sites) * 0.05)]
    idx_intermediate = sorted_idx[ceil(Int, length(λ_sites) * 0.5)]
    idx_strong = sorted_idx[ceil(Int, length(λ_sites) * 0.95)]
    data_n_sites = [Vector{Float64}(undef, s) for _ in 1:3]
    
    # Monte Carlo diagnostic data (unthinned)
    data_accept = 0

    # Annealed disorder averaged observables (unthinned)
    omega_op = 0.0:0.01:10.0
    omega_A = -10.0:0.01:10.0
    Je_buf = zeros(ComplexF64, v, v)
    JΦ_buf = zeros(ComplexF64, v, v)
    Jsq_buf = zeros(Float64, v, v)
    Jsite_x = sparse(J_site(lx, ly, lz, [1, 0, 0], t, tp; θs = θs, α = α))
    Jsite_y = sparse(J_site(lx, ly, lz, [0, 1, 0], t, tp; θs = θs, α = α))
    tau_space_x = sparse(tau_lattice(lx, ly, lz, [1, 0, 0], t, tp; θs = θs, α = α))
    tau_space_y = sparse(tau_lattice(lx, ly, lz, [0, 1, 0], t, tp; θs = θs, α = α))
    tau_buf = zeros(ComplexF64, v, v)
    tauΦ_buf = zeros(ComplexF64, v, v)
    op_cond_fast_x = zeros(length(omega_op))
    op_cond_fast_y = zeros(length(omega_op))
    A_n = zeros(length(omega_A))
    A_a = zeros(length(omega_A))
    epsilon_bins = -10.0:0.01:10.0
    n_ebins = length(epsilon_bins) - 1
    e_step = step(epsilon_bins)
    e_lo = first(epsilon_bins)
    ll = zeros(n_ebins)
    dos = zeros(n_ebins)
    Dx = 0.0            # exact Drude weight in x-direction 
    Dy = 0.0            # exact Drude weight in y-direction

    kn, ka = find_nodal_antinodal(lx, ly, t, tp, μ0)
    ϕk_n = bloch_2d(kn[1], kn[2], lx, ly, lz)
    ϕk_a = bloch_2d(ka[1], ka[2], lx, ly, lz)

    println("Walking...")
    w_i = 1
    walk_steps = s_tot - s_burn
    print_interval = max(1, walk_steps ÷ 20)
    q = zeros(v)

    for i in s_burn+1:s_tot
        x, E, ϕ, _, a = mala_step(x, K, γ, Δ, H0, H_work, n0s, ν, β, τ, E, ϕ, μ; mc_rng)

        tb_i = i - s_burn

        # progress tracker 
        if tb_i % print_interval == 0
            percent = round(Int, 100 * tb_i / walk_steps)
            println("  Walking progress: $percent% ($tb_i / $walk_steps steps)")
            flush(stdout) 
        end

        xmax_trace[i] = x[x_max_ind]
        @timeit to "Drude weight" Dx += tau(E, ϕ, β, μ, tau_space_x, tau_buf, tauΦ_buf)
        @timeit to "Drude weight" Dy += tau(E, ϕ, β, μ, tau_space_y, tau_buf, tauΦ_buf)

        q .= γ .* x
        @timeit to "qi" for (site, qi) in enumerate(q)
            k = floor(Int, (qi - q_lo) / q_step) + 1
            if 1 <= k <= size(q_hist, 2)
                q_hist[site, k] += 1
            end
        end

        @timeit to "ni" n_sites = N_site_select(E, ϕ, β, μ, (idx_weak, idx_intermediate, idx_strong))
        data_n_sites[1][w_i] = n_sites[1]
        data_n_sites[2][w_i] = n_sites[2]
        data_n_sites[3][w_i] = n_sites[3]
        data_μ[w_i] = μ
        data_n[w_i] = N_tot(E, β, μ) / v    # total spinful density 

        @timeit to "sigma" op_cond_fast_x .+= op_cond_fast(omega_op, E, ϕ, β, μ, lx, ly, lz, Jsite_x, Je_buf, JΦ_buf, Jsq_buf; η = η) 
        @timeit to "sigma" op_cond_fast_y .+= op_cond_fast(omega_op, E, ϕ, β, μ, lx, ly, lz, Jsite_y, Je_buf, JΦ_buf, Jsq_buf; η = η) 

        ipr_vals = ipr(ϕ)
        @timeit to "ll and dos" for j in 1:v
            k = floor(Int, ((E[j] - μ) - e_lo) / e_step) + 1
            if 1 <= k <= n_ebins
                ll[k] += ipr_vals[j]^(-1/d)
                dos[k] += 1
            end
        end

        @timeit to "spectral" A_n .+= A_ind(omega_A, ϕ, E .- μ, ϕk_n; η = η)
        @timeit to "spectral" A_a .+= A_ind(omega_A, ϕ, E .- μ, ϕk_a; η = η)

        data_accept += a
        w_i += 1

    end
    accept_ratio = data_accept / s
    op_cond_fast_x = op_cond_fast_x ./ s
    op_cond_fast_y = op_cond_fast_y ./ s
    dc_cond_fast_x = op_cond_fast_x[1]
    dc_cond_fast_y = op_cond_fast_y[1]
    A_n = A_n ./ s 
    A_a = A_a ./ s
    dos_mean = dos ./ (s * v * e_step)        
    ll_mean = ll ./ max.(dos, 1)
    Dx = Dx / (s * v)
    Dy = Dy / (s * v)
    n_mean = mean(data_n)
    n_var = var(data_n)

    diagnostics = (
        accept_ratio = accept_ratio,
        xmax_trace = xmax_trace, 
    )

    observables = (
        q_bins = q_bins,
        q_hist = q_hist,
        data_n = data_n,
        data_n_sites = data_n_sites,
        wis_sites = (idx_weak, idx_intermediate, idx_strong),
        data_μ = data_μ,
        γ = γ,
        λ_sites = λ_sites,
        θs = θs,
        Δ = Δ,
        ν = ν,
        τ = τ,
        omega_op = omega_op,
        omega_A = omega_A,
        epsilon_bins = epsilon_bins,
        kn = kn, 
        ka = ka,
        op_cond_fast_x = op_cond_fast_x, 
        op_cond_fast_y = op_cond_fast_y,
        dc_cond_fast_x = dc_cond_fast_x,
        dc_cond_fast_y = dc_cond_fast_y,
        A_n = A_n,
        A_a = A_a,
        ll = ll_mean,
        dos = dos_mean,
        Dx = Dx,
        Dy = Dy, 
        n_mean = n_mean, 
        n_var = n_var
    )

    return diagnostics, observables 
    
end

# Single function to run the simulation at given parameters 
function simulate(lx::Int, ly::Int, lz::Int, t::Float64, tp::Float64, K::Float64, Δ::Float64, ρf::Float64, λ::Float64, T::Float64, α::Float64, ν::Float64, s_tot::Int, s_burn::Int; 
                  θs=zeros(3), folder, η = 2.0 / lx, seed_name=false, γ_seed = rand(UInt64), x0_seed = rand(UInt64), mc_seed = rand(UInt64))
    
    w = λ / ρf
    β = 1 / T

    # γᵢ ~ Uniform(-√(3w), √(3w))
    γ_rng = Xoshiro(γ_seed)
    γ = rand(γ_rng, Uniform(-sqrt(3 * w), sqrt(3 * w)), vol(lx, ly, lz))    # λ = λ̄ = <γ²>ρf/K in this convention 
    
    x0_rng = Xoshiro(x0_seed)
    x0 = rand(x0_rng, Normal(0, sqrt(T)), vol(lx, ly, lz))

    mc_rng = Xoshiro(mc_seed)

    seeds = (g_seed=γ_seed, x0_seed=x0_seed, mc_seed=mc_seed)
    parameters = (lx, ly, lz, t, tp, K, Δ, ρf, λ, T, α, ν, s_tot, s_burn)
    malaed_parameters = (lx, ly, lz, t, tp, K, γ, Δ, ρf, β, α, ν, s_tot, s_burn)
    diagnostics, observables = malaed(malaed_parameters...; θs=θs, x0=x0, η=η, mc_rng=mc_rng)

    # saving data
    if seed_name 
        file_name = (lx, ly, lz, t, tp, K, Δ, ρf, λ, T, α, ν, s_tot, s_burn, γ_seed, x0_seed, mc_seed)
    else
        file_name = (lx, ly, lz, t, tp, K, Δ, ρf, λ, T, α, ν, s_tot, s_burn, γ_seed)
    end
    file = encode(file_name)
    file_path = joinpath(folder, file)
    mkpath(dirname(file_path))
    jldsave(file_path; 
            seeds = seeds,
            pars = parameters,
            diagnostics = diagnostics,
            observables = observables
        )
    println("Saved: $file_path")
end

end