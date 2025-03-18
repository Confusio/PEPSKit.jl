abstract type GradMode{F} end

const GRADIENT_MODE_SYMBOLS = IdDict{Symbol,Type{<:GradMode}}()
const LINSOLVER_SOLVER_SYMBOLS = IdDict{Symbol,Type{<:KrylovKit.LinearSolver}}(
    :gmres => GMRES, :bicgstab => BiCGStab
)
const EIGSOLVER_SOLVER_SYMBOLS = IdDict{Symbol,Type{<:KrylovKit.KrylovAlgorithm}}(
    :arnoldi => Arnoldi
)

function GradMode(;
    alg=Defaults.gradient_alg,
    tol=Defaults.gradient_tol,
    maxiter=Defaults.gradient_maxiter,
    verbosity=Defaults.gradient_verbosity,
    iterscheme=Defaults.gradient_iterscheme,
    solver_alg=(;),
)
    # replace symbol with GradMode alg type
    haskey(GRADIENT_MODE_SYMBOLS, alg) ||
        throw(ArgumentError("unknown GradMode algorithm: $alg"))
    alg_type = GRADIENT_MODE_SYMBOLS[alg]

    # parse GradMode algorithm
    gradient_algorithm = if alg_type <: Union{GeomSum,ManualIter}
        alg_type{iterscheme}(tol, maxiter, verbosity)
    elseif alg_type <: Union{<:LinSolver,<:EigSolver}
        solver = if solver_alg isa NamedTuple # determine linear/eigen solver algorithm
            solver_kwargs = (; tol, maxiter, verbosity, solver_alg...)

            solver_type = if alg_type <: LinSolver # replace symbol with solver alg type
                solver_kwargs = (; alg=Defaults.gradient_linsolver, solver_kwargs...)
                haskey(LINSOLVER_SOLVER_SYMBOLS, solver_kwargs.alg) || throw(
                    ArgumentError("unknown LinSolver solver: $(solver_kwargs.alg)"),
                )
                LINSOLVER_SOLVER_SYMBOLS[solver_kwargs.alg]
            elseif alg_type <: EigSolver
                solver_kwargs = (;
                    alg=Defaults.gradient_eigsolver,
                    eager=Defaults.gradient_eigsolver_eager,
                    solver_kwargs...,
                )
                haskey(EIGSOLVER_SOLVER_SYMBOLS, solver_kwargs.alg) || throw(
                    ArgumentError("unknown EigSolver solver: $(solver_kwargs.alg)"),
                )
                EIGSOLVER_SOLVER_SYMBOLS[solver_kwargs.alg]
            end

            solver_kwargs = Base.structdiff(solver_kwargs, (; alg=nothing)) # remove `alg` keyword argument
            solver_type(; solver_kwargs...)
        else
            solver_alg
        end

        alg_type{iterscheme}(solver)
    else
        throw(ArgumentError("unknown gradient algorithm: $alg"))
    end

    return gradient_algorithm
end

iterscheme(::GradMode{F}) where {F} = F

"""
    struct GeomSum <: GradMode{iterscheme}
    GeomSum(; kwargs...)

Gradient mode for CTMRG using explicit evaluation of the geometric sum.

## Keyword arguments

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance for the difference of norms of two consecutive summands in the geometric sum.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity that can be one of the following:
    0. Suppress output information
    1. Print convergence warnings
    2. Information at each gradient iteration
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)` : Style of CTMRG iteration which is being differentiated, which can be:
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well
"""
struct GeomSum{F} <: GradMode{F}
    tol::Real
    maxiter::Int
    verbosity::Int
end
GeomSum(; kwargs...) = GradMode(; alg=:geomsum, kwargs...)

GRADIENT_MODE_SYMBOLS[:geomsum] = GeomSum

"""
    struct ManualIter <: GradMode{iterscheme}
    ManualIter(; kwargs...)

Gradient mode for CTMRG using manual iteration to solve the linear problem.

## Keyword arguments

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance for the norm difference of two consecutive `dx` contributions.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity that can be one of the following:
    0. Suppress output information
    1. Print convergence warnings
    2. Information at each gradient iteration
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)` : Style of CTMRG iteration which is being differentiated, which can be:
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well
"""
struct ManualIter{F} <: GradMode{F}
    tol::Real
    maxiter::Int
    verbosity::Int
end
ManualIter(; kwargs...) = GradMode(; alg=:manualiter, kwargs...)

GRADIENT_MODE_SYMBOLS[:manualiter] = ManualIter

"""
    struct LinSolver <: GradMode{iterscheme}
    LinSolver(; kwargs...)

Gradient mode wrapper around `KrylovKit.LinearSolver` for solving the gradient linear
problem using iterative solvers.

## Keyword arguments

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance of the linear solver.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of solver iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity of the linear solver.
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)` : Style of CTMRG iteration which is being differentiated, which can be:
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well
* `solver_alg::Union{KrylovKit.LinearSolver,NamedTuple}=(; alg::Symbol=:$(Defaults.gradient_linsolver)` : Linear solver algorithm which, if supplied directly as a `KrylovKit.LinearSolver` overrides the above specified `tol`, `maxiter` and `verbosity`. Alternatively, it can be supplied via a `NamedTuple` where `alg` can be one of the following:
    - `:gmres` : GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab` : BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
"""
struct LinSolver{F} <: GradMode{F}
    solver_alg::KrylovKit.LinearSolver
end
LinSolver(; kwargs...) = GradMode(; alg=:linsolver, kwargs...)

GRADIENT_MODE_SYMBOLS[:linsolver] = LinSolver

"""
    struct EigSolver <: GradMode{iterscheme}
    EigSolver(; kwargs...)

Gradient mode wrapper around `KrylovKit.KrylovAlgorithm` for solving the gradient linear
problem as an eigenvalue problem.

## Keyword arguments

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance of the eigen solver.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of solver iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity of the linear solver.
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)` : Style of CTMRG iteration which is being differentiated, which can be:
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well
* `solver_alg::Union{KrylovKit.KrylovAlgorithm,NamedTuple}=(; alg=:$(Defaults.gradient_eigsolver)` : Eigen solver algorithm which, if supplied directly as a `KrylovKit.KrylovAlgorithm` overrides the above specified `tol`, `maxiter` and `verbosity`. Alternatively, it can be supplied via a `NamedTuple` where `alg` can be one of the following:
    - `:arnoldi` : Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
"""
struct EigSolver{F} <: GradMode{F}
    solver_alg::KrylovKit.KrylovAlgorithm
end
EigSolver(; kwargs...) = GradMode(; alg=:eigsolver, kwargs...)

GRADIENT_MODE_SYMBOLS[:eigsolver] = EigSolver

#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

function _rrule(
    gradmode::GradMode{:diffgauge},
    config::RuleConfig,
    ::typeof(leading_boundary),
    envinit,
    state,
    alg::CTMRGAlgorithm,
)
    env, info = leading_boundary(envinit, state, alg)
    alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation() # fix spaces during differentiation

    function leading_boundary_diffgauge_pullback((Δenv′, Δinfo))
        Δenv = unthunk(Δenv′)

        # find partial gradients of gauge_fixed single CTMRG iteration
        function f(A, x)
            return gauge_fix(x, ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1])[1]
        end
        _, env_vjp = rrule_via_ad(config, f, state, env)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(env) = env_vjp(x)[3]
        ∂F∂env = fpgrad(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂env, NoTangent()
    end

    return (env, info), leading_boundary_diffgauge_pullback
end

# Here f is differentiated from an pre-computed SVD with fixed U, S and V
function _rrule(
    gradmode::GradMode{:fixed},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::SimultaneousCTMRG,
)
    env, = leading_boundary(envinit, state, alg)
    alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation() # fix spaces for gauge fixing
    env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, alg_fixed)
    env_fixed, signs = gauge_fix(env, env_conv)

    # Fix SVD
    svd_alg_fixed = _fix_svd_algorithm(alg.projector_alg.svd_alg, signs, info)
    alg_fixed = @set alg.projector_alg.svd_alg = svd_alg_fixed
    alg_fixed = @set alg_fixed.projector_alg.trscheme = notrunc()

    function leading_boundary_fixed_pullback((Δenv′, Δinfo))
        Δenv = unthunk(Δenv′)

        function f(A, x)
            return fix_global_phases(
                x, ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1]
            )
        end
        _, env_vjp = rrule_via_ad(config, f, state, env_fixed)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(env) = env_vjp(x)[3]
        ∂F∂env = fpgrad(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂env, NoTangent()
    end

    return (env_fixed, info), leading_boundary_fixed_pullback
end

function _fix_svd_algorithm(alg::SVDAdjoint, signs, info)
    # embed gauge signs in larger space to fix gauge of full U and V on truncated subspace
    signs_full = map(zip(signs, info.S_full)) do (σ, S_full)
        extended_σ = zeros(scalartype(σ), space(S_full))
        for (c, b) in blocks(extended_σ)
            σc = block(σ, c)
            kept_dim = size(σc, 1)
            b[diagind(b)] .= one(scalartype(σ)) # put ones on the diagonal
            b[1:kept_dim, 1:kept_dim] .= σc # set to σ on kept subspace
        end
        return extended_σ
    end

    # fix kept and full U and V
    U_fixed, V_fixed = fix_relative_phases(info.U, info.V, signs)
    U_full_fixed, V_full_fixed = fix_relative_phases(info.U_full, info.V_full, signs_full)
    return SVDAdjoint(;
        fwd_alg=FixedSVD(U_fixed, info.S, V_fixed, U_full_fixed, info.S_full, V_full_fixed),
        rrule_alg=alg.rrule_alg,
        broadening=alg.broadening,
    )
end
function _fix_svd_algorithm(alg::SVDAdjoint{F}, signs, info) where {F<:IterSVD}
    # fix kept U and V only since iterative SVD doesn't have access to full spectrum
    U_fixed, V_fixed = fix_relative_phases(info.U, info.V, signs)
    return SVDAdjoint(;
        fwd_alg=FixedSVD(U_fixed, info.S, V_fixed, nothing, nothing, nothing),
        rrule_alg=alg.rrule_alg,
        broadening=alg.broadening,
    )
end

@doc """
    fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y0, alg)

Compute the gradient of the cost function for CTMRG by solving the following equation:

dx = ∑ₙ (∂f∂x)ⁿ ∂f∂A dA = (1 - ∂f∂x)⁻¹ ∂f∂A dA

where `∂F∂x` is the gradient of the cost function with respect to the PEPS tensors, `∂f∂x`
is the partial gradient of the CTMRG iteration with respect to the environment tensors,
`∂f∂A` is the partial gradient of the CTMRG iteration with respect to the PEPS tensors, and
`y0` is the initial guess for the fixed-point iteration. The function returns the gradient
`dx` of the fixed-point iteration.
"""
fpgrad

# TODO: can we construct an implementation that does not need to evaluate the vjp
# twice if both ∂f∂A and ∂f∂x are needed?
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, _, alg::GeomSum)
    g = ∂F∂x
    dx = ∂f∂A(g) # n = 0 term: ∂F∂x ∂f∂A
    ϵ = 2 * alg.tol
    for i in 1:(alg.maxiter)
        g = ∂f∂x(g)
        Σₙ = ∂f∂A(g)
        dx += Σₙ
        ϵnew = norm(Σₙ)  # TODO: normalize this error?
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 &&
            @printf("Gradient iter: %3d   ‖Σₙ‖: %.2e   Δ‖Σₙ‖: %.2e\n", i, ϵnew, Δϵ)
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Σₙ‖ = $ϵ"
        end
    end
    return dx
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::ManualIter)
    y = deepcopy(y₀)  # Do not mutate y₀
    dx = ∂f∂A(y)
    ϵ = 1.0
    for i in 1:(alg.maxiter)
        y′ = ∂F∂x + ∂f∂x(y)

        dxnew = ∂f∂A(y′)
        ϵnew = norm(dxnew - dx)
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 && @printf(
            "Gradient iter: %3d   ‖Cᵢ₊₁-Cᵢ‖/N: %.2e   Δ‖Cᵢ₊₁-Cᵢ‖/N: %.2e\n", i, ϵnew, Δϵ
        )
        y = y′
        dx = dxnew
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Cᵢ₊₁-Cᵢ‖ = $ϵ"
        end
    end
    return dx
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::LinSolver)
    y, info = reallinsolve(∂f∂x, ∂F∂x, y₀, alg.solver_alg, 1, -1)
    if alg.solver_alg.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, x₀, alg::EigSolver)
    function f(X)
        y = ∂f∂x(X[1])
        return (y + X[2] * ∂F∂x, X[2])
    end
    X₀ = (x₀, one(scalartype(x₀)))
    _, vecs, info = realeigsolve(f, X₀, 1, :LM, alg.solver_alg)
    if alg.solver_alg.verbosity > 0 && info.converged < 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end
    if norm(vecs[1][2]) < 1e-2 * alg.solver_alg.tol
        @warn "Fixed-point gradient computation using Arnoldi failed: auxiliary component should be finite but was $(vecs[1][2]). Possibly the Jacobian does not have a unique eigenvalue 1."
    end
    y = scale(vecs[1][1], 1 / vecs[1][2])

    return ∂f∂A(y)
end
