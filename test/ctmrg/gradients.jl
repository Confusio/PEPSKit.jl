using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

## Test models, gradmodes and CTMRG algorithm
# -------------------------------------------
χbond = 2
χenv = 4
Pspaces = [ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0 => χbond / 2, 1 => χbond / 2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)]
models = [heisenberg_XYZ(InfiniteSquare()), pwave_superconductor(InfiniteSquare())]
names = ["Heisenberg", "p-wave superconductor"]

gradtol = 1e-4
ctmrg_algs = [
    [
        SimultaneousCTMRG(; verbosity=0, projector_alg=HalfInfiniteProjector),
        SimultaneousCTMRG(; verbosity=0, projector_alg=FullInfiniteProjector),
    ],
    [
        SequentialCTMRG(; verbosity=0, projector_alg=HalfInfiniteProjector),
    ],
]
gradmodes = [
    [
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:fixed),
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:fixed),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
    ],
    [  # Only use :diffgauge due to high gauge-sensitivity (perhaps due to small χenv?)
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
    ],
]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD CTMRG energy gradients for $(names[i]) model" verbose = true for i in [2]
#                                                                               eachindex(
#     models
# )
    Pspace = Pspaces[i]
    Vspace = Pspaces[i]
    Espace = Espaces[i]
    gms = gradmodes[i]
    calgs = ctmrg_algs[i]
    psi_init = InfinitePEPS(Pspace, Vspace, Vspace)
    @testset "$ctmrg_alg and $alg_rrule" for (ctmrg_alg, alg_rrule) in
                                             Iterators.product(calgs, gms)
        @info "optimtest of $ctmrg_alg and $alg_rrule on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPS(Pspace, Vspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace, Vspace)
        env = leading_boundary(CTMRGEnv(psi, Espace), psi, ctmrg_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.peps_retract,
            inner=PEPSKit.real_inner,
        ) do (peps, envs)
            E, g = Zygote.withgradient(peps) do psi
                envs2 = PEPSKit.hook_pullback(leading_boundary, envs, psi, ctmrg_alg; alg_rrule)
                return costfun(psi, envs2, models[i])
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
