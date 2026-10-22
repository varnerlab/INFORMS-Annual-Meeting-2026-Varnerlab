using Test, LinearAlgebra, Random, JuMP, VLINFORMS2026
include("helpers.jl")
const ETAS = [0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0]          # both spec grids
const CASES = (([0.2, 0.5, 0.3], [10.0, 20.0, 40.0], 1000.0),
    (0.05 .+ 0.2 .* rand(MersenneTwister(5660), 13), 10.0 .+ 490.0 .* rand(MersenneTwister(5661), 13), 1e5))

@testset "budget_only_weights: the L8b closed forms" begin
    γ = [0.2, 0.5, 0.3]; S = [10.0, 20.0, 40.0]
    @test budget_only_weights(MyCobbDouglasRule(), γ, S; W=1000.0) ≈ γ ./ sum(γ)
    for η in ETAS
        n = (γ ./ S) .^ η
        @test budget_only_weights(build(MyCESRule, (η=η,)), γ, S; W=1000.0) ≈ S .* n ./ sum(S .* n)
    end
    @test budget_only_weights(build(MyCESRule, (η=1.0,)), γ, S; W=1000.0) ≈ γ ./ sum(γ)
    # a γ ≤ 0 name holds ε shares, paid first: 5 shares × 20 USD, then 900 USD split 0.4 / 0.6
    @test budget_only_weights(MyCobbDouglasRule(), [0.2, -0.5, 0.3], S; W=1000.0, ε=5.0) ≈ [0.36, 0.1, 0.54]
    # no γ > 0: only the pins
    @test budget_only_weights(MyCobbDouglasRule(), [-0.2, 0.0, -0.3], S; W=1000.0) ≈ 1e-3 .* S ./ 1000
end

@testset "CES matches nᵢ ∝ (γᵢ/Sᵢ)^η at every η (budget only)" begin
    for (γ, S, W) in CASES, η in ETAS
        rule = build(MyCESRule, (η=η,))
        r = allocate(rule, P(γ=γ, prices=S, W=W, σ_g_max=Inf))
        @test r.status == :optimal
        @test r.w ≈ budget_only_weights(rule, γ, S; W=W) atol=1e-4
    end
    for η in (0.05, 20.0)                                             # with a pinned name
        rule = build(MyCESRule, (η=η,)); r = allocate(rule, P(γ=[0.2, -0.5, 0.3], σ_g_max=Inf))
        @test r.n[2] == 1e-3
        @test r.w ≈ budget_only_weights(rule, [0.2, -0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) atol=1e-4
    end
end

@testset "CESw matches wᵢ ∝ γᵢ^η at every η (budget only)" begin
    @test budget_only_weights(build(MyCESWeightRule, (η=2.0,)), [0.2, 0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) ≈
        [0.04, 0.25, 0.09] ./ 0.38
    for (γ, S, W) in CASES, η in ETAS
        rule = build(MyCESWeightRule, (η=η,))
        r = allocate(rule, P(γ=γ, prices=S, W=W, σ_g_max=Inf))
        @test r.status == :optimal
        @test r.w ≈ γ .^ η ./ sum(γ .^ η) atol=1e-4
    end
    for η in (0.05, 20.0)
        rule = build(MyCESWeightRule, (η=η,)); r = allocate(rule, P(γ=[0.2, -0.5, 0.3], σ_g_max=Inf))
        @test r.n[2] == 1e-3
        @test r.w ≈ budget_only_weights(rule, [0.2, -0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) atol=1e-4
    end
    @test allocate(build(MyCESWeightRule, (η=1.0,)), P()).n ≈ allocate(MyCobbDouglasRule(), P()).n rtol=1e-8
    @test_throws ArgumentError build(MyCESWeightRule, (η=0.0,))
end

@testset "a split changes CES's weights, not CESw's" begin
    S2 = [1.0, 20.0, 40.0]                                            # name 1 splits 10:1, γ unchanged
    for η in (0.25, 4.0)
        cw = build(MyCESWeightRule, (η=η,)); ces = build(MyCESRule, (η=η,))
        @test allocate(cw, P(prices=S2, σ_g_max=Inf)).w ≈ allocate(cw, P(σ_g_max=Inf)).w atol=1e-5
        @test maximum(abs.(allocate(ces, P(prices=S2, σ_g_max=Inf)).w .- allocate(ces, P(σ_g_max=Inf)).w)) > 0.05
    end
end

@testset "CESw under the shared limits" begin
    Σ = Diagonal([9.0, 16.0, 25.0])
    for η in (0.5, 2.0)
        cw = build(MyCESWeightRule, (η=η,))
        @test maximum(allocate(cw, P(w_max=0.4)).w) ≤ 0.4 + 1e-6
        r = allocate(cw, P(Σ_g=Σ, σ_g_max=1.5)); @test sqrt(r.w' * Σ * r.w) ≈ 1.5 rtol=1e-5
        @test sum(allocate(cw, P(sector=["A", "A", "B"], s_max=0.5)).w[1:2]) ≈ 0.5 rtol=1e-5
        @test allocate(cw, P(γ=[0.2, -0.5, 0.3])).n[2] == 1e-3
        r = allocate(cw, P(γ=[-0.1, 0.0, -0.3])); @test r.status == :no_preferred && all(r.n .== 1e-3)
    end
end

@testset "extreme η with a binding β cap and a hedge name" begin              # Review Focus 2
    γ = [0.5, 0.4, -0.2, -0.1]; β = [1.2, 1.0, -0.5, -0.5]; S = [10.0, 20.0, 40.0, 25.0]
    for R in (MyCESRule, MyCESWeightRule), η in (0.05, 20.0)
        r = allocate(build(R, (η=η,)), P(γ=γ, prices=S, β=β, β_max=0.4, hedge=[false, false, true, false], w_max=0.5))
        @test r.status == :optimal && dot(r.w, β) ≤ 0.4 + 1e-6 && r.n[4] == 1e-3
    end
end

@testset "cap off: concentration stops at the volatility cap" begin            # Review Focus 3
    Σ = Diagonal([9.0, 16.0, 25.0])
    for R in (MyCESRule, MyCESWeightRule)
        r = allocate(build(R, (η=10.0,)), P(Σ_g=Σ, σ_g_max=1.5, w_max=1.0))
        @test r.status == :optimal
        @test sqrt(r.w' * Σ * r.w) ≤ 1.5 + 1e-6
        @test sqrt(r.w' * Σ * r.w) ≈ 1.5 rtol=1e-5
    end
end

@testset "solver outcomes" begin
    V = VLINFORMS2026
    @test V._solver_outcome([MOI.OPTIMAL]) == :clarabel
    @test V._solver_outcome([MOI.ALMOST_OPTIMAL]) == :clarabel_almost
    @test V._solver_outcome([MOI.SLOW_PROGRESS, MOI.OPTIMAL]) == :scs
    @test V._solver_outcome([MOI.SLOW_PROGRESS, MOI.ALMOST_OPTIMAL]) == :scs
    @test V._solver_outcome([MOI.INFEASIBLE, MOI.INFEASIBLE]) == :failed
    @test V._worst_solver(Symbol[]) == :none && V._worst_solver([:clarabel, :scs, :clarabel_almost]) == :scs
    log = Symbol[]
    r = Base.ScopedValues.with(() -> allocate(MyCobbDouglasRule(), P(σ_g_max=Inf)), V._SOLVER_LOG => log)
    @test r.status == :optimal && length(log) == 1 && log[1] in (:clarabel, :clarabel_almost)
    allocate(MyCobbDouglasRule(), P(σ_g_max=Inf))                     # outside a scope: not recorded
    @test length(log) == 1
end
