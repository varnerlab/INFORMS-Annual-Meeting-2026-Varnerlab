using Test, Dates, VLINFORMS2026
include("helpers.jl")

# A hand-made path diagnostic for the gate tests: 4 names in 2 sectors, every value on its target unless a keyword
# replaces it.
function fake_diag(; kw...)
    base = (ew_gap=0.01, var_ratio_draw=fill(1.0, 4), var_ratio_real=fill(1.07, 4), β_diff=fill(0.01, 4),
        acf_syn=fill(0.2, 4, 3), acf_draw=fill(0.2, 4, 3), acf_real=fill(0.3, 4, 3),
        sector_corr=[0.10, 0.20], cross_corr=0.02, ewls_sd=fill(0.3, 4), acf252=0.18, n_beta_le0=0,
        α_spread=0.076, α_corr=0.18, clip_share=0.01)
    return merge(base, NamedTuple(kw))
end

# The matching hand-made real reference.
fake_ref() = (sectors=["A", "B"], sector_corr=[0.10, 0.20], cross_corr=0.02, sd_β=fill(0.3, 4), acf252=0.18,
    acf=fill(0.3, 4, 3), ew_gap=0.01, spread=0.076, corr=0.18, spy_growth=0.10, n_beta_le0=2)

# The pass values of the gated rows of one check, in table order.
gates(t, check) = t.pass[(t.check .== check) .& .!ismissing.(t.pass)]

@testset "check helpers" begin
    G = [1.0 3.0; -2.0 0.5; 4.0 -1.0]; Δt = 1/252
    reb = sum(log(mean(exp.(G[t, :] .* Δt))) for t in 1:3); bh = log(mean(exp.(vec(sum(G; dims=1)) .* Δt)))
    @test ew_rebalance_gap(G) ≈ (reb - bh) / (3Δt)
    @test ew_rebalance_gap(hcat(G[:, 1], G[:, 1])) ≈ 0.0 atol=1e-12
    x = randn(Xoshiro(4), 1000); @test abs_acf(x, [1, 5]) ≈ [cor(abs.(x[1:end-1]), abs.(x[2:end])), cor(abs.(x[1:end-5]), abs.(x[6:end]))]
    R = randn(Xoshiro(5), 500, 3); C = cor(R); @test mean_pairwise_corr(R) ≈ (C[1,2] + C[1,3] + C[2,3]) / 3
end
@testset "check gates" begin
    ref = fake_ref(); v1 = (median_growth=0.10, n_paths=10)
    ok = [fake_diag() for _ in 1:3]
    t = check_table(ok, ref, v1)
    @test names(t) == ["check", "statistic", "synthetic", "reference", "threshold", "pass"]
    @test nrow(t) == 20 && count(ismissing, t.pass) == 8 && all(skipmissing(t.pass))      # 12 gates, all on target
    @test sort(unique(t.check[ismissing.(t.pass)])) == ["V3", "V4", "V5", "V6"]           # report-only rows
    # V1: |median train growth − real| ≤ 0.02
    @test gates(check_table(ok, ref, (median_growth=0.119, n_paths=10)), "V1") == [true]
    @test gates(check_table(ok, ref, (median_growth=0.121, n_paths=10)), "V1") == [false]
    @test gates(check_table(ok, ref, (median_growth=0.079, n_paths=10)), "V1") == [false]
    # V2: the median over paths is gated, so one outlying path does not decide it
    @test gates(check_table([fake_diag(ew_gap=g) for g in (0.029, 0.029, 0.5)], ref, v1), "V2") == [true]
    @test gates(check_table([fake_diag(ew_gap=g) for g in (0.031, 0.031, 0.0)], ref, v1), "V2") == [false]
    # V3: variance ratio within [0.95, 1.05], then median |β_diff| ≤ 0.05
    @test gates(check_table([fake_diag(var_ratio_draw=fill(1.04, 4)) for _ in 1:3], ref, v1), "V3") == [true, true]
    @test gates(check_table([fake_diag(var_ratio_draw=fill(1.06, 4)) for _ in 1:3], ref, v1), "V3") == [false, true]
    @test gates(check_table([fake_diag(var_ratio_draw=fill(0.94, 4)) for _ in 1:3], ref, v1), "V3") == [false, true]
    @test gates(check_table([fake_diag(β_diff=fill(0.04, 4)) for _ in 1:3], ref, v1), "V3") == [true, true]
    @test gates(check_table([fake_diag(β_diff=fill(-0.06, 4)) for _ in 1:3], ref, v1), "V3") == [true, false]
    # per name the median over paths, then the median over names: name medians [1, 1, 1.08, 1.08] give 1.04
    vr = [[1.0, 1.0, 1.08, 1.08], [1.0, 1.0, 1.08, 1.08], fill(1.08, 4)]
    tv = check_table([fake_diag(var_ratio_draw=v) for v in vr], ref, v1)
    @test tv.synthetic[findfirst(==("V3"), tv.check)] ≈ 1.04 && gates(tv, "V3") == [true, true]
    # V4: one row per lag (1, 5, 20); only lag 5 moves by 0.06
    s5 = fill(0.2, 4, 3); s5[:, 2] .= 0.26
    @test gates(check_table([fake_diag(acf_syn=s5) for _ in 1:3], ref, v1), "V4") == [true, false, true]
    s5[:, 2] .= 0.24
    @test gates(check_table([fake_diag(acf_syn=s5) for _ in 1:3], ref, v1), "V4") == [true, true, true]
    # V5: one row per sector
    @test gates(check_table([fake_diag(sector_corr=[0.10, 0.26]) for _ in 1:3], ref, v1), "V5") == [true, false]
    @test gates(check_table([fake_diag(sector_corr=[0.06, 0.20]) for _ in 1:3], ref, v1), "V5") == [true, true]
    # V6: median over names of (median-over-paths ewls_sd ÷ sd_β) within [0.9, 1.1]
    @test gates(check_table([fake_diag(ewls_sd=fill(0.3 * 1.08, 4)) for _ in 1:3], ref, v1), "V6") == [true]
    @test gates(check_table([fake_diag(ewls_sd=fill(0.3 * 1.12, 4)) for _ in 1:3], ref, v1), "V6") == [false]
    @test gates(check_table([fake_diag(ewls_sd=fill(0.3 * 0.88, 4)) for _ in 1:3], ref, v1), "V6") == [false]
    # V6 reports, after the gate and acf252 rows: the share of paths with a name whose β̂ ≤ 0 next to the real
    # market's own yes/no, then the median count of such names over paths next to the real count
    tn = check_table([fake_diag(n_beta_le0=n) for n in (0, 1, 3)], ref, v1)
    v6 = findall(==("V6"), tn.check)
    share = v6[3]
    names_le0 = v6[4]
    @test length(v6) == 4 && ismissing(tn.pass[share]) && ismissing(tn.pass[names_le0])
    @test tn.synthetic[share] ≈ 2 / 3 && tn.reference[share] == 1.0                    # real has 2 such names: yes
    @test tn.synthetic[names_le0] == 1.0 && tn.reference[names_le0] == ref.n_beta_le0   # median of 0, 1, 3
    @test occursin("range 0–3", tn.statistic[names_le0])
    t0 = check_table([fake_diag(n_beta_le0=n) for n in (0, 1, 3)], merge(ref, (n_beta_le0=0,)), v1)
    @test t0.reference[share] == 0.0 && t0.reference[names_le0] == 0.0                 # real has none: no
    # V9: spread within 0.01, corr within 0.05
    @test gates(check_table([fake_diag(α_spread=0.087) for _ in 1:3], ref, v1), "V9") == [false, true]
    @test gates(check_table([fake_diag(α_spread=0.085) for _ in 1:3], ref, v1), "V9") == [true, true]
    @test gates(check_table([fake_diag(α_corr=0.24) for _ in 1:3], ref, v1), "V9") == [true, false]
    @test_throws ArgumentError check_table(NamedTuple[], ref, v1)
end
