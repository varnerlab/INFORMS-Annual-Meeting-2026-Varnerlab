using Test, Dates, VLINFORMS2026
include("helpers.jl")

@testset "market path stats" begin
    gen = toy_generator(); st = market_path_stats(gen, 1:5; window=(gen.dates[1001], gen.dates[1400]), train=(gen.dates[1], gen.dates[1000]))
    md = synthetic_market(gen, 3).md; i0, i1 = 1001, 1400
    S = md.spy_close[i0:i1]
    @test st.window_max_drawdown[3] ≈ maximum(1 .- S ./ accumulate(max, S))
    @test st.train_growth[3] ≈ log(md.spy_close[1000] / md.spy_close[1]) / (999 / 252)
    @test !any(st.has_burst)                                     # toy market is untuned: ϵ = 0
end
@testset "path sets" begin
    df = DataFrame(path_id=1:1000, window_max_drawdown=rand(Xoshiro(2), 1000), has_burst=(1:1000) .% 10 .== 0)
    ps = path_sets(df; n_burst=5, n_bear=5, n_typical=10)
    b = df[df.has_burst, :]; @test sort(ps.path_id[ps.burst_stress]) == sort(b.path_id[sortperm(b.window_max_drawdown; rev=true)[1:5]])
    @test count(ps.bear_stress) == 5 && count(ps.typical) == 10 && !any(ps.bear_stress .& ps.has_burst)
    ro = collect(skipmissing(ps.run_order)); @test sort(ro) == 1:length(ro)
    first4 = ps[coalesce.(ps.run_order, 10^9) .<= 4, :]
    @test any(first4.burst_stress) && any(first4.bear_stress) && any(first4.typical)
    @test_throws ArgumentError path_sets(df; n_burst=500)
end
