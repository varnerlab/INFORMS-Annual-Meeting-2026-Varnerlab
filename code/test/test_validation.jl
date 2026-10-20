using Test
using Dates
using VLINFORMS2026

write_tmp(s) = (p = tempname(); write(p, s); p)

@testset "compare_targets" begin
    a = Dict("A" => 0.1, "B" => 0.2)
    r = compare_targets(a, copy(a))
    @test r.pass && r.max_abs_diff == 0.0 && isempty(r.only_sim) && isempty(r.only_trial)
    r2 = compare_targets(a, Dict("A" => 0.12, "B" => 0.2))
    @test !r2.pass
    @test r2.max_abs_diff ≈ 0.02
    @test r2.worst == "A"
    r3 = compare_targets(Dict("A" => 0.1, "B" => 0.2, "C" => 0.05), a)
    @test r3.only_sim == ["C"] && !r3.pass
    # names with w ≤ 1e-3 do not count toward the name sets
    r4 = compare_targets(Dict("A" => 0.1, "Z" => 5e-4), Dict("A" => 0.1))
    @test r4.pass && isempty(r4.only_sim)
    r5 = compare_targets(Dict("A" => 0.1), Dict("A" => 0.1, "T" => 0.3))
    @test r5.only_trial == ["T"] && !r5.pass
end

@testset "ewls_calendar_from_log" begin
    log = """
    2026-05-28 16:15:25 [EWLS] updated=33  skipped=0
    2026-05-29 16:15:30 [EWLS] updated=33  skipped=0
    2026-05-29 18:50:20 [EWLS] updated=33  skipped=0
    2026-07-02 16:15:27 [EWLS] updated=33  skipped=0
    2026-07-03 16:15:27 [EWLS] updated=33  skipped=0
    2026-07-06 16:15:27 [EWLS] updated=33  skipped=0
    """
    td = Date.(["2026-05-28","2026-05-29","2026-06-01","2026-07-01","2026-07-02","2026-07-06"])
    @test ewls_calendar_from_log(write_tmp(log), td) ==
        Date.(["2026-05-28","2026-05-29","2026-05-29","2026-07-02","2026-07-02","2026-07-06"])
    # R32: the real skip line is not an update; other log lines are ignored
    log3 = """
    2026-05-29 16:15:21 [MARKET] gm_t=0.5537  λ_t=-0.7376  spy_close=756.34
    2026-05-29 16:15:30 [EWLS] updated=33  skipped=0
    2026-05-29 18:50:19 [EWLS] skipped (skip_ewls_update): this date's update was already applied and persisted
    2026-07-02 16:15:27 [EWLS] updated=33  skipped=0
    """
    @test ewls_calendar_from_log(write_tmp(log3), td) == Date.(["2026-05-29", "2026-07-02"])
    # R30/R32: the window filters on the line's RAW date before mapping. The 07-07 line would map to
    # 07-06 (inside the window) under map-then-filter but lies after `stop`, so it is dropped; the
    # Saturday 07-04 line maps to 07-02.
    log2 = "2026-05-28 1 [EWLS] updated=33\n2026-05-28 2 [DECIDE] x\n2026-07-04 16:15:27 [EWLS] updated=33\n" *
        "2026-07-07 16:15:27 [EWLS] updated=33\n"
    @test ewls_calendar_from_log(write_tmp(log2), td; start=Date(2026,5,28), stop=Date(2026,7,6)) ==
        [Date(2026,5,28), Date(2026,7,2)]
    @test ewls_calendar_from_log(write_tmp(log2), td; start=Date(2026,5,28), stop=Date(2026,6,30)) ==
        [Date(2026,5,28)]
    # a line before `start` is dropped as well
    @test ewls_calendar_from_log(write_tmp(log2), td; start=Date(2026,6,1), stop=Date(2026,7,6)) ==
        [Date(2026,7,2)]
end

@testset "logged_market_from_log" begin
    log = """
    2026-05-28 16:15:20 [MARKET] gm_t=0.5  λ_t=-0.7  spy_close=1.0
    2026-05-28 16:15:25 [EWLS] updated=33  skipped=0
    2026-05-29 10:43:00 [MARKET] gm_t=1.5  λ_t=-0.6  spy_close=1.0
    2026-05-29 10:43:05 [EWLS] updated=33  skipped=0
    2026-05-29 16:15:00 [MARKET] gm_t=2.5  λ_t=-0.5  spy_close=1.0
    2026-05-29 16:15:05 [EWLS] updated=33  skipped=0
    2026-05-29 18:50:19 [EWLS] skipped (skip_ewls_update): already applied
    2026-07-02 16:15:27 [EWLS] updated=33  skipped=0
    2026-07-07 16:15:00 [MARKET] gm_t=9.0  λ_t=-0.1  spy_close=1.0
    2026-07-07 16:15:27 [EWLS] updated=33  skipped=0
    """
    td = Date.(["2026-05-28","2026-05-29","2026-07-02","2026-07-06"])
    r = logged_market_from_log(write_tmp(log), td; start=Date(2026,5,28), stop=Date(2026,7,6))
    @test r.ewls_g_m[1:3] == [0.5, 1.5, 2.5]
    @test isnan(r.ewls_g_m[4]) && length(r.ewls_g_m) == 4
    @test r.decision[Date(2026,5,29)] == (g_m = 2.5, λ = -0.5)      # last line of the day
    @test !haskey(r.decision, Date(2026,7,7))                       # not a trading day
    @test length(r.ewls_g_m) == length(ewls_calendar_from_log(write_tmp(log), td; start=Date(2026,5,28), stop=Date(2026,7,6)))
end
