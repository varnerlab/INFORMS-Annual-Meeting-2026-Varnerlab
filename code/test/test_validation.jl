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
    # R30: lines mapping past the last trading day are dropped; non-EWLS lines ignored
    log2 = "2026-05-28 1 [EWLS] updated=33\n2026-05-28 2 [DECIDE] x\n2026-07-07 16:15:27 [EWLS] updated=33\n"
    @test ewls_calendar_from_log(write_tmp(log2), td; start=Date(2026,5,28), stop=Date(2026,7,6)) ==
        [Date(2026,5,28), Date(2026,7,6)]
    @test ewls_calendar_from_log(write_tmp(log2), td; start=Date(2026,5,28), stop=Date(2026,7,2)) ==
        [Date(2026,5,28)]
end
