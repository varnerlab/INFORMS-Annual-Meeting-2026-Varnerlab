using Test, Dates, VLINFORMS2026
include("helpers.jl")

# One hand-made run row of a path file. `w` is the ending wealth ÷ start (W0 = 100,000 USD). Growth `g` (yr⁻¹) and
# max drawdown `dd` (fraction) default to simple functions of `w`, and Sharpe is 2w, so each metric has its own values.
function syn_row(p::Int, screen::String, alloc::String, seed::Int, hedge::Bool, w::Float64;
        g::Float64=w - 1.0, dd::Float64=0.5 - 0.25 * w, K::Int=13)
    return (path_id=p, label="$(alloc)|$(screen)|hedge=$(hedge)|seed=$(seed)", status="ok", allocator=alloc,
        screen=screen, seed=seed, hedge=hedge, basket_K=K, W0=100_000.0, terminal=100_000.0 * w,
        realized_growth=g, max_drawdown=dd, sharpe=2.0 * w)
end

# The hand-made path rows: 3 paths × seeds 5660 and 5661 × CD and CES × random per sector and γ per sector, hedge off.
# Ending wealth ÷ start per (picker, utility), one (seed 5660, seed 5661) pair per path; NaN marks the seed-5661 run
# of path 3, random per sector with CES, which errors. As in script 22, its row keeps only the path, label and status.
# Hedge-on rows exist for random per sector with CD only: wealth 0.10 higher, growth moved by +0.01, −0.02, −0.03 yr⁻¹
# on paths 1–3, and drawdown 0.05 lower.
function syn_rows()::DataFrame
    W = Dict(("random", "CD") => [(0.95, 1.05), (1.10, 1.30), (1.30, 1.50)],
        ("random", "CES") => [(1.00, 1.00), (1.00, 1.20), (1.20, NaN)],
        ("gamma", "CD") => [(1.10, 1.10), (1.30, 1.30), (1.20, 1.20)],
        ("gamma", "CES") => [(1.05, 1.05), (1.25, 1.25), (1.35, 1.35)])
    seeds = [5660, 5661]
    rows = []
    for (key, pairs) in W
        for p in 1:3
            for k in 1:2
                w = pairs[p][k]
                if !isnan(w)
                    push!(rows, syn_row(p, key[1], key[2], seeds[k], false, w))
                end
            end
        end
    end
    shift = [0.01, -0.02, -0.03]
    for p in 1:3
        for k in 1:2
            w = W[("random", "CD")][p][k]
            push!(rows, syn_row(p, "random", "CD", seeds[k], true, w + 0.10; g=w - 1.0 + shift[p],
                dd=0.5 - 0.25 * w - 0.05))
        end
    end
    ok = DataFrame()
    for r in rows
        push!(ok, r)
    end
    err =DataFrame(path_id=[3], label=["CES|random|hedge=false|seed=5661"], status=["error"])
    return vcat(ok, err; cols=:union)
end

# The hand-made bench rows: SPY ends at 1.10, 1.10, 1.20 and the equal-weight universe at 1.05, 1.15, 1.25 on paths
# 1–3; SPY's max drawdown is 0.30, 0.15, 0.10.
function syn_bench()::DataFrame
    return DataFrame(path_id=[1, 1, 2, 2, 3, 3], name=repeat(["SPY buy-and-hold", "equal-weight universe"], 3),
        terminal=100_000.0 .* [1.10, 1.05, 1.10, 1.15, 1.20, 1.25], realized_growth=[0.06, 0.03, 0.06, 0.08, 0.11, 0.13],
        max_drawdown=[0.30, 0.20, 0.15, 0.20, 0.10, 0.20], sharpe=[0.5, 0.3, 0.5, 0.6, 0.9, 1.0])
end

@testset "synthetic tables" begin
    C = path_cells(syn_rows())
    B = bench_cells(syn_bench())
    # path_cells: the median over 2 seeds equals the hand value for path 2, random per sector, CD
    c = C[(C.path_id .== 2) .& (C.picker .== "random") .& (C.allocator .== "CD") .& .!C.hedge, :]
    @test nrow(c) == 1 && c.n_seeds == [2]
    @test c.wealth[1] ≈ 1.20 && c.realized_growth[1] ≈ 0.20 && c.max_drawdown[1] ≈ 0.20 && c.sharpe[1] ≈ 2.40
    @test nrow(C) == 15                     # 3 paths × 2 pickers × 2 utilities with the hedge off, 3 cells with it on
    @test nrow(B) == 6 && B.wealth ≈ [1.10, 1.05, 1.10, 1.15, 1.20, 1.25]
    @test B.name[1:2] == ["SPY buy-and-hold", "equal-weight universe"] && B.max_drawdown[5] == 0.10
    # grid_table(:median): reference rows first, then PICKER_ROWS; the CD cell is the median over paths of the
    # per-path medians
    G = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:median)
    @test names(G) == vcat("Ticker picker", TABLE_ALLOCATOR_ORDER) && nrow(G) == 16
    @test G[1:2, "Ticker picker"] == ["SPY buy-and-hold", "equal-weight universe"]
    @test G[3:16, "Ticker picker"] == [name for (key, name) in PICKER_ROWS]
    @test all(collect(G[1, 2:8]) .≈ 1.10) && all(collect(G[2, 2:8]) .≈ 1.15)   # same in every utility column
    @test G[3, "CD"] ≈ 1.20 && G[3, "CES"] ≈ 1.25   # γ per sector: medians of 1.10, 1.30, 1.20 and 1.05, 1.25, 1.35
    @test G[5, "CD"] ≈ 1.20 && G[5, "CES"] ≈ 1.10   # random per sector: of 1.00, 1.20, 1.40 and 1.00, 1.10, 1.20
    @test isnan(G[5, "SG"]) && isnan(G[4, "CD"])     # no runs
    @test grid_table(C, B, :wealth; ids=[1, 2], hedge=false, stat=:median)[5, "CD"] ≈ 1.10   # only `ids` count
    Gon = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=true, stat=:median)
    @test Gon[5, "CD"] ≈ 1.30 && isnan(Gon[3, "CD"]) && Gon[1, "CD"] ≈ 1.10   # hedge-on cells; same reference rows
    # grid_table(:beat_spy): 2/3 when 2 of the 3 paths beat their own SPY
    S = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:beat_spy)
    @test S[5, "CD"] ≈ 2 / 3                # random per sector, CD: 1.00 < 1.10, 1.20 > 1.10, 1.40 > 1.20
    @test S[3, "CD"] ≈ 1 / 3                # γ per sector, CD ties SPY on paths 1 and 3; a tie does not beat it
    @test isnan(S[1, "CD"]) && S[2, "LS"] ≈ 2 / 3   # SPY is not compared with itself; equal weight 1.05, 1.15, 1.25
    @test isnan(S[5, "SG"])
    Sd = grid_table(C, B, :max_drawdown; ids=[1, 2, 3], hedge=false, stat=:beat_spy)
    @test Sd[5, "CD"] ≈ 1 / 3               # a lower drawdown beats SPY: 0.25 < 0.30, 0.20 > 0.15, 0.15 > 0.10
    # grid_table(:band): "median (p10 to p90)" with 3 decimals
    Bd = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:band)
    @test Bd[5, "CD"] == "1.200 (1.040 to 1.360)"   # random per sector, CD: 1.00, 1.20, 1.40
    @test Bd[1, "CD"] == "1.100 (1.100 to 1.180)"   # SPY: 1.10, 1.10, 1.20
    @test Bd[5, "SG"] == "—"
    @test grid_table(C, B, :realized_growth; ids=[1, 2, 3], hedge=false, stat=:band)[5, "CD"] == "0.200 (0.040 to 0.360)"
    # band_table: the same bands as numbers, one row per grid row and utility
    BT = band_table(C, B, :wealth; ids=[1, 2, 3], hedge=false)
    @test names(BT) == ["Ticker picker", "utility", "n_paths", "median", "p10", "p90"] && nrow(BT) == 16 * 7
    r = BT[(BT[!, "Ticker picker"] .== "Random per sector (33)") .& (BT.utility .== "CD"), :]
    @test r.n_paths == [3] && r.median[1] ≈ 1.20 && r.p10[1] ≈ 1.04 && r.p90[1] ≈ 1.36
    @test BT[1, "Ticker picker"] == "SPY buy-and-hold" && BT.p90[1] ≈ 1.18 && BT.utility[1:7] == TABLE_ALLOCATOR_ORDER
    sg = BT[(BT[!, "Ticker picker"] .== "Random per sector (33)") .& (BT.utility .== "SG"), :]
    @test sg.n_paths == [0] && isnan(sg.median[1])
    # an error row is excluded and n_seeds drops to 1 for that cell
    e = C[(C.path_id .== 3) .& (C.picker .== "random") .& (C.allocator .== "CES") .& .!C.hedge, :]
    @test e.n_seeds == [1] && e.wealth[1] ≈ 1.20
    @test_throws ArgumentError grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:mean)
    @test_throws ArgumentError grid_table(C, B, :wealth; ids=[1, 4], hedge=false, stat=:median)   # path 4: no SPY row
end

@testset "synthetic picker keys" begin
    @test length(PICKER_ROWS) == 14 && length(unique([key for (key, name) in PICKER_ROWS])) == 14
    @test PICKER_ROWS[1] == ("gamma", "γ per sector (33)") && PICKER_ROWS[14] == ("fixed:l8b_15", "User's 13 + WMT, PG (15)")
    @test picker_key((screen="gamma", basket_K=13, label="CD|gamma|hedge=false|seed=5660")) == "gamma"
    @test picker_key((screen="random", basket_K=13, label="CD|random|hedge=false|seed=5663")) == "random"
    @test picker_key((screen="random_global", basket_K=34, label="LS|random_global|K=34|hedge=true|seed=5662")) ==
          "random_global|K=34"
    @test picker_key((screen="fixed", basket_K=15, label="CD|fixed:l8b_15|hedge=true|seed=5660")) == "fixed:l8b_15"
    @test_throws ArgumentError picker_key((screen="fixed", basket_K=13, label="CD|fixed|hedge=true|seed=5660"))
end

@testset "synthetic head-to-heads and real percentiles" begin
    C = path_cells(syn_rows())
    B = bench_cells(syn_bench())
    # head_to_head: (4 picker pairs × 2 hedge settings + 14 pickers × 2 metrics) × 7 utilities
    H = head_to_head(C; ids=[1, 2, 3])
    @test nrow(H) == 252
    r = H[(H.comparison .== "γ per sector (33) minus Random per sector (33)") .& (H.hedge .== "off") .&
          (H.allocator .== "CD"), :]
    @test r.metric == ["wealth"] && r.n_paths == [3]
    @test r.median[1] ≈ 0.10 && r.p10[1] ≈ -0.14 && r.p90[1] ≈ 0.10 && r.share_above_0[1] ≈ 2 / 3   # 0.10, 0.10, −0.20
    @test r.share_below_0[1] ≈ 1 / 3
    g = H[(H.comparison .== "Random per sector (33)") .& (H.metric .== "realized_growth") .& (H.allocator .== "CD"), :]
    @test g.hedge == ["on minus off"] && g.n_paths == [3]
    @test g.median[1] ≈ -0.02 && g.p10[1] ≈ -0.028 && g.p90[1] ≈ 0.004 && g.share_above_0[1] ≈ 1 / 3
    d = H[(H.comparison .== "Random per sector (33)") .& (H.metric .== "max_drawdown") .& (H.allocator .== "CD"), :]
    @test d.median[1] ≈ -0.05 && d.share_above_0[1] == 0.0 && d.share_below_0[1] == 1.0
    n = H[(H.comparison .== "γ per sector (33)") .& (H.metric .== "realized_growth") .& (H.allocator .== "CD"), :]
    @test n.n_paths == [0] && isnan(n.median[1])     # no hedge-on runs of γ per sector
    # head_to_head_grid: comparisons (or pickers, for hedge on minus off) as rows, utilities as columns; one grid of
    # bands and one of shares
    HG = head_to_head_grid(H; hedge="off", metric="wealth", stat=:band)
    @test names(HG) == vcat("Comparison", TABLE_ALLOCATOR_ORDER) && nrow(HG) == 4
    @test HG[1, "Comparison"] == "Global γ (5) minus Random, whole universe (5)" && HG[1, "CD"] == "—"
    @test HG[4, "CD"] == "0.100 (−0.140 to 0.100)"                        # negatives print with U+2212
    @test head_to_head_grid(H; hedge="off", metric="wealth", stat=:share_above_0)[4, "CD"] == "0.667"
    HD = head_to_head_grid(H; hedge="on minus off", metric="max_drawdown", stat=:share_below_0)
    @test names(HD)[1] == "Ticker picker" && nrow(HD) == 14 && HD[3, "CD"] == "1.000"   # random per sector
    @test head_to_head_grid(H; hedge="on minus off", metric="realized_growth", stat=:band)[3, "CD"] ==
          "−0.020 (−0.028 to 0.004)"
    @test_throws ArgumentError head_to_head_grid(H; hedge="off", metric="wealth", stat=:median)
    # format_grid: 3 decimals, and bold where the share of paths beating SPY is above 0.5
    G = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:median)
    S = grid_table(C, B, :wealth; ids=[1, 2, 3], hedge=false, stat=:beat_spy)
    F = format_grid(G; beat=S)
    @test F[5, "CD"] == "**1.200**" && F[3, "CD"] == "1.200" && F[2, "CD"] == "**1.150**" && F[1, "CD"] == "1.100"
    @test F[5, "SG"] == "—" && format_grid(S)[5, "CD"] == "0.667"
    @test format_grid(S)[1, "CD"] == "—"                 # the SPY cell of the beat-SPY table
    # real_percentiles: the share of paths at or below the real cell. Real γ per sector, CD ends at 1.20; real random
    # per sector, CD at the median of 1.05 and 1.15; real SPY at 1.15 and equal weight at 1.20.
    real_rows = DataFrame([syn_row(0, "gamma", "CD", 5660, false, 1.20), syn_row(0, "random", "CD", 5660, false, 1.05),
        syn_row(0, "random", "CD", 5661, false, 1.15)])
    RC = path_cells(real_rows)
    RB = bench_cells(DataFrame(path_id=[0, 0], name=["SPY buy-and-hold", "equal-weight universe"],
        terminal=[115_000.0, 120_000.0], realized_growth=[0.08, 0.11], max_drawdown=[0.19, 0.18], sharpe=[0.9, 1.0]))
    P = real_percentiles(C, RC; ids=[1, 2, 3], hedge=false, bench=B, real_bench=RB)
    @test names(P) == names(G) && P[!, "Ticker picker"] == G[!, "Ticker picker"]
    @test P[3, "CD"] ≈ 2 / 3     # γ per sector: 1.10 and the tie at 1.20 count, 1.30 does not
    @test P[5, "CD"] ≈ 1 / 3     # random per sector: 1.00 at or below 1.10; 1.20 and 1.40 above
    @test P[1, "CD"] ≈ 2 / 3 && P[2, "LS"] ≈ 2 / 3   # SPY 1.10, 1.10, 1.20 vs 1.15; equal weight 1.05, 1.15, 1.25 vs 1.20
    @test isnan(P[5, "CES"]) && isnan(P[4, "CD"])    # no real cell; no synthetic cell
end
