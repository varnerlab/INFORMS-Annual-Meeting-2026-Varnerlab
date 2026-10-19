using Test, Dates, DataFrames, CSV, JLD2
using VLINFORMS2026

# Write a tiny three-file dataset: AAA, GAP (missing 2025-01-06) and SPY.
# Timestamps carry different hours per ticker (AAA/GAP 04:00, SPY 05:00) so alignment must be by Date.
# Closes are distinct per day: AAA = 100 + dayofyear + 1000*(year-2014), SPY = AAA + 200.
function make_fixture(dir::String; dtb3::Bool=false)::String
    days = Dict(
        DATA_FILES[1] => [Date(2014,1,3), Date(2014,1,6), Date(2014,1,7)],
        DATA_FILES[2] => [Date(2025,1,2), Date(2025,1,3), Date(2025,1,6), Date(2025,1,7), Date(2025,1,8)],
        DATA_FILES[3] => [Date(2026,1,2), Date(2026,1,5), Date(2026,1,6)])
    for (file, ds) in days
        frame(ds, hour, bump) = DataFrame(
            open=fill(10.0, length(ds)),
            close=[100.0 + dayofyear(d) + 1000 * (year(d) - 2014) + bump for d in ds],
            volume=fill(1000.0, length(ds)),
            timestamp=DateTime.(ds) .+ Hour(hour))
        gap = filter(d -> d != Date(2025,1,6), ds)
        JLD2.save(joinpath(dir, file), "dataset",
                  Dict("AAA" => frame(ds, 4, 0.0), "GAP" => frame(gap, 4, 0.0), "SPY" => frame(ds, 5, 200.0)))
    end
    CSV.write(joinpath(dir, "sp500-sectors.csv"),
              DataFrame(Symbol=["AAA","GAP"], var"GICS Sector"=["Tech","Energy"]))
    if dtb3   # real FRED format: first observation after the first trading day, holiday as empty field
        write(joinpath(dir, "DTB3.csv"),
              "observation_date,DTB3\n2025-01-03,4.0\n2025-01-06,\n2025-01-07,5.0\n")
    end
    return dir
end

@testset "growth_rates" begin
    @test growth_rates([100.0, 101.0, 99.0]) ≈ 252 .* [log(1.01), log(99/101)]
end
@testset "universe excludes gappy ticker" begin
    md = @test_logs (:warn, r"DTB3") load_market_data(make_fixture(mktempdir()); start=Date(2025,1,2), stop=Date(2025,1,8))
    @test md.tickers == ["AAA"]
    @test !any(isnan, md.close) && all(md.g_f .== 0.04)
    @test md.sector == ["Tech"] && size(md.close) == (5, 1)
    @test index_of(md, Date(2025,1,3)) == 2
    @test_throws ArgumentError index_of(md, Date(2025,1,4))
end
@testset "dates align across tickers despite differing timestamp hours" begin
    md = load_market_data(make_fixture(mktempdir(); dtb3=true); start=Date(2025,1,2), stop=Date(2025,1,8))
    @test md.dates == [Date(2025,1,2), Date(2025,1,3), Date(2025,1,6), Date(2025,1,7), Date(2025,1,8)]
    @test md.close[:, 1] == 11000.0 .+ 100 .+ dayofyear.(md.dates)
    @test md.spy_close == md.close[:, 1] .+ 200.0
end
@testset "risk-free rate from DTB3" begin
    md = load_market_data(make_fixture(mktempdir(); dtb3=true); start=Date(2025,1,2), stop=Date(2025,1,8))
    # Jan 2 precedes the first observation (takes it); Jan 6 is empty (forward-filled from Jan 3)
    pct = [4.0, 4.0, 4.0, 5.0, 5.0]
    @test md.g_f == log.(1 .+ pct ./ 100)
end
@testset "real data" begin
    dir = joinpath(@__DIR__, "..", "..", "data")
    isfile(joinpath(dir, DATA_FILES[3])) || return
    md = load_market_data(dir; start=Date(2014,1,3), stop=Date(2026,9,4))
    @test md.dates[1] == Date(2014,1,3) && md.dates[end] == Date(2026,9,4)
    @test 360 ≤ length(md.tickers) ≤ 430 && "SPY" ∉ md.tickers
    @test Date(2025,7,4) ∉ md.dates && length(md.g_f) == length(md.dates)
    @test all(isfinite, md.open) && all(isfinite, md.close) && all(isfinite, md.volume)
    @test all(isfinite, md.spy_open) && all(isfinite, md.spy_close) && all(isfinite, md.g_f)
    @test issorted(md.dates; lt=<=)
end
