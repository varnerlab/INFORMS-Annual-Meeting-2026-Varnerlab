using Test, Dates, DataFrames, CSV, JLD2
using VLINFORMS2026

# Write a tiny three-file dataset: AAA, GAP (missing 2025-01-06) and SPY.
function make_fixture(dir::String)::String
    days = Dict(
        DATA_FILES[1] => [Date(2014,1,3), Date(2014,1,6), Date(2014,1,7)],
        DATA_FILES[2] => [Date(2025,1,2), Date(2025,1,3), Date(2025,1,6), Date(2025,1,7), Date(2025,1,8)],
        DATA_FILES[3] => [Date(2026,1,2), Date(2026,1,5), Date(2026,1,6)])
    for (file, ds) in days
        frame(ds) = DataFrame(open=fill(10.0, length(ds)), close=fill(11.0, length(ds)),
                              volume=fill(1000.0, length(ds)), timestamp=DateTime.(ds))
        gap = filter(d -> d != Date(2025,1,6), ds)
        JLD2.save(joinpath(dir, file), "dataset",
                  Dict("AAA" => frame(ds), "GAP" => frame(gap), "SPY" => frame(ds)))
    end
    CSV.write(joinpath(dir, "sp500-sectors.csv"),
              DataFrame(Symbol=["AAA","GAP"], var"GICS Sector"=["Tech","Energy"]))
    return dir
end

@testset "growth_rates" begin
    @test growth_rates([100.0, 101.0, 99.0]) ≈ 252 .* [log(1.01), log(99/101)]
end
@testset "universe excludes gappy ticker" begin
    md = load_market_data(make_fixture(mktempdir()); start=Date(2025,1,2), stop=Date(2025,1,8))
    @test md.tickers == ["AAA"]
    @test !any(isnan, md.close) && all(md.g_f .== 0.04)
    @test md.sector == ["Tech"] && size(md.close) == (5, 1)
    @test index_of(md, Date(2025,1,3)) == 2
    @test_throws ArgumentError index_of(md, Date(2025,1,4))
end
@testset "real data" begin
    dir = joinpath(@__DIR__, "..", "..", "data")
    isfile(joinpath(dir, DATA_FILES[3])) || return
    md = load_market_data(dir; start=Date(2014,1,3), stop=Date(2026,9,4))
    @test md.dates[1] == Date(2014,1,3) && md.dates[end] == Date(2026,9,4)
    @test 360 ≤ length(md.tickers) ≤ 430 && "SPY" ∉ md.tickers
    @test Date(2025,7,4) ∉ md.dates && length(md.g_f) == length(md.dates)
end
