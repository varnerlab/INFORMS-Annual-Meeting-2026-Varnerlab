# Copy the course price/sector files into data/ and download FRED DTB3.
using Downloads

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const SOURCE_DIR = expanduser(get(ENV, "CHEME5660_DATA",
    "~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026/code/src/data"))
const FILES = [
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2",
    "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2",
    "SP500-Daily-OHLC-1-2-2026-to-09-04-2026.jld2",
    "sp500-sectors.csv",
]
const DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

function main()
    mkpath(DATA_DIR)
    for f in FILES
        cp(joinpath(SOURCE_DIR, f), joinpath(DATA_DIR, f); force=true)
        println("copied ", f)
    end
    try
        Downloads.download(DTB3_URL, joinpath(DATA_DIR, "DTB3.csv"))
        println("downloaded DTB3.csv")
    catch err
        @warn "DTB3 download failed; the loader will use a constant risk-free rate" exception=err
    end
end

main()
