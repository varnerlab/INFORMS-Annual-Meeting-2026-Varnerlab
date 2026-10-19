"""
    MyMarketData

Daily end-of-day market data for the study universe. Matrices are days × tickers;
`dates` and the price vectors have one entry per SPY trading day.

Fields: `dates`, `tickers` (sorted, SPY excluded), `open`/`close` (USD/share),
`volume` (shares/day), `spy_open`/`spy_close` (USD/share), `sector` (GICS, one per ticker),
`g_f` (risk-free growth rate, yr⁻¹, one per day).
"""
mutable struct MyMarketData
    dates::Vector{Date}
    tickers::Vector{String}
    open::Matrix{Float64}
    close::Matrix{Float64}
    volume::Matrix{Float64}
    spy_open::Vector{Float64}
    spy_close::Vector{Float64}
    sector::Vector{String}
    g_f::Vector{Float64}
    MyMarketData() = new()
end
