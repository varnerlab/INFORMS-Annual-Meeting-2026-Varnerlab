using Dates, Random
using VLINFORMS2026

# Synthetic market: 2014–2019 weekdays, 4 sectors × 12 names, β = 1, within-sector residual
# correlation ρ, SPY as the market factor. Growth units yr⁻¹: market SD 2.5, residual SD 4.
function synthetic_md(; ρ::Float64, seed::Int=5660)
    rng = MersenneTwister(seed)
    dates = filter(d -> dayofweek(d) <= 5, collect(Date(2014,1,1):Day(1):Date(2019,12,31)))
    n = length(dates); Δt = 1/252
    sectors = ["S$(s)" for s in 1:4 for _ in 1:12]
    gm = 2.5 .* randn(rng, n - 1)
    z = randn(rng, n - 1, 4); u = randn(rng, n - 1, 48)
    E = 4.0 .* [sqrt(ρ) * z[t, parse(Int, sectors[i][2:end])] + sqrt(1 - ρ) * u[t, i] for t in 1:n-1, i in 1:48]
    g = gm .+ E
    cl = 100 .* exp.(vcat(zeros(1, 48), cumsum(g .* Δt; dims=1)))
    spy = 100 .* exp.(vcat(0.0, cumsum(gm .* Δt)))
    return build(MyMarketData, (dates=dates, tickers=["T$(lpad(i,2,'0'))" for i in 1:48], open=cl, close=cl,
        volume=ones(n, 48), spy_open=spy, spy_close=spy, sector=sectors, g_f=zeros(n)))
end

# Double every price (tickers and SPY) strictly after date d; used as a look-ahead canary.
function perturb_after(md::MyMarketData, d::Date)
    k = md.dates .> d
    f = ifelse.(k, 2.0, 1.0)
    return build(MyMarketData, (dates=md.dates, tickers=md.tickers, open=md.open .* f, close=md.close .* f,
        volume=md.volume, spy_open=md.spy_open .* f, spy_close=md.spy_close .* f, sector=md.sector, g_f=md.g_f))
end
