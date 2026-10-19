using Dates, Random, LinearAlgebra
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

# Allocation problem for the allocator tests (Tasks 4 and 5). Defaults: 3 assets, γ = [0.2, 0.5, 0.3],
# prices [10, 20, 40] USD/share, W = 1000 USD, μ_g = 0.1 yr⁻¹, Σ_g = 4I yr⁻², g_f = 0, and every
# limit slack (w_max = s_max = 1, σ_g_max = 1e6 yr⁻¹, c̄ = 0, n_prev = 0). Per-asset defaults are
# sized from length(γ), so overriding γ changes the asset count; each name is its own sector.
function P(; γ=[0.2, 0.5, 0.3], prices=[10.0, 20.0, 40.0], W=1000.0, μ_g=nothing, Σ_g=nothing,
        g_f=0.0, w_max=1.0, s_max=1.0, σ_g_max=1e6, K_turnover=0.10 * W, c̄=0.0, sector=nothing,
        n_prev=nothing, ε=1e-3)
    K = length(γ)
    cs = build(MyConstraintSet, (w_max=w_max, s_max=s_max, σ_g_max=σ_g_max, K_turnover=K_turnover,
        c̄=c̄, sector=something(sector, ["S$(i)" for i in 1:K]), n_prev=something(n_prev, zeros(K)), ε=ε))
    return build(MyAllocationProblem, (γ=γ, μ_g=something(μ_g, fill(0.1, K)),
        Σ_g=something(Σ_g, 4.0 .* Matrix(I, K, K)), prices=prices, W=W, g_f=g_f, constraints=cs))
end

# 5-asset version of P (prices 10..50 USD/share).
P5(; γ=[0.3, 0.25, 0.2, 0.15, 0.1], prices=[10.0, 20.0, 30.0, 40.0, 50.0], kw...) = P(; γ=γ, prices=prices, kw...)

# Engine fixture (scripts/dev/make_engine_fixture.jl) → package problem. The engine's Σ is the
# annualized covariance Δt·Σ_g and its σ_max is annualized, so Σ_g = Σ/Δt and σ_g_max = σ_max/√Δt.
function problem_from_engine(fx::AbstractDict; Δt::Float64=1/252)
    K = length(fx["γ"])
    cs = build(MyConstraintSet, (w_max=fx["w_max"], s_max=1.0, σ_g_max=fx["σ_max"] / sqrt(Δt),
        K_turnover=fx["K_turnover"], c̄=fx["c̄"], sector=["E$(i)" for i in 1:K], n_prev=fx["n_prev"], ε=1e-3))
    return build(MyAllocationProblem, (γ=fx["γ"], μ_g=zeros(K), Σ_g=fx["Σ"] ./ Δt, prices=fx["p"],
        W=fx["B"], g_f=0.0, constraints=cs))
end
