# 2014–2024 calibration bundle for the screened utility-panel study.

"""
    MyCalibration

Baseline values fitted once on the training window: `base_sim` (per-ticker SIM fit, α in yr⁻¹),
`σ_m` (yr⁻¹, N−1 SD of SPY growth), `m_m` (yr⁻¹, mean SPY growth), `v_m` (yr⁻², population
variance of SPY growth), `λ` (dimensionless residual shrinkage), and the window `train_start`,
`train_stop`.
"""
mutable struct MyCalibration
    base_sim::Dict{String,MySIMParameterEstimate}
    σ_m::Float64
    m_m::Float64
    v_m::Float64
    λ::Float64
    train_start::Date
    train_stop::Date
    MyCalibration() = new()
end

"""
    build(::Type{MyCalibration}, data::NamedTuple)::MyCalibration

Build a `MyCalibration` from a NamedTuple holding every field; throws `ArgumentError` on a missing one.
"""
function build(::Type{MyCalibration}, data::NamedTuple)::MyCalibration
    model = MyCalibration()
    for f ∈ fieldnames(MyCalibration)
        haskey(data, f) || throw(ArgumentError("Cannot build MyCalibration: missing field $(f)"))
        setfield!(model, f, data[f])
    end
    return model
end

"""
    calibrate(md::MyMarketData; train_start::Date, train_stop::Date, seed::Int=5660,
        n_portfolios::Int=200)::MyCalibration

Fit the SIM, SPY growth moments (yr⁻¹, yr⁻²) and residual shrinkage λ on the dates in
[`train_start`, `train_stop`]. `seed` and `n_portfolios` (count) drive `choose_shrinkage`.
"""
function calibrate(md::MyMarketData; train_start::Date, train_stop::Date, seed::Int=5660,
    n_portfolios::Int=200)::MyCalibration
    fit = fit_base_sim(md, train_start, train_stop)
    λ = choose_shrinkage(md, train_start, train_stop; n_portfolios=n_portfolios, seed=seed)
    return build(MyCalibration, (base_sim=fit.params, σ_m=fit.σ_m, m_m=fit.m_m, v_m=fit.v_m, λ=λ,
        train_start=train_start, train_stop=train_stop))
end

"""
    save_calibration(path::AbstractString, c::MyCalibration)

Write `c` to a JLD2 file at `path` under the key `"calibration"`.
"""
function save_calibration(path::AbstractString, c::MyCalibration)
    jldsave(path; calibration=c)
    return nothing
end

"""
    load_calibration(path::AbstractString)::MyCalibration

Read the `"calibration"` entry of the JLD2 file at `path`.
"""
function load_calibration(path::AbstractString)::MyCalibration
    return load(path, "calibration")
end
