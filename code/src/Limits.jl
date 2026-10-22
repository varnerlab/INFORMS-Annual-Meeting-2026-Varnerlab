"""
    limits_row(γ::AbstractMatrix{<:Real}, cols::AbstractVector{Int}; min_names::Int=5)::Int

The run row s (1 = the run start) of the one-day dial picture. `γ` is the run's γ matrix
(dimensionless, rows = run days, columns = tickers) and `cols` the ticker columns of the list.
It is 1 when at least `min_names` names in `cols` have γ > 0 on row 1; otherwise it is the first row
with the most names at γ > 0 **(plan decision)**.
"""
function limits_row(γ::AbstractMatrix{<:Real}, cols::AbstractVector{Int}; min_names::Int=5)::Int
    counts = [count(>(0), @view γ[s, cols]) for s in axes(γ, 1)]
    counts[1] >= min_names && return 1
    return argmax(counts)       # argmax returns the first maximum
end

"""
    limits_day(ctx::MyStudyContext, runs::Vector{MyStudyRun}, s::Int)::DataFrame

The budget-only allocation of every run on run row `s` (market-data row `ctx.rows[s]`), one DataFrame row
per run and ticker with columns `label, allocator, eta, date, ticker, gamma, price, weight, closed_form,
status`. Each run allocates a fresh book on day s **(plan decision)**: nothing held, equity
`settings.W0` (USD), on the sorted columns of `settings.fixed_tickers`. `gamma` is dimensionless, `price`
is USD/share, `weight` is the solver's wᵢ = nᵢSᵢ/W and `closed_form` is `budget_only_weights`
(both dimensionless fractions of equity); `eta` is the CES elasticity (dimensionless), 1.0 for CD
whatever `settings.ces_eta` says. Throws `ArgumentError` for an allocator other than CD, CES and CESw,
a screen other than `:fixed`, or settings whose `context_key` differs from the context's.
"""
function limits_day(ctx::MyStudyContext, runs::Vector{MyStudyRun}, s::Int)::DataFrame
    md = ctx.md
    out = DataFrame(label=String[], allocator=String[], eta=Float64[], date=Date[], ticker=String[],
        gamma=Float64[], price=Float64[], weight=Float64[], closed_form=Float64[], status=Symbol[])
    for run in runs
        st = run.settings
        run.allocator in ("CD", "CES", "CESw") ||
            throw(ArgumentError("limits_day: allocator $(repr(run.allocator)) is not CD, CES or CESw"))
        run.screen == :fixed || throw(ArgumentError("limits_day: screen :$(run.screen) is not :fixed"))
        isequal(context_key(st), context_key(ctx.settings)) ||
            throw(ArgumentError("limits_day: run settings do not match the context's"))
        cols = sort!([findfirst(==(tk), md.tickers) for tk in st.fixed_tickers])
        rule = make_rule(run.allocator, st)
        p = _study_problem(ctx, run, ctx.rows[s], cols, fill(false, length(cols)), zeros(length(md.tickers)),
            st.W0; basket=cols)
        res = allocate(rule, p)
        cf = budget_only_weights(rule, p.γ, p.prices; W=p.W, ε=p.constraints.ε)
        η = run.allocator == "CD" ? 1.0 : st.ces_eta
        for (i, j) in enumerate(cols)
            push!(out, (run.label, run.allocator, η, md.dates[ctx.rows[s]], md.tickers[j], p.γ[i], p.prices[i],
                res.w[i], cf[i], res.status))
        end
    end
    return out
end
