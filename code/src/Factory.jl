"""
    build(::Type{MyMarketData}, data::NamedTuple)::MyMarketData

Build a `MyMarketData` from a NamedTuple holding every field (units as in the type docstring).
"""
function build(::Type{MyMarketData}, data::NamedTuple)::MyMarketData
    model = MyMarketData()
    for f ∈ fieldnames(MyMarketData)
        haskey(data, f) || throw(ArgumentError("Cannot build MyMarketData: missing field $(f)"))
        setfield!(model, f, data[f])
    end
    return model
end
