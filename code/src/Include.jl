# Single place for package imports and source includes.
using Dates
using Statistics
using LinearAlgebra
using Random
using TOML
using Downloads
using DataFrames
using CSV
using JLD2

include("Types.jl")
include("Factory.jl")
include("Data.jl")
include("SIM.jl")
include("Covariance.jl")

export MyMarketData, build, load_market_data, growth_rates, index_of, DATA_FILES,
    MySIMParameterEstimate, MyEWLSState, MyEWVarState, estimate_sim, fit_base_sim, ewls_init, ewls_update!,
    ewvar_init, ewvar_update!, market_state, preference_weights
export AbstractCovarianceModel, MySIMCovariance, MyShrunkResidualCovariance, growth_covariance,
    residual_matrix, shrinkage_loss, choose_shrinkage
