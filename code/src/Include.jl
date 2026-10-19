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
using JuMP
using Clarabel
using SCS

include("Types.jl")
include("Factory.jl")
include("Data.jl")
include("SIM.jl")
include("Covariance.jl")
include("Allocators.jl")
include("AllocatorsMeanVariance.jl")
include("AllocatorsLongShort.jl")
include("Projection.jl")
include("Costs.jl")
include("Triggers.jl")
include("Metrics.jl")
include("Backtest.jl")

export MyMarketData, build, load_market_data, growth_rates, index_of, DATA_FILES,
    MySIMParameterEstimate, MyEWLSState, MyEWVarState, estimate_sim, fit_base_sim, ewls_init, ewls_update!,
    ewvar_init, ewvar_update!, market_state, preference_weights
export AbstractCovarianceModel, MySIMCovariance, MyShrunkResidualCovariance, growth_covariance,
    residual_matrix, shrinkage_loss, choose_shrinkage
export AbstractPortfolioRule, MyCobbDouglasRule, MyCESRule, MyStoneGearyRule, MyConstraintSet,
    MyAllocationProblem, MyAllocationResult, allocate, add_shared_constraints!
export MyMinVarRule, MyTangentRule, MyLongShortCDRule, max_investable, expected_growth
export MyProjection, project_gbm
export trade_cost, borrow_fee, MyMPCSpec, MyTriggerState, check_trigger
export scorecard
export MyRunConfig, MyBacktestResult, run_backtest
