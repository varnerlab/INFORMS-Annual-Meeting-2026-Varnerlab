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
using SHA

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
include("Validation.jl")
include("Bandit.jl")
include("Experiments.jl")
include("MarketInputs.jl")
include("Calibration.jl")
include("OnlineSIM.jl")
include("HedgePool.jl")
include("Screens.jl")
include("StudySettings.jl")
include("Context.jl")
include("GrowthScorecard.jl")
include("StudyBacktest.jl")
include("StudyRunner.jl")
include("StudyTables.jl")

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
export compare_targets, ewls_calendar_from_log, logged_market_from_log
export assign_quotas, beat_spy_reward, train_sector_bandit, quarter_starts, bandit_basket_schedule,
    random_sector_basket, random_matched_schedule
export config_key, config_hash, run_and_save, run_grid, rebuild_summary, run_stats, load_experiment
export xi_gain, xi_series, gtilde_series, gamma_weights
export MyCalibration, calibrate, save_calibration, load_calibration
export MyOnlineSIM, online_sim
export down_day_beta, hedge_pool, beta_cap
export AbstractScreen, MyBanditScreen, MyGammaScreen, MyRandomScreen, excess_growth_windows,
    train_ticker_bandit, rank_arms, pick, short_candidates
export MyStudySettings, load_study, with_settings, context_key, MyStudyContext, build_context
export growth_scorecard
export MyStudyRun, MyStudyResult, make_rule, make_screen, run_study
export study_key, study_hash, study_stats, run_and_save_study, run_study_grid, tier1_runs, fixed_schedule_runs,
    rebuild_study_summary
export group_summary, paired_win_rate, gamma_percentile, hedge_effect, long_vs_short_hedge, passive_references
