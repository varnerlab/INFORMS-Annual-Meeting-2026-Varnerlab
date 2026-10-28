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
import JumpHMM

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
include("Limits.jl")
include("StudyTables.jl")
include("SyntheticMarket.jl")
include("SyntheticRunner.jl")
include("SyntheticCalibration.jl")
include("SyntheticChecks.jl")
include("SyntheticTables.jl")
include("TalkData.jl")

export MyMarketData, build, load_market_data, growth_rates, index_of, DATA_FILES, boundary_split_factor, split_adjustments,
    MySIMParameterEstimate, MyEWLSState, MyEWVarState, estimate_sim, fit_base_sim, ewls_init, ewls_update!,
    ewvar_init, ewvar_update!, market_state, preference_weights
export AbstractCovarianceModel, MySIMCovariance, MyShrunkResidualCovariance, growth_covariance,
    residual_matrix, shrinkage_loss, choose_shrinkage
export AbstractPortfolioRule, MyCobbDouglasRule, MyCESRule, MyCESWeightRule, MyStoneGearyRule,
    MyConstraintSet, MyAllocationProblem, MyAllocationResult, allocate, add_shared_constraints!, budget_only_weights
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
export MyCalibration, calibrate, save_calibration, load_calibration, calibration_summary
export MyOnlineSIM, online_sim
export down_day_beta, hedge_pool, beta_cap
export AbstractScreen, MyBanditScreen, MyGammaScreen, MyRandomScreen, excess_growth_windows,
    train_ticker_bandit, rank_arms, pick, short_candidates,
    MyGlobalGammaScreen, MyLiquidGammaScreen, MyCorrGammaScreen, MyGlobalRandomScreen,
    MyLiquidRandomScreen, MyFixedScreen, liquid_shortlist
export MyStudySettings, load_study, with_settings, context_key, MyStudyContext, build_context
export growth_scorecard
export MyStudyRun, MyStudyResult, make_rule, make_screen, run_study
export study_key, regime_stats, study_key_string, study_hash, data_fingerprint, study_stats, run_and_save_study, run_study_grid, tier1_runs, fixed_schedule_runs, eta_runs, limits_runs,
    universe_runs, fixed_list_runs,
    rebuild_study_summary
export limits_row, limits_day, limits_failures
export group_summary, SUMMARY_METRICS, paired_win_rate, gamma_percentile, hedge_effect, long_vs_short_hedge,
    passive_references, markdown_table, compare_summaries, sort_table, TABLE_ALLOCATOR_ORDER, TABLE_SCREEN_ORDER,
    eta_frame, ETA_METRICS, REGIME_METRICS, eta_solver_table
export center_seed, model_long_run_mean
export path_seed, growth_window, ema_rate, drift_path, drift_variance, sim_residuals, sector_factors, loo_loadings, sector_loadings, SYNTH_FLOOR, SYNTH_R2_PRESERVE, compose_name, MySyntheticGenerator, with_generator, synthetic_draws, synthetic_growth, market_data_from_growth, synthetic_market
export SYNTH_HMM, SYNTH_TRAIN, SYNTH_ALPHA_WINDOWS, fit_name_model, model_sd_ratio, ewls_beta_path, beta_path_stats,
    real_drift_targets, bank_beta_stats, fit_beta_drift, alpha_parts, fit_alpha_drift
export market_path_stats, path_sets, synthetic_runs, run_synthetic_path, path_done, write_path_files
export ew_rebalance_gap, abs_acf, mean_pairwise_corr, real_reference, path_diagnostics, check_table,
    beta_le0_windows, alpha_correlation
export PICKER_ROWS, picker_key, path_cells, bench_cells, grid_table, band_table, format_grid, head_to_head,
    head_to_head_grid, real_percentiles
export TALK_REALRUN_ROWS, talk_realrun_grid, talk_row_ranges, talk_largek_shares, talk_numbers, write_tex_macros,
    talk_mpc_episode, talk_stylized_facts, talk_table_rows
