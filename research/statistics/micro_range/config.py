#Author: Marlon Dominguez
#Date  : 08/13/2026

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


INITIAL_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (

    #VOLATILITY AND CURRENT CANDLE
    "atr",
    "candle_range_atr",
    "body_size_atr",
    "body_to_range_ratio",
    "candle_direction",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "wick_imbalance",

    #SHORT-TERM TREND: 5 BARS
    #quick directional condition
    "trend_signed_efficiency_5",
    "trendline_move_robust_atr_5",
    "trend_linear_r2_5",
    "trend_residual_std_atr_5",
    "trend_smoothness_5",
    "trend_structure_balance_5",
    "trend_close_sma_distance_atr_5",
    "trend_signed_strength_raw_5",
    "trend_abs_strength_raw_5",
    "trend_quality_raw_5",
    "trend_primary_direction_5",
    "trend_start_pressure_5",
    "trend_continuation_pressure_5",
    "trend_decay_pressure_5",
    "trend_end_pressure_5",
    "trend_to_range_transition_pressure_5",

    #MEDIUM-TERM TREND: 20 BARS
    #primary local regime
    "trend_signed_efficiency_20",
    "trendline_move_robust_atr_20",
    "trend_linear_r2_20",
    "trend_residual_std_atr_20",
    "trend_smoothness_20",
    "trend_structure_balance_20",
    "trend_close_sma_distance_atr_20",
    "trend_signed_strength_raw_20",
    "trend_abs_strength_raw_20",
    "trend_quality_raw_20",
    "trend_primary_direction_20",
    "trend_start_pressure_20",
    "trend_continuation_pressure_20",
    "trend_decay_pressure_20",
    "trend_end_pressure_20",
    "trend_to_range_transition_pressure_20",


    #LONG-TERM TREND: 100 BARS
    
    #larger directional regime
    "trend_signed_efficiency_100",
    "trendline_move_robust_atr_100",
    "trend_linear_r2_100",
    "trend_residual_std_atr_100",
    "trend_smoothness_100",
    "trend_structure_balance_100",
    "trend_close_sma_distance_atr_100",
    "trend_signed_strength_raw_100",
    "trend_abs_strength_raw_100",
    "trend_quality_raw_100",
    "trend_primary_direction_100",
    "trend_continuation_pressure_100",
    "trend_decay_pressure_100",
    "trend_to_range_transition_pressure_100",


    #MULTI-TIMEFRAME TREND RELATIONSHIPS
    "trend_abs_strength_ratio_5_10",
    "trend_efficiency_ratio_5_10",
    "trend_quality_ratio_5_10",
    "trend_direction_agreement_5_10",
    "trend_direction_conflict_5_10",

    "trend_abs_strength_ratio_20_50",
    "trend_efficiency_ratio_20_50",
    "trend_quality_ratio_20_50",
    "trend_direction_agreement_20_50",
    "trend_direction_conflict_20_50",

    "trend_abs_strength_ratio_50_100",
    "trend_efficiency_ratio_50_100",
    "trend_quality_ratio_50_100",
    "trend_direction_agreement_50_100",
    "trend_direction_conflict_50_100",


    #LOCAL RANGE BEHAVIOR: 20 BARS
    
    #micro-range event
    "range_width_atr_20",
    "range_outlier_sensitivity_20",
    "position_in_range_20",
    "directional_efficiency_20",
    "boundary_activity_score_20",
    "touch_balance_20",
    "two_sided_touch_score_20",
    "mid_cross_frequency_20",
    "rotation_score_20",
    "flatness_score_20",
    "failed_break_frequency_20",
    "close_outside_frequency_20",
    "close_location_imbalance_20",
    "directional_body_pressure_20",
    "avg_wick_imbalance_20",
    "range_expansion_pressure_20",
    "range_compression_pressure_20",
    "one_sided_position_pressure_20",
    "atr_compression_ratio_20",
    "range_behavior_candidate_20",
    "range_candidate_persistence_20",
    "compression_persistence_20",
    "one_sided_pressure_persistence_20",


    #MEDIUM RANGE REGIME: 50 BARS
    "range_width_atr_50",
    "directional_efficiency_50",
    "boundary_activity_score_50",
    "touch_balance_50",
    "two_sided_touch_score_50",
    "mid_cross_frequency_50",
    "rotation_score_50",
    "flatness_score_50",
    "failed_break_frequency_50",
    "close_outside_frequency_50",
    "close_location_imbalance_50",
    "directional_body_pressure_50",
    "range_expansion_pressure_50",
    "range_compression_pressure_50",
    "one_sided_position_pressure_50",
    "atr_compression_ratio_50",
    "range_behavior_candidate_50",


    #LARGE RANGE REGIME: 100 BARS
    "range_width_atr_100",
    "directional_efficiency_100",
    "boundary_activity_score_100",
    "touch_balance_100",
    "two_sided_touch_score_100",
    "mid_cross_frequency_100",
    "rotation_score_100",
    "flatness_score_100",
    "failed_break_frequency_100",
    "close_outside_frequency_100",
    "one_sided_position_pressure_100",
    "atr_compression_ratio_100",
    "range_behavior_candidate_100",


    #MULTI-TIMEFRAME RANGE AGREEMENT

    "position_alignment_20_50",
    "range_component_agreement_20_50",
    "range_candidate_agreement_20_50",
    "range_agreement_20_50",

    "position_alignment_50_100",
    "range_component_agreement_50_100",
    "range_candidate_agreement_50_100",
    "range_agreement_50_100",


    #VOLUME AND ORDER-FLOW CONTEXT
    "total_volume",
    "volume_imbalance",
    "volume_ratio_20",
    "volume_zscore_20",
    "volume_boundary_imbalance_20",
    "volume_ratio_50",
    "volume_zscore_50",
    "volume_boundary_imbalance_50",
    "volume_ratio_100",
    "volume_zscore_100",


    #TIME AND SESSION CONTEXT
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_asian_session",
    "is_london_session",
    "is_new_york_session",
    "is_london_ny_overlap",
    "is_rollover_window",
    "is_asian_london_transition",
    "is_london_ny_transition",
)


CONTEXT_CHANGE_COLUMNS: tuple[str, ...] = (
    #short-term trend changes
    "trend_signed_efficiency_5",
    "trendline_move_robust_atr_5",
    "trend_linear_r2_5",
    "trend_structure_balance_5",
    "trend_signed_strength_raw_5",
    "trend_abs_strength_raw_5",
    "trend_quality_raw_5",
    "trend_start_pressure_5",
    "trend_continuation_pressure_5",
    "trend_decay_pressure_5",
    "trend_end_pressure_5",
    "trend_to_range_transition_pressure_5",

    #medium-term trend changes
    "trend_signed_efficiency_20",
    "trend_signed_strength_raw_20",
    "trend_quality_raw_20",
    "trend_continuation_pressure_20",
    "trend_decay_pressure_20",

    #local range-behavior changes
    "range_width_atr_20",
    "directional_efficiency_20",
    "boundary_activity_score_20",
    "touch_balance_20",
    "two_sided_touch_score_20",
    "mid_cross_frequency_20",
    "rotation_score_20",
    "flatness_score_20",
    "failed_break_frequency_20",
    "close_outside_frequency_20",
    "close_location_imbalance_20",
    "directional_body_pressure_20",
    "range_expansion_pressure_20",
    "range_compression_pressure_20",
    "one_sided_position_pressure_20",
    "atr_compression_ratio_20",
    "range_behavior_candidate_20",
    "range_candidate_persistence_20",

    #volume changes
    "volume_imbalance",
    "volume_ratio_20",
    "volume_zscore_20",
    "volume_boundary_imbalance_20")



EVENT_RELATIVE_FEATURE_COLUMNS: tuple[str, ...] = (
"range_width",
"range_width_atr",
"range_age_bars",
"bars_since_confirmation",
"bars_since_first_tradable",
"bars_since_invalidation",
"close_position_in_confirmed_range",
"close_distance_to_upper_atr",
"close_distance_to_lower_atr",
"high_distance_to_upper_atr",
"low_distance_to_lower_atr",
"event_type",
"boundary_side",
"range_is_active",
"range_was_invalidated",
"upper_touch_count",
"lower_touch_count",
"upper_break_attempt_count",
"lower_break_attempt_count",
"upper_outside_close_count",
"lower_outside_close_count",
"upper_wick_break_count",
"lower_wick_break_count",
"upper_reentry_count",
"lower_reentry_count",
"upper_outside_run",
"lower_outside_run",
"prior_outside_side",
"pending_breakout_side",
"bars_since_breakout",
)


@dataclass(frozen=True)
class ContextSnapshotConfig:

    feature_columns: tuple[str, ...] = (INITIAL_CONTEXT_FEATURE_COLUMNS)
    
    change_columns: tuple[str, ...] = CONTEXT_CHANGE_COLUMNS
        
    event_relative_columns: tuple[str, ...] = (EVENT_RELATIVE_FEATURE_COLUMNS)
    
    include_confirmation: bool = True
    include_decision: bool = True
    include_numeric_changes: bool = True
    
    strict_missing_columns: bool = True

    forbidden_name_parts: tuple[str, ...] = ("_segment_",)
    
    
    

@dataclass(frozen=True)
class ColumnConfig:
    timestamp: str          = "timestamp"
    open: str               = "open"
    high: str               = "high"
    low: str                = "low"
    close: str              = "close"
    atr: str                = "atr"
    confirmed_now: str      = "micro_range_confirmed_now"
    first_tradable_now: str = "micro_range_first_tradable_now"
    active_live: str        = "micro_range_active_live"
    invalidated_now: str    = "micro_range_invalidated_now"
    confirmed_upper: str    = "micro_range_confirmed_upper"
    confirmed_lower: str    = "micro_range_confirmed_lower"


@dataclass(frozen=True)
class EventConfig:
    post_invalidation_bars: int = 20
    max_bars_from_confirmation: int = 100
    boundary_tolerance_atr: float = 0.05
    minimum_break_atr: float = 0.0
    consecutive_outside_closes: tuple[int, ...] = (2, 3)
    retest_tolerance_atr: float = 0.10


@dataclass(frozen=True)
class OutcomeConfig:
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    excursion_horizon: int = 20
    r_levels: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    range_width_levels: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    risk_atr: float = 1.0
    intrabar_policy: Literal["conservative", "optimistic", "unresolved"] = "conservative"


@dataclass(frozen=True)
class CostConfig:
    spread_price: float = 0.0
    slippage_price: float = 0.0
    commission_price_round_trip: float = 0.0
    stress_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)

    @property
    def round_trip_price(self) -> float:
        
        return self.spread_price + 2.0 * self.slippage_price + self.commission_price_round_trip


@dataclass(frozen=True)
class MicroRangeStatConfig:
    
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    events: EventConfig = field(default_factory=EventConfig)
    outcomes: OutcomeConfig = field(default_factory=OutcomeConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    snapshots: ContextSnapshotConfig = field(default_factory=ContextSnapshotConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "MicroRangeStatConfig":
        
        try:
            
            import yaml
        
        except ImportError as exc:
            
            raise RuntimeError("[ERROR] READING YAML CONFIG REQUIRES PYYAML - INSTALL THE PROJECT DEPENDENCIES FIRST") from exc
        
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        
        snapshot_values = _tuples(raw.get("snapshots", {}),
            {
                "feature_columns",
                "change_columns",
                "event_relative_columns",
                "forbidden_name_parts",
            },
        )
        
        return cls(columns=ColumnConfig(**raw.get("columns", {})),
            	   events=EventConfig(**_tuples(raw.get("events", {}), {"consecutive_outside_closes"})),
            	   outcomes=OutcomeConfig(**_tuples(raw.get("outcomes", {}), {"horizons", "r_levels", "range_width_levels"})),
                   costs=CostConfig(**_tuples(raw.get("costs", {}), {"stress_multipliers"})),
                   snapshots=ContextSnapshotConfig(**snapshot_values))


    def as_dict(self) -> dict[str, Any]:
        
        return asdict(self)


def _tuples(values: dict[str, Any], names: set[str]) -> dict[str, Any]:
    
    result = dict(values)
    
    for name in names:
        
        if name in result:
            
            result[name] = tuple(result[name])
    
    return result
