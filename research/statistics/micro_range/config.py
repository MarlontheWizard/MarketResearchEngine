#Author: Marlon Dominguez
#Date  : 08/13/2026

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase1Config":
        
        try:
            
            import yaml
        
        except ImportError as exc:
            
            raise RuntimeError("[ERROR] READING YAML CONFIG REQUIRES PYYAML - INSTALL THE PROJECT DEPENDENCIES FIRST") from exc
        
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        
        return cls(columns=ColumnConfig(**raw.get("columns", {})),
            	   events=EventConfig(**_tuples(raw.get("events", {}), {"consecutive_outside_closes"})),
            	   outcomes=OutcomeConfig(**_tuples(raw.get("outcomes", {}), {"horizons", "r_levels", "range_width_levels"})),
                   costs=CostConfig(**_tuples(raw.get("costs", {}), {"stress_multipliers"})))


    def as_dict(self) -> dict[str, Any]:
        
        return asdict(self)


def tuples(values: dict[str, Any], names: set[str]) -> dict[str, Any]:
    
    result = dict(values)
    
    for name in names:
        
        if name in result:
            
            result[name] = tuple(result[name])
    
    return result
