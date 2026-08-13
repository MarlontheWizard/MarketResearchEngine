#Author: Marlon Dominguez
#Date  : 08/13/2026

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config         import MicroRangeStatConfig
from .data_processor import prepare_frame


@dataclass
class MicroRangeStatResult: confirmed_ranges: pd.DataFrame
    		    	       boundary_events: pd.DataFrame
    	            	       event_outcomes: pd.DataFrame
    		    	       event_summary: pd.DataFrame
    	            	       validation_report: pd.DataFrame


    def write(self, output_dir: str | Path) -> None:
        
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        self.confirmed_ranges.to_parquet(output / "confirmed_ranges.parquet", index=False)
        self.boundary_events.to_parquet(output / "boundary_events.parquet", index=False)
        self.event_outcomes.to_parquet(output / "event_outcomes.parquet", index=False)
        self.event_summary.to_csv(output / "event_summary.csv", index=False)
        self.validation_report.to_csv(output / "validation_report.csv", index=False)


class MicroRangeStatProcessor: #Builds range events and future outcome measurements"""

    def __init__(self, config: MicroRangeStatConfig | None = None):
        
        self.config = config or MicroRangeStatConfig()


    def run(self, frame: pd.DataFrame) -> MicroRangeStatResult:
        
        df = prepare_frame(frame, self.config.columns)
        
        validation = self._validate_lifecycle(df)
        
        ranges, events = self._discover(df)
        
        outcomes = self._measure_outcomes(df, events)
        
        summary = self._summarize(outcomes)
        
        return MicroRangeStatResult(ranges, events, outcomes, summary, validation)


    def _discover(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        c, ec = self.config.columns, 
                self.config.events
        
        confirmations = np.flatnonzero(df[c.confirmed_now].to_numpy())
        
        ranges: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        
        for range_id, confirm_idx in enumerate(confirmations, start=1):
            
            upper = float(df.at[confirm_idx, c.confirmed_upper])
            
            lower = float(df.at[confirm_idx, c.confirmed_lower])
            
            if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
                
                continue
            
            atr = _finite_or(df.at[confirm_idx, c.atr], upper - lower)
            
            first_idx = confirm_idx + 1
            
            max_end = min(len(df) - 1, confirm_idx + ec.max_bars_from_confirmation)
            
            future_confirmations = confirmations[confirmations > confirm_idx]
            
            if len(future_confirmations):
                
                max_end = min(max_end, int(future_confirmations[0]) - 1)
            
            invalidation_candidates = np.flatnonzero(df.loc[first_idx:max_end, c.invalidated_now].to_numpy())
            
            invalid_idx = first_idx + int(invalidation_candidates[0]) if len(invalidation_candidates) else None
            
            observation_end = max_end if invalid_idx is None else min(max_end, invalid_idx + ec.post_invalidation_bars)
            
            ranges.append({"range_id": range_id, "confirmation_idx": confirm_idx,
                           "confirmation_timestamp": df.at[confirm_idx, c.timestamp],
                           "first_tradable_idx": first_idx,
                           "first_tradable_timestamp": df.at[first_idx, c.timestamp] if first_idx < len(df) else pd.NaT,
                           "invalidation_idx": invalid_idx,
                           "invalidation_timestamp": df.at[invalid_idx, c.timestamp] if invalid_idx is not None else pd.NaT,
                           "observation_end_idx": observation_end, "upper": upper, "lower": lower,
                           "midpoint": (upper + lower) / 2.0, "width": upper - lower,
                           "confirmation_atr": atr})
                           
            if first_idx >= len(df):
                
                continue
            
            self._append_event(events, df, range_id, confirm_idx, first_idx, "FIRST_TRADABLE", "NONE", upper, lower, atr, invalid_idx)
            
            if invalid_idx is not None:
                
                kind = self._invalidation_kind(df.iloc[invalid_idx], upper, lower, atr)
                
                self._append_event(events, df, range_id, confirm_idx, invalid_idx, kind, _side_from_kind(kind), upper, lower, atr, invalid_idx)
            
            self._scan_boundary_events(df, events, range_id, confirm_idx, first_idx, observation_end, invalid_idx, upper, lower, atr)
       
       
        return pd.DataFrame(ranges), pd.DataFrame(events)



    def _scan_boundary_events(self, 
                              df: pd.DataFrame, 
                              events: list[dict[str, Any]], 
                              range_id: int,
                              confirm_idx: int, 
                              start: int, 
                              end: int, 
                              invalid_idx: int | None,
                              upper: float, 
                              lower: float, atr: float) -> None:
                              
        c, ec = self.config.columns, 
        	self.config.events
        
        seen: set[str] = set()
        
        above_run = below_run = 0
        
        prior_outside: str | None = None
        
        breakout_side: str | None = None
        
        for idx in range(start, end + 1):
            
            row = df.iloc[idx]
            
            tolerance = ec.boundary_tolerance_atr * atr
            
            close, high, low = float(row[c.close]), float(row[c.high]), float(row[c.low])
            
            above = close > upper + ec.minimum_break_atr * atr
            
            below = close < lower - ec.minimum_break_atr * atr
            
            above_run = above_run + 1 if above else 0
            
            below_run = below_run + 1 if below else 0
            
            candidates: list[tuple[str, str]] = []
            
            if high >= upper - tolerance:
                
                candidates.append(("UPPER_TOUCH", "UPPER"))
            
            if low <= lower + tolerance:
                
                candidates.append(("LOWER_TOUCH", "LOWER"))
            
            if high > upper and close <= upper:
                
                candidates.append(("UPPER_WICK_BREAK_CLOSE_INSIDE", "UPPER"))
            
            if low < lower and close >= lower:
                
                candidates.append(("LOWER_WICK_BREAK_CLOSE_INSIDE", "LOWER"))
            
            if above:
                
                candidates.append(("CLOSE_ABOVE_UPPER", "UPPER"))
                
                prior_outside = "UPPER"
                
                breakout_side = "UPPER"
            
            elif below:
                
                candidates.append(("CLOSE_BELOW_LOWER", "LOWER"))
                
                prior_outside = "LOWER"
                
                breakout_side = "LOWER"
            
            elif lower <= close <= upper and prior_outside:
                
                candidates.append((f"REENTRY_FROM_{prior_outside}", prior_outside))
                
                prior_outside = None
            
            retest_tolerance = ec.retest_tolerance_atr * atr
            
            if breakout_side == "UPPER" and low <= upper + retest_tolerance and close >= upper:
                
                candidates.append(("UPPER_BREAKOUT_RETEST_HOLD", "UPPER"))
                
                breakout_side = None
            
            elif breakout_side == "LOWER" and high >= lower - retest_tolerance and close <= lower:
                
                candidates.append(("LOWER_BREAKOUT_RETEST_HOLD", "LOWER"))
                
                breakout_side = None
            
            for count in ec.consecutive_outside_closes:
                
                if above_run == count:
                    
                    candidates.append((f"{count}_CLOSES_ABOVE_UPPER", "UPPER"))
                
                if below_run == count:
                    
                    candidates.append((f"{count}_CLOSES_BELOW_LOWER", "LOWER"))
                    
           
            #Output first occurrence of each event per range storing repeated touch behavior as counters.
            for kind, side in candidates:
                
                if kind not in seen:
                    
                    self._append_event(events, df, range_id, confirm_idx, idx, kind, side, upper, lower, atr, invalid_idx)
                    
                    seen.add(kind)
