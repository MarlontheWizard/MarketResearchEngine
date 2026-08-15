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
class MicroRangeStatResult: 
    confirmed_ranges: pd.DataFrame
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


class MicroRangeStatProcessor: #Builds range events and future outcome measurements

    def __init__(self, config: MicroRangeStatConfig | None = None):
        
        self.config = config or MicroRangeStatConfig()


    def run(self, frame: pd.DataFrame) -> MicroRangeStatResult:
        
        df = prepare_frame(frame, self.config.columns)
        
        validation = self.validate_lifecycle(df)
        
        ranges, events = self.discover(df)
        
        outcomes = self.measure_outcomes(df, events)
        
        summary = self.summarize(outcomes)
        
        return MicroRangeStatResult(ranges, events, outcomes, summary, validation)


    def discover(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        c, ec = self.config.columns, self.config.events
        
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
            
            self._append_event(events, 
                               df, 
                               range_id, 
                               confirm_idx, 
                               first_idx, 
                               "FIRST_TRADABLE", 
                               "NONE", 
                               upper, 
                               lower, 
                               atr, 
                               invalid_idx, 
                               decision_idx=confirm_idx,
                               execution_idx=first_idx)
            
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
                              
        c, ec = self.config.columns, self.config.events
        
        seen: set[str] = set()
        
        above_run = below_run = 0
        
        prior_outside: str | None = None
        
        breakout_side: str | None = None
       
        breakout_idx: int | None = None
        
        upper_touch_zone_active = False
        
        lower_touch_zone_active = False

        counters = {"upper_touch_count": 0,
                    "lower_touch_count": 0,
                    "upper_break_attempt_count": 0,
                    "lower_break_attempt_count": 0,
                    "upper_outside_close_count": 0,
                    "lower_outside_close_count": 0,
                    "upper_wick_break_count": 0,
                    "lower_wick_break_count": 0,
                    "upper_reentry_count": 0,
                    "lower_reentry_count": 0,
                   }
        
        for idx in range(start, end + 1):
            
            row = df.iloc[idx]
            
            tolerance = ec.boundary_tolerance_atr * atr
            
            close, high, low = float(row[c.close]), float(row[c.high]), float(row[c.low])
            
            above = close > upper + ec.minimum_break_atr * atr
            
            below = close < lower - ec.minimum_break_atr * atr
            
            above_run = above_run + 1 if above else 0
            
            below_run = below_run + 1 if below else 0
            
            candidates: list[tuple[str, str]] = []
            
            in_upper_touch_zone = high >= upper - tolerance
            
            in_lower_touch_zone = low <= lower + tolerance
            
            if in_upper_touch_zone and not upper_touch_zone_active:
                
                counters["upper_touch_count"] += 1
                candidates.append(("UPPER_TOUCH", "UPPER"))
            
            if in_lower_touch_zone and not lower_touch_zone_active:
                
                counters["lower_touch_count"] += 1
                candidates.append(("LOWER_TOUCH", "LOWER"))
            
            upper_touch_zone_active = in_upper_touch_zone
            lower_touch_zone_active = in_lower_touch_zone
        
            
            if high > upper and close <= upper:
                
                counters["upper_wick_break_count"] += 1 #Wick break counts represent individual rejection candles. Touches represent visits to a zone.
                candidates.append(("UPPER_WICK_BREAK_CLOSE_INSIDE", "UPPER"))
            	
            if low < lower and close >= lower:
                
                counters["lower_wick_break_count"] += 1
                candidates.append(("LOWER_WICK_BREAK_CLOSE_INSIDE", "LOWER"))
            
            if above:
                
                counters["upper_outside_close_count"] += 1
                
                if above_run == 1:
                    
                    counters["upper_break_attempt_count"] += 1                    
                    candidates.append(("CLOSE_ABOVE_UPPER", "UPPER"))
                
                prior_outside = "UPPER"
                
                if breakout_side != "UPPER":
                
                    breakout_side = "UPPER"
                    
                    breakout_idx = idx
            
            elif below:
                
                counters["lower_outside_close_count"] += 1
                
                if below_run == 1:
                    
                    counters["lower_break_attempt_count"] += 1

                    candidates.append(("CLOSE_BELOW_LOWER", "LOWER"))
                
                prior_outside = "LOWER"
                
                if breakout_side != "LOWER":
  
                    breakout_side = "LOWER" 	
                    breakout_idx = idx
            
            elif lower <= close <= upper and prior_outside:
                
                counters[f"{prior_outside.lower()}_reentry_count"] += 1
                                
                candidates.append((f"REENTRY_FROM_{prior_outside}", prior_outside))
                
                prior_outside = None
                
                breakout_side = None
                
                breakout_idx = None
                
            
            retest_tolerance = ec.retest_tolerance_atr * atr
            
            if breakout_side == "UPPER" and breakout_idx is not None and idx > breakout_idx and low <= upper + retest_tolerance and close >= upper:
                
                candidates.append(("UPPER_BREAKOUT_RETEST_HOLD", "UPPER"))
                
                breakout_side = None
                
                breakout_idx = None
            
            elif breakout_side == "LOWER" and breakout_idx is not None and idx > breakout_idx and high >= lower - retest_tolerance and close <= lower:
                
                candidates.append(("LOWER_BREAKOUT_RETEST_HOLD", "LOWER"))
                
                breakout_side = None
                
                breakout_idx = None
                
                
            
            for count in ec.consecutive_outside_closes:
                
                if above_run == count:
                    
                    candidates.append((f"{count}_CLOSES_ABOVE_UPPER", "UPPER"))
                
                if below_run == count:
                    
                    candidates.append((f"{count}_CLOSES_BELOW_LOWER", "LOWER"))
                    
           
            if invalid_idx is not None and idx == invalid_idx:
            
                invalidation_kind = self._invalidation_kind(row, upper, lower, atr)
                candidates.append((invalidation_kind, _side_from_kind(invalidation_kind)))
            
            
            repeatable = {
                "UPPER_TOUCH", "LOWER_TOUCH",
                "UPPER_WICK_BREAK_CLOSE_INSIDE", "LOWER_WICK_BREAK_CLOSE_INSIDE",
                "CLOSE_ABOVE_UPPER", "CLOSE_BELOW_LOWER",
                "REENTRY_FROM_UPPER", "REENTRY_FROM_LOWER"
            }
            
        
        
            #Retain every occurrence of repeatable boundary events
            #Output non repeatable sequence events only once per confirmed range
            for kind, side in candidates:
                
                if kind in repeatable or kind not in seen:
                    
                    self._append_event(events, 
                                       df, range_id,
                                       confirm_idx, 
                                       idx, 
                                       kind, 
                                       side, 
                                       upper, 
                                       lower, 
                                       atr, 
                                       invalid_idx, 
                                       counters=counters,
                                       breakout_idx=breakout_idx,
                                       upper_outside_run=above_run,
                                       lower_outside_run=below_run,
                                       prior_outside_side=prior_outside,
                                       breakout_side=breakout_side)
                    
                    seen.add(kind)
                    
    def _append_event(self, 
    	              events: list[dict[str, Any]], 
                      df: pd.DataFrame, 
                      range_id: int,
                      confirm_idx: int, 
                      event_idx: int, 
                      event_type: str, 
                      boundary_side: str,
                      upper: float, 
                      lower: float, 
                      atr: float, 
                      invalid_idx: int | None,
                      decision_idx: int | None = None, 
                      execution_idx: int | None = None,
                      counters: dict[str, int] | None = None,
                      breakout_idx: int | None = None,
                      upper_outside_run: int = 0,
                      lower_outside_run: int = 0,
                      prior_outside_side: str | None = None,
                      breakout_side: str | None = None) -> None:
                      
        c = self.config.columns
        
        decision_idx = event_idx if decision_idx is None else decision_idx
        execution_idx = event_idx + 1 if execution_idx is None else execution_idx
 	
        event = {
            "event_id": len(events) + 1,
            "range_id": range_id,
            "event_type": event_type,
            "boundary_side": boundary_side,

            "confirmation_idx": confirm_idx,

            "decision_idx": decision_idx,
            "decision_timestamp": df.at[decision_idx, c.timestamp],

            "event_idx": event_idx,
            "event_timestamp": df.at[event_idx, c.timestamp],

            "execution_idx": execution_idx,
            "execution_timestamp": (df.at[execution_idx, c.timestamp] if execution_idx < len(df) else pd.NaT),

            "phase": ("POST_INVALIDATION" if invalid_idx is not None and event_idx >= invalid_idx else "ACTIVE_RANGE"),

            "upper": upper,
            "lower": lower,
            "midpoint": (upper + lower) / 2.0,

            "confirmation_atr": atr,

            "event_open": df.at[event_idx, c.open],
            "event_high": df.at[event_idx, c.high],
            "event_low": df.at[event_idx, c.low],
            "event_close": df.at[event_idx, c.close],
        }
        
        relative_features = self._build_event_relative_features(df,
                                                                confirmation_idx=confirm_idx,
                                                                first_tradable_idx=confirm_idx + 1,
                                                                decision_idx=decision_idx,
                                                                upper=upper,
                                                                lower=lower,
                                                                invalidation_idx=invalid_idx,
                                                                breakout_idx=breakout_idx,
                                                                counters=counters or {},
                                                                upper_outside_run=upper_outside_run,
                                                                lower_outside_run=lower_outside_run,
                                                                prior_outside_side=prior_outside_side,
                                                                breakout_side=breakout_side,
                                                               )


        event.update(relative_features)
            
        snapshot = self._build_context_snapshot(df,
                                                confirmation_idx=confirm_idx,
                                                decision_idx=decision_idx,
                                               )

        event.update(snapshot)
        
        events.append(event)


    def measure_outcomes(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
       	
        if events.empty:

            return pd.DataFrame()
        
        rows: list[dict[str, Any]] = []
        
        for event in events.to_dict("records"):
            
            entry_idx = int(event["execution_idx"])
            
            if entry_idx >= len(df):
                continue
            
            for direction in ("LONG", "SHORT"):
                rows.append(self._measure_one(df, event, entry_idx, direction))
        
        return pd.DataFrame(rows)


    def _measure_one(self, 
    	             df: pd.DataFrame, 
                     event: dict[str, Any], 
                     entry_idx: int, 
                     direction: str) -> dict[str, Any]: 
    	             
        c, oc, costs = (self.config.columns, 
                        self.config.outcomes,
                        self.config.costs)
                        
        sign = 1.0 if direction == "LONG" else -1.0
        
        entry = float(df.at[entry_idx, c.open])
        
        future_confirmation_positions = np.flatnonzero(df.loc[entry_idx + 1:,
                                                              c.confirmed_now].to_numpy())

       
        if len(future_confirmation_positions):

            next_confirmation_idx = (entry_idx + 1
                                               + int(future_confirmation_positions[0]))

            next_confirmation_timestamp = df.at[next_confirmation_idx,
                                                c.timestamp]

            bars_until_next_confirmation = (next_confirmation_idx - entry_idx)

        else:

            next_confirmation_idx = None
            next_confirmation_timestamp = pd.NaT
            bars_until_next_confirmation = None
            
        atr = max(float(event["confirmation_atr"]), np.finfo(float).eps)
        
        width = max(float(event["range_width"]), np.finfo(float).eps)
        
        full_end_idx = min(len(df) - 1,
                           entry_idx + oc.excursion_horizon - 1)

        full_path = df.iloc[entry_idx:full_end_idx + 1]

        if next_confirmation_idx is None:

            censored_end_idx = full_end_idx

        else:

            censored_end_idx = min(
                full_end_idx,
                next_confirmation_idx - 1,
            )

        if censored_end_idx >= entry_idx:

            censored_path = df.iloc[
                entry_idx:censored_end_idx + 1
            ]

        else:

            censored_path = df.iloc[0:0]
            
        
        highs = full_path[c.high].astype(float).to_numpy()

        lows = full_path[c.low].astype(float).to_numpy()

        closes = full_path[c.close].astype(float).to_numpy()

        favorable = highs - entry if direction == "LONG" else entry - lows
        
        adverse = entry - lows if direction == "LONG" else highs - entry
        
        mfe_i, mae_i = int(np.argmax(favorable)), int(np.argmax(adverse))
        
        result = dict(event)
        
        result.update({"direction": direction, "entry_idx": entry_idx,
            	       "entry_timestamp": df.at[entry_idx, c.timestamp], "entry_price": entry,
                       "mfe_price": float(favorable[mfe_i]), "mae_price": float(adverse[mae_i]),
                       "mfe_atr": float(favorable[mfe_i] / atr), "mae_atr": float(adverse[mae_i] / atr),
                       "mfe_range_width": float(favorable[mfe_i] / width), "mae_range_width": float(adverse[mae_i] / width),
                       "bars_to_mfe": mfe_i + 1, "bars_to_mae": mae_i + 1,
                       "path_efficiency": _path_efficiency(closes, entry, sign),
                       "favorable_to_adverse_ratio": float(favorable[mfe_i] / max(adverse[mae_i], np.finfo(float).eps)),
                       "outside_upper_close_share": float(np.mean(closes > event["upper"])),
                       "outside_lower_close_share": float(np.mean(closes < event["lower"])),
                       "inside_close_share": float(np.mean((closes >= event["lower"]) & (closes <= event["upper"]))),
                       "next_confirmation_idx": next_confirmation_idx,
                        "next_confirmation_timestamp": next_confirmation_timestamp,
                        "bars_until_next_confirmation": bars_until_next_confirmation,
                        "outcome_censored_by_next_confirmation": (next_confirmation_idx is not None and next_confirmation_idx <= full_end_idx),
                        "full_outcome_bars": len(full_path),
                        "censored_outcome_bars": len(censored_path)})
        
        
        if not censored_path.empty:

            censored_highs = censored_path[c.high].astype(float).to_numpy()

            censored_lows = censored_path[c.low].astype(float).to_numpy()

            censored_favorable = np.maximum((censored_highs - entry if direction == "LONG"
                                                                    else entry - censored_lows), 0.0)

            censored_adverse = np.maximum((entry - censored_lows
                                           if direction == "LONG"
                                           else censored_highs - entry), 0.0)

            censored_mfe_idx = int(np.argmax(censored_favorable))

            censored_mae_idx = int(np.argmax(censored_adverse))

            result.update(
                {
                    "censored_mfe_price": float(
                        censored_favorable[
                            censored_mfe_idx
                        ]
                    ),
                    "censored_mae_price": float(
                        censored_adverse[
                            censored_mae_idx
                        ]
                    ),
                    "censored_mfe_atr": float(
                        censored_favorable[
                            censored_mfe_idx
                        ] / atr
                    ),
                    "censored_mae_atr": float(
                        censored_adverse[
                            censored_mae_idx
                        ] / atr
                    ),
                    "censored_bars_to_mfe":
                        censored_mfe_idx + 1,
                    "censored_bars_to_mae":
                        censored_mae_idx + 1,
                }
            )

        else:

            result.update(
                {
                    "censored_mfe_price": np.nan,
                    "censored_mae_price": np.nan,
                    "censored_mfe_atr": np.nan,
                    "censored_mae_atr": np.nan,
                    "censored_bars_to_mfe": pd.NA,
                    "censored_bars_to_mae": pd.NA,
                }
            )
    
    
        for horizon in oc.horizons:
            
            target_idx = entry_idx + horizon - 1

            if target_idx >= len(df):

                result[f"return_{horizon}b_price"] = np.nan

                result[f"return_{horizon}b_atr"] = np.nan

                result[f"return_{horizon}b_censored_price"] = np.nan

                result[f"return_{horizon}b_censored_atr"] = np.nan
                
                
                for multiplier in costs.stress_multipliers:

                    label = str(multiplier).replace(".", "p")

                    result[f"return_{horizon}b_net_cost_x{label}"] = np.nan

                continue


            raw = sign * (float(df.at[target_idx, c.close])- entry)

            #full outcome
            result[f"return_{horizon}b_price"] = raw

            result[f"return_{horizon}b_atr"] = raw / atr

            for multiplier in costs.stress_multipliers:

                label = str(multiplier).replace(".","p",)

                result[f"return_{horizon}b_net_cost_x{label}"] = (raw - costs.round_trip_price * multiplier)

            #censored at the next confirmation
            available_before_next_confirmation = (next_confirmation_idx is None or target_idx < next_confirmation_idx)

            if available_before_next_confirmation:

                result[f"return_{horizon}b_censored_price"] = raw

                result[f"return_{horizon}b_censored_atr"] = raw / atr

            else:

                result[f"return_{horizon}b_censored_price"] = np.nan

                result[f"return_{horizon}b_censored_atr"] = np.nan
        
        
        risk = oc.risk_atr * atr
        
        result.update(self._first_passage(highs, lows, entry, sign, risk, event))
        
        result["ambiguous_bars"] = self._ambiguous_count(highs, lows, entry, sign, risk)
        
        return result

    
    def _first_passage(self, 
    		       highs: np.ndarray, 
                       lows: np.ndarray, 
                       entry: float, 
                       sign: float,
                       risk: float, 
                       event: dict[str, Any]) -> dict[str, Any]:
        
        
        result: dict[str, Any] = {}
        
        for level in self.config.outcomes.r_levels:
            
            favorable_level = entry + sign * level * risk
            
            adverse_level = entry - sign * level * risk
            
            fav = _first_hit(highs, lows, favorable_level, sign, favorable=True)
            
            adv = _first_hit(highs, lows, adverse_level, sign, favorable=False)
            
            key = str(level).replace(".", "p")
            
            result[f"first_plus_{key}r_bar"] = fav
            
            result[f"first_minus_{key}r_bar"] = adv
            
            result[f"plus_before_minus_{key}r"] = _ordered(fav, adv, self.config.outcomes.intrabar_policy)
        
        for level in self.config.outcomes.range_width_levels:
            
            favorable_level = entry + sign * level * float(event["range_width"])
            
            key = str(level).replace(".", "p")
            
            result[f"first_plus_{key}_range_width_bar"] = _first_hit(highs, lows, favorable_level, sign, favorable=True)
            
        for name, price in (("midpoint", event["midpoint"]), ("upper", event["upper"]), ("lower", event["lower"])):
            
            result[f"first_{name}_touch_bar"] = _first_level_touch(highs, lows, float(price))
        
        return result


    def _ambiguous_count(self, highs: np.ndarray, lows: np.ndarray, entry: float, sign: float, risk: float) -> int:
       
        target, stop = entry + sign * risk, entry - sign * risk
        
        return int(np.sum((highs >= max(target, stop)) & (lows <= min(target, stop))))


    def _invalidation_kind(self, row: pd.Series, upper: float, lower: float, atr: float) -> str:
        
        c, threshold = self.config.columns, self.config.events.minimum_break_atr * atr
        
        close, high, low = float(row[c.close]), float(row[c.high]), float(row[c.low])
        
        if close > upper + threshold:
            return "INVALIDATION_BREAK_UP"
        
        if close < lower - threshold:
            return "INVALIDATION_BREAK_DOWN"
        
        if high > upper and close <= upper:
            return "INVALIDATION_WICK_UP"
        
        if low < lower and close >= lower:
            return "INVALIDATION_WICK_DOWN"
        
        if lower <= close <= upper:
            
            return "INVALIDATION_INSIDE_DECAY"
        
        return "INVALIDATION_AMBIGUOUS"

   
    def validate_lifecycle(self,
                           df: pd.DataFrame) -> pd.DataFrame:
        c = self.config.columns
        issues: list[dict[str, Any]] = []

        range_is_open = False
        active_range_confirmation_idx: int | None = None
        frozen_upper: float | None = None
        frozen_lower: float | None = None

        def add_issue(severity: str,
                      row_idx: int,
                      issue: str) -> None:

            issues.append({"severity": severity,
                           "row": row_idx,
                           "timestamp": df.at[row_idx, c.timestamp],
                           "issue": issue})


        for idx in range(len(df)):

            confirmed_now = bool(df.at[idx, c.confirmed_now])

            first_tradable_now = bool(df.at[idx, c.first_tradable_now])

            active_live = bool(df.at[idx, c.active_live])

            invalidated_now = bool(df.at[idx, c.invalidated_now])

            current_upper = df.at[idx, c.confirmed_upper]
            current_lower = df.at[idx, c.confirmed_lower]


            #validate confirmation
            if confirmed_now:

                if range_is_open:
                    
                    add_issue("ERROR",
                              idx,
                              "new_confirmation_before_previous_invalidation")

                if (pd.isna(current_upper) or 
                    pd.isna(current_lower) or 
                    float(current_upper) <= float(current_lower)):
                    
                    add_issue("ERROR",
                              idx,
                              "invalid_confirmed_boundaries")

                    frozen_upper = None
                    
                    frozen_lower = None

                else:
                    frozen_upper = float(current_upper)
                    
                    frozen_lower = float(current_lower)

                if not active_live:
                    
                    add_issue("ERROR",
                              idx,
                              "confirmation_not_active_live")

                range_is_open = True
                
                active_range_confirmation_idx = idx


            #validate first-tradable shift

            if first_tradable_now:

                if idx == 0:
                    
                    add_issue("ERROR",
                              idx,
                              "first_tradable_on_first_row")

                elif not bool(df.at[idx - 1, c.confirmed_now]):
                    
                    add_issue("ERROR",
                              idx,
                              "first_tradable_without_prior_confirmation")

                if not range_is_open:
                    add_issue("ERROR",
                              idx,
                              "first_tradable_without_open_range")

                if not active_live:
                    add_issue("ERROR",
                              idx,
                              "first_tradable_not_active_live")

            if confirmed_now:

                if idx + 1 >= len(df):
                    
                    add_issue("WARNING",
                              idx,
                              "confirmation_at_end_of_dataset")

                elif not bool(df.at[idx + 1, c.first_tradable_now]):
                    
                    add_issue("ERROR",
                              idx,
                              "confirmation_not_followed_by_first_tradable")

            #validate active-live state
            if active_live and not range_is_open:

                add_issue(
                    "ERROR",
                    idx,
                    "active_live_without_open_range",
                )

            if (
                range_is_open
                and not invalidated_now
                and not active_live
            ):
                add_issue(
                    "ERROR",
                    idx,
                    "active_live_dropped_before_invalidation",
                )


            #validate frozen boundaries while range is open

            if (range_is_open
                and frozen_upper is not None
                and frozen_lower is not None):

                if not pd.isna(current_upper):

                    if not np.isclose(float(current_upper),
                        	          frozen_upper,
                                      rtol=0.0,
                                      atol=np.finfo(float).eps):
                        
                        add_issue("ERROR",
                                  idx,
                                  "confirmed_upper_changed_while_active")

                if not pd.isna(current_lower):

                    if not np.isclose(
                        float(current_lower),
                        frozen_lower,
                        rtol=0.0,
                        atol=np.finfo(float).eps,
                    ):
                        add_issue(
                            "ERROR",
                            idx,
                            "confirmed_lower_changed_while_active",
                        )


            #validate invalidation (LOL)

            if invalidated_now:

                if not range_is_open:
                    add_issue(
                        "ERROR",
                        idx,
                        "invalidation_without_open_range",
                    )

                # active_live may be either True or False on this exact
                # candle. The range is considered closed afterward.
                range_is_open = False
                active_range_confirmation_idx = None
                frozen_upper = None
                frozen_lower = None


            #normal result


        if not issues:
        
            issues.append({"severity": "INFO",
                           "row": pd.NA,
                           "timestamp": pd.NaT,
                           "issue": "no_lifecycle_issues_found",
                         })

        return pd.DataFrame(issues)

    @staticmethod
    def summarize(outcomes: pd.DataFrame) -> pd.DataFrame:
        
        if outcomes.empty:
            
            return pd.DataFrame()
        
        metric_cols = [name for name in outcomes if name.startswith("return_") and (name.endswith("_atr") or "net_cost" in name)]
        
        metric_cols += ["mfe_atr", "mae_atr", "path_efficiency", "favorable_to_adverse_ratio", "inside_close_share"]
        
        grouped = outcomes.groupby(["event_type", "phase", "direction"], dropna=False)
        
        summary = grouped[metric_cols].agg(["count", "mean", "median"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        
        return summary.reset_index()

    def _build_event_relative_features(self,
                                       df: pd.DataFrame,
                                       *,
                                       confirmation_idx: int,
                                       first_tradable_idx: int,
                                       decision_idx: int,
                                       upper: float,
                                       lower: float,
                                       invalidation_idx: int | None,
                                       breakout_idx: int | None,
                                       counters: dict[str, int],
                                       upper_outside_run: int,
                                       lower_outside_run: int,
                                       prior_outside_side: str | None,
                                       breakout_side: str | None) -> dict[str, Any]:

        
        c = self.config.columns
        row = df.iloc[decision_idx]
        
        #invalidation is usable only when it has occurred by the
        #decision candle, this makes a future invalidation unknown.
        invalidation_known = (invalidation_idx is not None and invalidation_idx <= decision_idx)
        
        high = float(row[c.high])
        low = float(row[c.low])
        close = float(row[c.close])
        atr = float(row[c.atr])

        #range-relative
        range_width = upper - lower

        safe_atr = (atr if np.isfinite(atr) and atr > 0.0 else np.nan)

        safe_width = (range_width if np.isfinite(range_width) and range_width > 0.0 else np.nan)

        relative_features = {"range_width": range_width,
                             "range_width_atr": range_width / safe_atr,

                             "close_position_in_confirmed_range": (close - lower) / safe_width,
                             "close_distance_to_upper_atr": (close - upper) / safe_atr,
                             "close_distance_to_lower_atr": (close - lower) / safe_atr,
                             "high_distance_to_upper_atr": (high - upper) / safe_atr,
                             "low_distance_to_lower_atr": (low - lower) / safe_atr}

        default_counters = {"upper_touch_count": 0,
                            "lower_touch_count": 0,
                            "upper_break_attempt_count": 0,
                            "lower_break_attempt_count": 0,
                            "upper_outside_close_count": 0,
                            "lower_outside_close_count": 0,
                            "upper_wick_break_count": 0,
                            "lower_wick_break_count": 0,
                            "upper_reentry_count": 0,
                            "lower_reentry_count": 0,
                        }

        default_counters.update(counters)

        #add the remaining lifecycle and counter features.
        relative_features.update({
            "range_age_bars": decision_idx - confirmation_idx,
            "bars_since_confirmation": decision_idx - confirmation_idx,
            "bars_since_first_tradable":     decision_idx - first_tradable_idx if decision_idx >= first_tradable_idx else None,

            "bars_since_invalidation": ( decision_idx - invalidation_idx if invalidation_known else None),
            "range_is_active": not invalidation_known,
            "range_was_invalidated": invalidation_known,

            "upper_outside_run": upper_outside_run,
            "lower_outside_run": lower_outside_run,
            "prior_outside_side": prior_outside_side,
            "pending_breakout_side": breakout_side,

            "bars_since_breakout": ( decision_idx - breakout_idx if breakout_idx is not None else None), **default_counters})

        return relative_features
        
        
    def _build_context_snapshot(self, 
                                df: pd.DataFrame,
                                *,
                                confirmation_idx: int,
                                decision_idx: int) -> dict[str, Any]:


        confirmation_row = df.iloc[confirmation_idx]
        decision_row = df.iloc[decision_idx]

        snapshot: dict[str, Any] = {}

        #save confirmation and decision context (for ai)
        for column in self.config.snapshots.feature_columns:
            confirmation_value = confirmation_row[column]
            decision_value = decision_row[column]

            snapshot[f"confirmation__{column}"] = confirmation_value
            snapshot[f"decision__{column}"] = decision_value


        #target changes only inapproved change columns.
        for column in self.config.snapshots.change_columns:
            
            confirmation_value = confirmation_row[column]
            
            decision_value = decision_row[column]

            if pd.isna(confirmation_value) or pd.isna(decision_value):
                
                snapshot[f"change__{column}"] = np.nan
                
                continue

            try:
                
                snapshot[f"change__{column}"] = (float(decision_value) - float(confirmation_value))
            
            except (TypeError, ValueError):
                
                snapshot[f"change__{column}"] = np.nan

        return snapshot
        
#---------------------------------------------------------------------------------------------------------------------------------------------------
    
def _finite_or(value: Any, fallback: float) -> float:
    
    try:
    
        number = float(value)
        
        return number if np.isfinite(number) and number > 0 else float(fallback)
    
    except (TypeError, ValueError):
        
        return float(fallback)


def _side_from_kind(kind: str) -> str:
    
    if "UP" in kind:
        
        return "UPPER"
    
    if "DOWN" in kind:
        
        return "LOWER"
    
    return "NONE"


def _path_efficiency(closes: np.ndarray, entry: float, sign: float) -> float:
    
    if not len(closes):
        
        return np.nan
    
    travel = abs(closes[0] - entry) + np.abs(np.diff(closes)).sum()
    
    return float(sign * (closes[-1] - entry) / max(travel, np.finfo(float).eps))


def _first_hit(highs: np.ndarray, lows: np.ndarray, level: float, sign: float, favorable: bool) -> int | None:
    
    use_high = (sign > 0) == favorable
    
    hits = np.flatnonzero(highs >= level) if use_high else np.flatnonzero(lows <= level)
    
    return int(hits[0] + 1) if len(hits) else None


def _first_level_touch(highs: np.ndarray, lows: np.ndarray, level: float) -> int | None:
    
    hits = np.flatnonzero((highs >= level) & (lows <= level))
    
    return int(hits[0] + 1) if len(hits) else None


def _ordered(first: int | None, second: int | None, policy: str) -> bool | None:
    
    if first is None:
        return False
    
    if second is None:
        return True
    
    if first < second:
        return True
    
    if first > second:
        return False
    
    if policy == "optimistic":
        return True
    
    if policy == "conservative":
        return False
    
    return None
