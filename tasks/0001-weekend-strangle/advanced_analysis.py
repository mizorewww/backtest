"""Advanced quantitative validation experiments for weekend crypto options short strangle:
1. Weekday Control & Variance Risk Premium (VRP) comparison (Fri->Sun vs Mon->Wed, Tue->Thu, Wed->Fri).
2. Intraday Mark Path, Margin Health & Saturday Liquidation Stress Simulation.
3. Out-Of-Sample (OOS) Walk-Forward & Leave-One-Tail-Event-Out Parameter Stability.
4. Asymmetric Delta (0.25C/0.60P) Directional Beta vs Volatility Alpha Decomposition.
5. Execution Quote Source & Data Integrity Audit.
6. Systematic Loss Scenarios Taxonomy & Stress Gate Design.

Usage:
  uv run advanced_analysis.py
"""

import json
import os
from itertools import product
from math import erf, sqrt, log, exp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import (
    DELTA_GRID, INDEX, OPEN_FEE_CAP, OPEN_FEE_RATE, RAW_DIR, RESULTS_DIR,
    SETTLE_FEE_CAP, SETTLE_FEE_RATE, T_YEARS, bs_delta, bs_price, invert_iv,
    load_delivery, load_quotes
)
from optimize import simulate, metrics, drawdown_episodes


def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / 1.4142135623730951))


# -----------------------------------------------------------------------------
# 1. Weekday Control & Variance Risk Premium (VRP) Experiment
# -----------------------------------------------------------------------------

def run_weekday_control(cur: str) -> pd.DataFrame:
    """Compare 40h Realized Volatility and short-vol payoffs across 4 windows:
    - Fri 16:00 -> Sun 08:00 (Weekend)
    - Mon 16:00 -> Wed 08:00 (Weekday 1)
    - Tue 16:00 -> Thu 08:00 (Weekday 2)
    - Wed 16:00 -> Fri 08:00 (Weekday 3)
    """
    perp_file = RAW_DIR / f"perp_1h_{cur}-PERPETUAL.csv"
    if not perp_file.exists():
        print(f"Perp file missing: {perp_file}")
        return pd.DataFrame()

    df_perp = pd.read_csv(perp_file)
    df_perp["dt"] = pd.to_datetime(df_perp["ts"], utc=True)
    df_perp = df_perp.sort_values("dt").reset_index(drop=True)
    df_perp.set_index("dt", inplace=True)

    # Windows definition: (start_dow, start_hour, duration_hours, label)
    # DOW: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    windows = [
        {"dow": 4, "hour": 16, "dur": 40, "name": "Fri 16:00 -> Sun 08:00 (Weekend)"},
        {"dow": 0, "hour": 16, "dur": 40, "name": "Mon 16:00 -> Wed 08:00 (Weekday Mon-Wed)"},
        {"dow": 1, "hour": 16, "dur": 40, "name": "Tue 16:00 -> Thu 08:00 (Weekday Tue-Thu)"},
        {"dow": 2, "hour": 16, "dur": 40, "name": "Wed 16:00 -> Fri 08:00 (Weekday Wed-Fri)"},
    ]

    records = []
    # Find all matching window starts
    for w_idx, win in enumerate(windows):
        target_dts = df_perp.index[(df_perp.index.dayofweek == win["dow"]) & (df_perp.index.hour == win["hour"])]
        for start_dt in target_dts:
            end_dt = start_dt + pd.Timedelta(hours=win["dur"])
            if end_dt not in df_perp.index:
                continue
            
            sub = df_perp.loc[start_dt:end_dt]
            if len(sub) < 35: # need enough candles
                continue
            
            prices = sub["close"].values
            log_rets = np.diff(np.log(prices))
            # 40h annualized RV
            rv_ann = np.std(log_rets) * np.sqrt(8760)
            rv_var_ann = (rv_ann ** 2)
            
            spot_start = prices[0]
            spot_end = prices[-1]
            abs_ret = abs(spot_end / spot_start - 1.0)
            max_up = np.max(prices) / spot_start - 1.0
            max_down = np.min(prices) / spot_start - 1.0
            max_swing = max_up - max_down
            
            records.append({
                "window": win["name"],
                "window_short": win["name"].split()[0],
                "start_dt": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_dt": end_dt.strftime("%Y-%m-%d %H:%M"),
                "spot_start": spot_start,
                "spot_end": spot_end,
                "rv_ann": rv_ann,
                "rv_var_ann": rv_var_ann,
                "spot_move_pct": (spot_end / spot_start - 1.0) * 100,
                "abs_move_pct": abs_ret * 100,
                "max_swing_pct": max_swing * 100,
                "is_weekend": (win["dow"] == 4),
            })

    df_res = pd.DataFrame(records)
    
    # Also attach weekend implied volatility from Deribit quotes if available
    quotes = load_quotes(cur)
    iv_map = {}
    for q in quotes:
        # extract mean ATM IV from quotes
        valid_ivs = [item["iv"] for item in q["quotes"] if 0.25 <= item["delta"] <= 0.45 or -0.45 <= item["delta"] <= -0.25]
        if valid_ivs:
            iv_map[q["friday"]] = np.mean(valid_ivs)

    df_res["iv_ann"] = np.nan
    for idx, row in df_res.iterrows():
        start_date = row["start_dt"][:10]
        if row["is_weekend"] and start_date in iv_map:
            df_res.loc[idx, "iv_ann"] = iv_map[start_date]

    return df_res


def summarize_weekday_control(df_res: pd.DataFrame, cur: str) -> pd.DataFrame:
    summary = []
    for w_name, grp in df_res.groupby("window"):
        rv_mean = grp["rv_ann"].mean() * 100
        rv_median = grp["rv_ann"].median() * 100
        abs_move_mean = grp["abs_move_pct"].mean()
        abs_move_p90 = grp["abs_move_pct"].quantile(0.90)
        max_swing_mean = grp["max_swing_pct"].mean()
        
        # simulated short 30Δ strangle payoff: assume typical premium ~ 0.012 BTC (~50% IV), strike distance ~ 4.5%
        # a strangle wins if abs move < strike_distance + premium
        win_sim = (grp["abs_move_pct"] < 4.5).mean() * 100
        
        summary.append({
            "window": w_name,
            "sample_count": len(grp),
            "mean_rv_ann_pct": rv_mean,
            "median_rv_ann_pct": rv_median,
            "mean_abs_move_pct": abs_move_mean,
            "p90_abs_move_pct": abs_move_p90,
            "mean_max_swing_pct": max_swing_mean,
            "sim_win_rate_pct": win_sim,
        })
    df_sum = pd.DataFrame(summary).sort_values("mean_rv_ann_pct")
    df_sum.to_csv(RESULTS_DIR / f"vrp_weekday_control_{cur}.csv", index=False)
    return df_sum


# -----------------------------------------------------------------------------
# 2. Intraday Mark Path, Margin Health & Saturday Liquidation Stress Simulation
# -----------------------------------------------------------------------------

def simulate_intraday_margin(cur: str, dc: float = 0.35, dp: float = 0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate hourly mark price, floating PnL, margin requirement, and liquidation
    checks across the 40-hour holding period for each week.
    
    Deribit Inverse Option Standard Margin Rules:
    - Base Initial Margin (IM) ~ 15% notional + Mark Price
    - Base Maintenance Margin (MM) ~ 10% notional + Mark Price
    - Strangle MM = MM(Call) + MM(Put) ~ 0.10 * 1.0 (since 1 unit notional) + Mark_C + Mark_P
    - Account Equity (BTC) = 1.0 (spot) + Net_Premium - Mark_C(t) - Mark_P(t)
    - Liquidation occurs if Account Equity < Total MM (Margin Health < 1.0)
    """
    perp_file = RAW_DIR / f"perp_1h_{cur}-PERPETUAL.csv"
    df_perp = pd.read_csv(perp_file)
    df_perp["dt"] = pd.to_datetime(df_perp["ts"], utc=True)
    df_perp.set_index("dt", inplace=True)

    weeks = load_quotes(cur)
    delivery = load_delivery(cur)

    weekly_stress = []
    hourly_records = []

    for wk in weeks:
        friday_str = wk["friday"]
        sunday_str = wk["sunday"]
        s_settle = delivery.get(sunday_str)
        if s_settle is None:
            continue
        
        pool_c = [q for q in wk["quotes"] if q["side"] == "C"]
        pool_p = [q for q in wk["quotes"] if q["side"] == "P"]
        if not pool_c or not pool_p:
            continue
        
        qc = min(pool_c, key=lambda q: abs(q["delta"] - dc))
        qp = min(pool_p, key=lambda q: abs(q["delta"] - (-dp)))
        
        kc, kp = qc["strike"], qp["strike"]
        prem_c, prem_p = qc["price"], qp["price"]
        open_fee = min(OPEN_FEE_CAP * prem_c, OPEN_FEE_RATE) + min(OPEN_FEE_CAP * prem_p, OPEN_FEE_RATE)
        net_prem = prem_c + prem_p - open_fee
        
        # Settle terminal PnL
        payoff_c = max(s_settle - kc, 0.0) / s_settle
        payoff_p = max(kp - s_settle, 0.0) / s_settle
        settle_fee = (min(SETTLE_FEE_CAP * payoff_c, SETTLE_FEE_RATE) if payoff_c > 0 else 0) + \
                     (min(SETTLE_FEE_CAP * payoff_p, SETTLE_FEE_RATE) if payoff_p > 0 else 0)
        terminal_pnl = net_prem - payoff_c - payoff_p - settle_fee

        # Extract 40h intraday hourly candles
        start_dt = pd.Timestamp(f"{friday_str} 16:00:00", tz="UTC")
        end_dt = pd.Timestamp(f"{sunday_str} 08:00:00", tz="UTC")
        sub_candles = df_perp.loc[start_dt:end_dt]
        if len(sub_candles) < 20:
            continue
        
        min_health = 999.0
        max_floating_loss = 0.0
        min_equity = 1.0 + net_prem
        max_mm = 0.10 + prem_c + prem_p
        liquidated = False
        liq_time = None
        liq_spot = None
        
        # Track hourly marks
        for i, (ts_curr, row) in enumerate(sub_candles.iterrows()):
            spot_t = row["close"]
            # remaining time to expiry in years
            hours_left = max(40.0 - i, 0.5)
            t_left = hours_left / 8760.0
            
            # Option mark pricing using entry IV as proxy
            iv_c = qc["iv"]
            iv_p = qp["iv"]
            # Stress IV smile: if spot drops, put IV expands; if spot jumps, call IV expands
            spot_ret = (spot_t / wk["spot"] - 1.0)
            iv_stress_p = iv_p * (1.0 + max(-spot_ret * 2.0, 0.0))
            iv_stress_c = iv_c * (1.0 + max(spot_ret * 1.5, 0.0))
            
            usd_mark_c = bs_price(spot_t, kc, iv_stress_c, "C")
            usd_mark_p = bs_price(spot_t, kp, iv_stress_p, "P")
            
            # Convert to base currency (BTC/ETH)
            mark_c_base = usd_mark_c / spot_t
            mark_p_base = usd_mark_p / spot_t
            
            # Floating option PnL per contract
            float_pnl = net_prem - mark_c_base - mark_p_base
            if float_pnl < max_floating_loss:
                max_floating_loss = float_pnl
            
            # Maintenance Margin requirement: 0.10 * notional + mark values
            mm_t = 0.10 + mark_c_base + mark_p_base
            equity_t = 1.0 + net_prem - mark_c_base - mark_p_base
            
            health_t = equity_t / mm_t if mm_t > 0 else 999.0
            if health_t < min_health:
                min_health = health_t
            if equity_t < min_equity:
                min_equity = equity_t
            if mm_t > max_mm:
                max_mm = mm_t
            
            if health_t < 1.0 and not liquidated:
                liquidated = True
                liq_time = ts_curr
                liq_spot = spot_t

            if i in (0, 10, 20, 30, 40):
                hourly_records.append({
                    "friday": friday_str,
                    "hour": i,
                    "ts": ts_curr.strftime("%Y-%m-%d %H:%M"),
                    "spot": spot_t,
                    "mark_c": mark_c_base,
                    "mark_p": mark_p_base,
                    "floating_pnl": float_pnl,
                    "margin_health": health_t,
                })

        weekly_stress.append({
            "friday": friday_str,
            "entry_spot": wk["spot"],
            "settle_spot": s_settle,
            "kc": kc,
            "kp": kp,
            "terminal_pnl": terminal_pnl,
            "max_floating_loss": max_floating_loss,
            "gap_intraday_vs_terminal": max_floating_loss - terminal_pnl,
            "min_margin_health": min_health,
            "min_equity": min_equity,
            "max_mm_required": max_mm,
            "liquidated": liquidated,
            "liq_time": str(liq_time) if liq_time else None,
            "liq_spot": liq_spot,
        })

    df_weekly = pd.DataFrame(weekly_stress)
    df_hourly = pd.DataFrame(hourly_records)
    
    df_weekly.to_csv(RESULTS_DIR / f"intraday_margin_stress_{cur}.csv", index=False)
    return df_weekly, df_hourly


# -----------------------------------------------------------------------------
# 3. Out-of-Sample (OOS) Walk-Forward & Leave-One-Tail-Event-Out Validation
# -----------------------------------------------------------------------------

def run_leave_one_tail_out(cur: str) -> pd.DataFrame:
    """Drop top drawdown weeks one by one and observe parameter stability."""
    weeks = load_quotes(cur)
    delivery = load_delivery(cur)

    # Baseline all weeks
    base_rows = []
    for dc, dp in product(DELTA_GRID, DELTA_GRID):
        t = simulate(weeks, delivery, dc, dp)
        base_rows.append({"delta_call": dc, "delta_put": dp, **metrics(t)})
    df_base = pd.DataFrame(base_rows)
    best_base_calmar = df_base.loc[df_base["calmar"].idxmax()]
    best_base_sharpe = df_base.loc[df_base["sharpe_w"].idxmax()]

    # Find top 7 worst weeks in baseline 35Δ
    t_35 = simulate(weeks, delivery, 0.35, 0.35)
    worst_weeks = t_35.nsmallest(7, "ret")["friday"].tolist()

    results = [{
        "dropped_event": "None (Full 208 Weeks)",
        "best_c_calmar": best_base_calmar["delta_call"],
        "best_p_calmar": best_base_calmar["delta_put"],
        "calmar_value": best_base_calmar["calmar"],
        "cagr": best_base_calmar["cagr"],
        "max_dd": best_base_calmar["max_dd"],
        "best_c_sharpe": best_base_sharpe["delta_call"],
        "best_p_sharpe": best_base_sharpe["delta_put"],
        "sharpe_value": best_base_sharpe["sharpe_w"],
    }]

    for w_drop in worst_weeks:
        sub_weeks = [w for w in weeks if w["friday"] != w_drop]
        rows = []
        for dc, dp in product(DELTA_GRID, DELTA_GRID):
            t = simulate(sub_weeks, delivery, dc, dp)
            rows.append({"delta_call": dc, "delta_put": dp, **metrics(t)})
        df_sub = pd.DataFrame(rows)
        best_c = df_sub.loc[df_sub["calmar"].idxmax()]
        best_s = df_sub.loc[df_sub["sharpe_w"].idxmax()]
        
        results.append({
            "dropped_event": f"Drop {w_drop}",
            "best_c_calmar": best_c["delta_call"],
            "best_p_calmar": best_c["delta_put"],
            "calmar_value": best_c["calmar"],
            "cagr": best_c["cagr"],
            "max_dd": best_c["max_dd"],
            "best_c_sharpe": best_s["delta_call"],
            "best_p_sharpe": best_s["delta_put"],
            "sharpe_value": best_s["sharpe_w"],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / f"leave_one_tail_out_{cur}.csv", index=False)
    return df_res


def run_walk_forward(cur: str, initial_train_weeks: int = 104, test_step_weeks: int = 26) -> tuple[pd.DataFrame, dict]:
    """Expanding walk-forward out-of-sample optimization.
    Train on historical weeks -> Select best (dc, dp) by Calmar -> Evaluate on next OOS window.
    """
    weeks = load_quotes(cur)
    delivery = load_delivery(cur)
    n = len(weeks)

    oos_trades = []
    windows_meta = []
    
    step_start = initial_train_weeks
    window_idx = 1
    
    while step_start < n:
        train_weeks = weeks[:step_start]
        test_end = min(step_start + test_step_weeks, n)
        test_weeks = weeks[step_start:test_end]
        
        # Optimize on train_weeks
        rows = []
        for dc, dp in product(DELTA_GRID, DELTA_GRID):
            # restrict to OTM strangle search space (dc <= 0.45, dp <= 0.45)
            if dc > 0.45 or dp > 0.45:
                continue
            t = simulate(train_weeks, delivery, dc, dp)
            rows.append({"delta_call": dc, "delta_put": dp, **metrics(t)})
        df_train = pd.DataFrame(rows)
        best = df_train.loc[df_train["calmar"].idxmax()]
        sel_dc, sel_dp = best["delta_call"], best["delta_put"]
        
        # Evaluate on test_weeks
        t_test = simulate(test_weeks, delivery, sel_dc, sel_dp)
        t_test["selected_dc"] = sel_dc
        t_test["selected_dp"] = sel_dp
        t_test["window_idx"] = window_idx
        oos_trades.append(t_test)
        
        windows_meta.append({
            "window": window_idx,
            "train_range": f"{train_weeks[0]['friday']} ~ {train_weeks[-1]['friday']}",
            "train_weeks": len(train_weeks),
            "test_range": f"{test_weeks[0]['friday']} ~ {test_weeks[-1]['friday']}",
            "test_weeks": len(test_weeks),
            "selected_dc": sel_dc,
            "selected_dp": sel_dp,
            "train_calmar": best["calmar"],
            "test_cagr": metrics(t_test)["cagr"],
            "test_win_rate": metrics(t_test)["win_rate"],
        })
        
        step_start = test_end
        window_idx += 1

    df_oos = pd.concat(oos_trades, ignore_index=True)
    df_meta = pd.DataFrame(windows_meta)
    
    oos_metrics = metrics(df_oos)
    df_meta.to_csv(RESULTS_DIR / f"walk_forward_windows_{cur}.csv", index=False)
    df_oos.to_csv(RESULTS_DIR / f"walk_forward_oos_trades_{cur}.csv", index=False)
    
    return df_meta, oos_metrics


# -----------------------------------------------------------------------------
# 4. Asymmetric Delta Decomposition & Directional Beta vs Volatility Alpha
# -----------------------------------------------------------------------------

def run_delta_decomposition(cur: str) -> pd.DataFrame:
    """Decompose returns of:
    - 30Δ Symmetric (0.30C / 0.30P) -> Pure Delta Neutral short-vol baseline
    - 35Δ Symmetric (0.35C / 0.35P)
    - 25C / 60P Asymmetric -> Bullish Bias (+0.35 option delta + 1.0 spot)
    """
    weeks = load_quotes(cur)
    delivery = load_delivery(cur)

    configs = [
        ("Symmetric 25Δ", 0.25, 0.25, 0.0),
        ("Symmetric 30Δ", 0.30, 0.30, 0.0),
        ("Symmetric 35Δ", 0.35, 0.35, 0.0),
        ("Asymmetric 25C/60P", 0.25, 0.60, 0.35),
        ("Asymmetric 40C/55P", 0.40, 0.55, 0.15),
    ]

    decomp = []
    for label, dc, dp, net_opt_delta in configs:
        t = simulate(weeks, delivery, dc, dp)
        m = metrics(t)
        
        # Spot returns
        spot_rets = t["settle"] / t["entry_spot"] - 1.0
        opt_rets = t["ret"]
        
        # Regression: opt_ret = alpha + beta * spot_ret
        cov = np.cov(opt_rets, spot_rets)
        var_spot = np.var(spot_rets)
        beta = cov[0, 1] / var_spot if var_spot > 0 else 0.0
        alpha_ann = (opt_rets.mean() - beta * spot_rets.mean()) * 52
        
        # Total portfolio return (Spot + Option Overlay)
        port_rets = (1 + opt_rets) * (1 + spot_rets) - 1.0
        port_equity = (1 + port_rets).cumprod()
        port_cagr = port_equity.iloc[-1] ** (52 / len(t)) - 1.0
        port_dd = (port_equity / port_equity.cummax() - 1.0).min()
        
        # ROE under 30% margin allocation
        roe_cagr = ((1 + opt_rets / 0.30).cumprod().iloc[-1]) ** (52 / len(t)) - 1.0
        
        decomp.append({
            "config": label,
            "call_delta": dc,
            "put_delta": dp,
            "net_option_delta": net_opt_delta,
            "total_portfolio_delta": 1.0 + net_opt_delta,
            "overlay_cagr": m["cagr"],
            "overlay_sharpe": m["sharpe_w"],
            "overlay_max_dd": m["max_dd"],
            "overlay_calmar": m["calmar"],
            "beta_to_spot": beta,
            "alpha_ann": alpha_ann,
            "spot_plus_overlay_cagr": port_cagr,
            "spot_plus_overlay_max_dd": port_dd,
            "roe_30pct_margin_cagr": roe_cagr,
        })

    df_decomp = pd.DataFrame(decomp)
    df_decomp.to_csv(RESULTS_DIR / f"delta_decomposition_{cur}.csv", index=False)
    return df_decomp


# -----------------------------------------------------------------------------
# 5. Execution Quote Source Audit
# -----------------------------------------------------------------------------

def run_quote_source_audit(cur: str) -> pd.DataFrame:
    """Audit the execution quote fallback sources across delta buckets."""
    raw_weeks = sorted((RAW_DIR / "weekly" / cur).glob("*.json"))
    
    bucket_counts = {round(0.05 * i, 2): {"open@16": 0, "close@15": 0, "close@14_12": 0, "open@17_18": 0, "total": 0}
                     for i in range(1, 13)}
    
    for fp in raw_weeks:
        try:
            rec = json.loads(fp.read_text())
        except Exception:
            continue
        contracts = rec.get("contracts", {})
        spot = rec.get("spot", 0.0)
        if not contracts or spot <= 0:
            continue
        
        for name, data in contracts.items():
            hours = [int(pd.Timestamp(t, unit="ms", tz="UTC").hour) for t in data["ticks"]]
            table = {h: (o, c, v) for h, o, c, v in zip(hours, data["open"], data["close"], data["volume"])}
            
            src = None
            price = None
            if 16 in table and table[16][2] > 0:
                price = table[16][0]
                src = "open@16"
            else:
                for h in (15, 14, 13, 12):
                    if h in table and table[h][2] > 0:
                        price = table[h][1]
                        src = "close@15" if h == 15 else "close@14_12"
                        break
                if src is None:
                    for h in (17, 18):
                        if h in table and table[h][2] > 0:
                            price = table[h][0]
                            src = "open@17_18"
                            break
            if src is None or price is None or price <= 0:
                continue
            
            # Invert IV & delta
            parts = name.split("-")
            side = parts[-1]
            strike = float(parts[-2])
            usd_p = price * spot
            iv = invert_iv(usd_p, spot, strike, side)
            if iv is None:
                continue
            d = abs(bs_delta(spot, strike, iv, side))
            
            # Find closest delta bucket
            closest_d = min(DELTA_GRID, key=lambda x: abs(x - d))
            bucket_counts[closest_d]["total"] += 1
            bucket_counts[closest_d][src] += 1

    rows = []
    for d in DELTA_GRID:
        b = bucket_counts[d]
        tot = b["total"]
        rows.append({
            "delta_bucket": d,
            "total_quotes": tot,
            "pct_open_16_ontime": (b["open@16"] / tot * 100) if tot else 0,
            "pct_close_15_stale": (b["close@15"] / tot * 100) if tot else 0,
            "pct_close_14_12_stale": (b["close@14_12"] / tot * 100) if tot else 0,
            "pct_open_17_18_lookahead": (b["open@17_18"] / tot * 100) if tot else 0,
        })
    df_audit = pd.DataFrame(rows)
    df_audit.to_csv(RESULTS_DIR / f"execution_source_audit_{cur}.csv", index=False)
    return df_audit


# -----------------------------------------------------------------------------
# 6. Loss Scenarios Classification & Case Studies
# -----------------------------------------------------------------------------

def run_loss_taxonomy(cur: str) -> tuple[pd.DataFrame, list[dict]]:
    """Categorize every losing week into 4 quantitative loss archetypes:
    1. Type 1: Downside Tail Crash (Put Breached, Spot dropped > 5%)
    2. Type 2: Upside Short Squeeze (Call Breached, Spot rose > 5%)
    3. Type 3: Intraday Path Stress / Margin Danger (High swing, severe MAE)
    4. Type 4: Vega Expansion / Choppy Grind (Neither breached or small breach, high fee/loss)
    """
    weeks = load_quotes(cur)
    delivery = load_delivery(cur)
    t = simulate(weeks, delivery, 0.35, 0.35)
    
    perp_file = RAW_DIR / f"perp_1h_{cur}-PERPETUAL.csv"
    df_perp = pd.read_csv(perp_file)
    df_perp["dt"] = pd.to_datetime(df_perp["ts"], utc=True)
    df_perp.set_index("dt", inplace=True)

    losses = []
    case_studies = []

    for _, row in t.iterrows():
        ret = row["ret"]
        s0 = row["entry_spot"]
        st = row["settle"]
        kc = row["C_strike"]
        kp = row["P_strike"]
        spot_chg_pct = (st / s0 - 1.0) * 100
        friday = row["friday"]

        # 40h intraday path
        start_dt = pd.Timestamp(f"{friday} 16:00:00", tz="UTC")
        end_dt = pd.Timestamp(f"{friday} 16:00:00", tz="UTC") + pd.Timedelta(hours=40)
        sub = df_perp.loc[start_dt:end_dt]
        intraday_max = (sub["close"].max() / s0 - 1.0) * 100 if len(sub) else spot_chg_pct
        intraday_min = (sub["close"].min() / s0 - 1.0) * 100 if len(sub) else spot_chg_pct
        intraday_swing = intraday_max - intraday_min

        # Categorize
        if st < kp:
            ltype = "Type 1: Downside Tail Crash (Put Breached)"
            desc = f"现货大跌 {spot_chg_pct:.2f}% 跌破 Put 行权价 ${kp:,.0f}，币本位负伽马与价值损失"
        elif st > kc:
            ltype = "Type 2: Upside Short Squeeze (Call Breached)"
            desc = f"现货暴涨 +{spot_chg_pct:.2f}% 冲破 Call 行权价 ${kc:,.0f}，空头踩踏导致深实值"
        elif intraday_swing > 8.0:
            ltype = "Type 3: Intraday Path / Swing Risk (High Volatility)"
            desc = f"终盘虽未深度穿行权价，但盘中剧烈震荡 {intraday_swing:.2f}%，产生巨大盘中浮亏压力"
        else:
            ltype = "Type 4: Choppy Loss / Volatility Mismatch"
            desc = "微幅震荡但由于手续费或接近行权价产生小幅亏损"

        is_loss = (ret < 0)
        losses.append({
            "friday": friday,
            "ret": ret,
            "is_loss": is_loss,
            "loss_type": ltype,
            "spot_entry": s0,
            "spot_settle": st,
            "spot_change_pct": spot_chg_pct,
            "call_strike": kc,
            "put_strike": kp,
            "intraday_swing_pct": intraday_swing,
            "intraday_min_pct": intraday_min,
            "intraday_max_pct": intraday_max,
            "description": desc,
        })

    df_loss = pd.DataFrame(losses)
    
    # Extract top case studies
    top_losses = df_loss[df_loss["is_loss"]].nsmallest(8, "ret")
    for _, r in top_losses.iterrows():
        case_studies.append({
            "date": r["friday"],
            "loss_type": r["loss_type"],
            "pnl_base": round(r["ret"], 4),
            "spot_move_pct": round(r["spot_change_pct"], 2),
            "intraday_swing_pct": round(r["intraday_swing_pct"], 2),
            "strikes": f"C: ${r['call_strike']:,.0f} / P: ${r['put_strike']:,.0f}",
            "description": r["description"],
        })

    df_loss.to_csv(RESULTS_DIR / f"loss_taxonomy_{cur}.csv", index=False)
    with open(RESULTS_DIR / f"loss_case_studies_{cur}.json", "w") as f:
        json.dump(case_studies, f, indent=2, ensure_ascii=False)

    return df_loss, case_studies


# -----------------------------------------------------------------------------
# Generate Publication Figures
# -----------------------------------------------------------------------------

def generate_advanced_plots(cur: str = "BTC") -> None:
    # 1. Weekday Control VRP Plot
    df_vrp = pd.read_csv(RESULTS_DIR / f"vrp_weekday_control_{cur}.csv")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_vrp["window"].apply(lambda x: x.split("(")[-1].replace(")", "")),
                   df_vrp["mean_rv_ann_pct"], color=["#10B981", "#3B82F6", "#6366F1", "#8B5CF6"],
                   width=0.5, edgecolor="black", linewidth=1.2)
    plt.ylabel("Annualized Realized Volatility (RV %)", fontsize=11)
    plt.title(f"Deribit {cur}: 40h Window Realized Volatility Comparison (2022-2026)", fontsize=13, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"fig_vrp_weekday_control_{cur}.png", dpi=140)
    plt.close()

    # 2. Intraday Max Floating Loss vs Terminal Settlement PnL
    df_stress = pd.read_csv(RESULTS_DIR / f"intraday_margin_stress_{cur}.csv")
    plt.figure(figsize=(11, 5))
    plt.scatter(df_stress["terminal_pnl"], df_stress["max_floating_loss"], color="#EF4444", alpha=0.6, edgecolors="none", s=30)
    plt.plot([-0.25, 0.05], [-0.25, 0.05], linestyle="--", color="#6B7280", label="Terminal == Intraday (No Path Divergence)")
    plt.xlabel("Terminal Settlement PnL (BTC)", fontsize=11)
    plt.ylabel("Maximum Intraday Floating PnL (MAE, BTC)", fontsize=11)
    plt.title(f"Deribit {cur} 35Δ: Terminal PnL vs Maximum Intraday Adverse Excursion", fontsize=13, pad=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"fig_intraday_margin_path_{cur}.png", dpi=140)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 80)
    print("Running Advanced Quantitative Experiments...")
    print("=" * 80)

    for cur in ("BTC", "ETH"):
        print(f"\n>>> Running Experiments for {cur}...")
        
        # 1. Weekday control
        df_vrp_raw = run_weekday_control(cur)
        if not df_vrp_raw.empty:
            df_vrp_sum = summarize_weekday_control(df_vrp_raw, cur)
            print(f"\n[1] Weekday Control Summary ({cur}):\n", df_vrp_sum.to_string(index=False))

        # 2. Intraday margin & path
        df_stress_wk, _ = simulate_intraday_margin(cur, dc=0.35, dp=0.35)
        print(f"\n[2] Intraday Stress Summary ({cur}):")
        print(f"Total weeks: {len(df_stress_wk)}, Liquidations breached: {(df_stress_wk['liquidated']).sum()}")
        print(f"Worst Intraday Floating Loss: {df_stress_wk['max_floating_loss'].min():.4f} {cur} vs Worst Terminal: {df_stress_wk['terminal_pnl'].min():.4f} {cur}")

        # 3. Leave-one-tail-out
        df_loto = run_leave_one_tail_out(cur)
        print(f"\n[3] Leave-One-Tail-Out Stability ({cur}):\n", df_loto.to_string(index=False))

        # 4. Walk-forward OOS
        df_wf_meta, oos_metrics = run_walk_forward(cur)
        print(f"\n[4] Walk-Forward OOS Metrics ({cur}):", oos_metrics)

        # 5. Delta decomposition
        df_decomp = run_delta_decomposition(cur)
        print(f"\n[5] Delta Decomposition ({cur}):\n", df_decomp[["config", "net_option_delta", "overlay_cagr", "beta_to_spot", "alpha_ann"]].to_string(index=False))

        # 6. Execution quote source audit
        df_audit = run_quote_source_audit(cur)
        print(f"\n[6] Execution Quote Audit ({cur}):\n", df_audit.head(6).to_string(index=False))

        # 7. Loss taxonomy
        df_loss, cases = run_loss_taxonomy(cur)
        print(f"\n[7] Top Loss Case Studies ({cur}): {len(cases)} critical events logged.")

        # Plots
        generate_advanced_plots(cur)

    print("\nAll advanced quantitative experiments completed successfully!")


if __name__ == "__main__":
    main()
