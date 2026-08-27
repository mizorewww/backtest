"""Compounding + asymmetric-delta optimization on the cached Deribit quotes.

Compounding model: start with 1 unit of base currency (BTC/ETH) held as spot.
Each week the whole stack is used as notional for the strangle (B_t units -> B_t
contracts per leg); the weekly base-currency return r_t is the per-unit pnl, so
  equity_base_t = prod(1 + r_t)            (stack size in BTC/ETH)
  equity_usd_t  = prod((1 + r_t) * S_t/S_{t-1})   (mark-to-market in USD)
buy&hold_usd    = prod(S_t/S_{t-1})

Also runs a call-delta x put-delta grid (legs chosen independently) and reports
drawdown episodes for selected configs.

Usage: uv run optimize.py
"""

import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import (DELTA_GRID, INDEX, OPEN_FEE_CAP, OPEN_FEE_RATE,
                      RAW_DIR, RESULTS_DIR, SETTLE_FEE_CAP,
                      SETTLE_FEE_RATE, load_delivery, load_quotes)


def week_leg(wk: dict, side: str, target: float) -> dict | None:
    pool = [q for q in wk["quotes"] if q["side"] == side]
    if not pool:
        return None
    return min(pool, key=lambda q: abs(q["delta"] - target))


def simulate(weeks: list[dict], delivery: dict[str, float],
             dc: float, dp: float) -> pd.DataFrame:
    """Weekly per-unit returns for independent call/put delta targets."""
    rows = []
    for wk in weeks:
        s_t = delivery.get(wk["sunday"])
        if s_t is None:
            continue
        qc, qp = week_leg(wk, "C", dc), week_leg(wk, "P", -dp)
        if qc is None or qp is None:
            continue
        pnl = 0.0
        for q, s0, st in ((qc, wk["spot"], s_t), (qp, wk["spot"], s_t)):
            k = q["strike"]
            payoff = (max(st - k, 0.0) if q["side"] == "C" else max(k - st, 0.0)) / st
            open_fee = min(OPEN_FEE_CAP * q["price"], OPEN_FEE_RATE)
            settle_fee = min(SETTLE_FEE_CAP * payoff, SETTLE_FEE_RATE) if payoff > 0 else 0.0
            pnl += q["price"] - payoff - open_fee - settle_fee
        rows.append({"friday": wk["friday"], "ret": pnl,
                     "spot_ratio": s_t / wk["spot"],
                     "C_strike": qc["strike"], "P_strike": qp["strike"],
                     "entry_spot": wk["spot"], "settle": s_t})
    return pd.DataFrame(rows)


def metrics(t: pd.DataFrame) -> dict:
    r = t["ret"]
    eq = (1 + r).cumprod()
    n = len(t)
    dd = (eq / eq.cummax() - 1).min()
    usd = ((1 + r) * t["spot_ratio"]).cumprod()
    bh = t["spot_ratio"].cumprod()
    dd_usd = (usd / usd.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (52 / n) - 1
    sharpe = r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan
    return {"weeks": n, "equity_final": eq.iloc[-1], "cagr": cagr,
            "sharpe_w": sharpe, "max_dd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "win_rate": (r > 0).mean(),
            "usd_final": usd.iloc[-1], "bh_final": bh.iloc[-1], "max_dd_usd": dd_usd}


def drawdown_episodes(t: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Top-N peak->trough episodes of the compounded base-currency equity."""
    eq = (1 + t["ret"]).cumprod().reset_index(drop=True)
    fridays = t["friday"].reset_index(drop=True)
    episodes = []
    peak_idx, peak_val = 0, eq.iloc[0]
    i = 0
    while i < len(eq):
        if eq.iloc[i] >= peak_val:
            peak_idx, peak_val = i, eq.iloc[i]
            i += 1
            continue
        # in a drawdown: find trough
        j = i
        while j < len(eq) and eq.iloc[j] < peak_val:
            j += 1
        trough_pos = eq.iloc[i:j].idxmin()
        episodes.append({"peak_friday": fridays.iloc[peak_idx],
                         "trough_friday": fridays.iloc[trough_pos],
                         "recovered_friday": fridays.iloc[j - 1] if j < len(eq) else None,
                         "depth": eq.iloc[trough_pos] / peak_val - 1,
                         "weeks_to_trough": trough_pos - peak_idx})
        i = j
    ep = pd.DataFrame(episodes)
    return ep.nsmallest(top, "depth") if len(ep) else ep


def heatmap(df: pd.DataFrame, value: str, cur: str, path: Path) -> None:
    piv = df.pivot(index="delta_put", columns="delta_call", values=value)
    plt.figure(figsize=(8, 6))
    plt.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    plt.colorbar(label=value)
    plt.xticks(range(len(piv.columns)), [f"{c:.2f}" for c in piv.columns])
    plt.yticks(range(len(piv.index)), [f"{c:.2f}" for c in piv.index])
    plt.xlabel("call delta")
    plt.ylabel("put delta")
    plt.title(f"Deribit {cur}: {value} over call/put delta grid")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def analyze(cur: str, weeks: list[dict], delivery: dict[str, float]) -> None:
    print(f"\n{'=' * 70}\n{cur} ({len(weeks)} weeks)\n{'=' * 70}")

    # symmetric grid with compounding
    sym = []
    for d in DELTA_GRID:
        t = simulate(weeks, delivery, d, d)
        sym.append({"delta": d, **metrics(t)})
    sym_df = pd.DataFrame(sym)
    sym_df.to_csv(RESULTS_DIR / f"compound_sym_{cur}.csv", index=False)
    cols = ["delta", "equity_final", "cagr", "sharpe_w", "max_dd", "calmar",
            "win_rate", "usd_final", "bh_final", "max_dd_usd"]
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n-- symmetric delta grid, compounded --")
    print(sym_df[cols].to_string(index=False))

    # asymmetric grid
    rows = []
    for dc, dp in product(DELTA_GRID, DELTA_GRID):
        t = simulate(weeks, delivery, dc, dp)
        rows.append({"delta_call": dc, "delta_put": dp, **metrics(t)})
    asym = pd.DataFrame(rows)
    asym.to_csv(RESULTS_DIR / f"compound_asym_{cur}.csv", index=False)
    heatmap(asym, "cagr", cur, RESULTS_DIR / f"heatmap_cagr_{cur}.png")
    heatmap(asym, "calmar", cur, RESULTS_DIR / f"heatmap_calmar_{cur}.png")
    print("\n-- top 10 (dc, dp) by Calmar --")
    print(asym.nlargest(10, "calmar")[["delta_call", "delta_put", "cagr",
                                       "sharpe_w", "max_dd", "calmar", "win_rate"]]
          .to_string(index=False))
    print("-- top 10 (dc, dp) by CAGR --")
    print(asym.nlargest(10, "cagr")[["delta_call", "delta_put", "cagr",
                                     "sharpe_w", "max_dd", "calmar", "win_rate"]]
          .to_string(index=False))

    # drawdown episodes for 35d symmetric and best-Calmar combo
    best = asym.loc[asym["calmar"].idxmax()]
    for label, dc, dp in (("35/35", 0.35, 0.35),
                          (f"best-calmar {best['delta_call']}/{best['delta_put']}",
                           best["delta_call"], best["delta_put"])):
        t = simulate(weeks, delivery, dc, dp)
        ep = drawdown_episodes(t)
        ep.to_csv(RESULTS_DIR / f"drawdowns_{cur}_{label.split()[0].replace('/', '-')}.csv",
                  index=False)
        print(f"\n-- top drawdown episodes, {cur} {label} --")
        print(ep.to_string(index=False))
        worst = t.nsmallest(6, "ret")
        print(f"worst weeks ({label}):")
        print(worst[["friday", "ret", "entry_spot", "settle", "C_strike", "P_strike"]]
              .to_string(index=False))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for cur in ("BTC", "ETH"):
        analyze(cur, load_quotes(cur), load_delivery(cur))


if __name__ == "__main__":
    main()
