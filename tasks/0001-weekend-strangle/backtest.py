"""Backtest on Deribit data: hold 1 unit of BTC/ETH, every Friday 16:00 UTC sell 1x
Sunday-expiry call + put with delta closest to the target, hold to 08:00 UTC settlement.

All option quantities are in base currency (BTC/ETH), as Deribit options are
base-currency margined/settled (contract size 1, prices in BTC/ETH):
  payoff_call = max(S_T - K, 0) / S_T,  premium = traded price (BTC/ETH per contract).
Greeks: Black-Scholes r=0, IV inverted from the traded entry price (T = 40h).
Fees (Deribit): open 0.03% of notional capped at 12.5% of premium;
settlement 0.015% of notional capped at 12.5% of payoff, ITM only.

Usage:
  uv run backtest.py --underlying BTC --delta 0.35
  uv run backtest.py            # grid for BTC + ETH
"""

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import erf


def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / 1.4142135623730951))

RAW_DIR = Path(os.environ.get("TRADE_DATA_ROOT", "/Volumes/trade/data")) / "deribit"
PROJECT = Path(__file__).parent
RESULTS_DIR = PROJECT / "results"

ENTRY_HOUR = 16
T_YEARS = 40 / (365 * 24)          # Friday 16:00 -> Sunday 08:00
OPEN_FEE_RATE = 0.0003             # 0.03% of 1-unit notional, in base currency
OPEN_FEE_CAP = 0.125               # x premium
SETTLE_FEE_RATE = 0.00015
SETTLE_FEE_CAP = 0.125             # x payoff

DELTA_GRID = [round(0.05 * i, 2) for i in range(1, 13)]
INDEX = {"BTC": "btc_usd", "ETH": "eth_usd"}


# ---------------------------------------------------------------- Black-Scholes (r = 0)

def bs_price(s: float, k: float, sigma: float, side: str) -> float:
    """USD value per 1 unit notional."""
    if sigma <= 0:
        return max(s - k, 0.0) if side == "C" else max(k - s, 0.0)
    sq = sigma * np.sqrt(T_YEARS)
    d1 = (np.log(s / k) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    if side == "C":
        return s * _ncdf(d1) - k * _ncdf(d2)
    return k * _ncdf(-d2) - s * _ncdf(-d1)


def bs_delta(s: float, k: float, sigma: float, side: str) -> float:
    sq = sigma * np.sqrt(T_YEARS)
    d1 = (np.log(s / k) + 0.5 * sq * sq) / sq
    return _ncdf(d1) if side == "C" else _ncdf(d1) - 1


def invert_iv(price_usd: float, s: float, k: float, side: str) -> float | None:
    intrinsic = bs_price(s, k, 0.0, side)
    if price_usd <= intrinsic * 0.999:
        return None
    lo, hi = 1e-3, 6.0
    if bs_price(s, k, hi, side) < price_usd:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(s, k, mid, side) < price_usd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- data loading

def load_delivery(cur: str) -> dict[str, float]:
    df = pd.read_csv(RAW_DIR / f"delivery_{INDEX[cur]}.csv")
    return dict(zip(df["date"], df["delivery_price"]))


def entry_price(candles: dict) -> tuple[float, str] | None:
    """(price in base currency, source tag); open of 16:00 candle preferred."""
    hours = [int(pd.Timestamp(t, unit="ms", tz="UTC").hour) for t in candles["ticks"]]
    table = {h: (o, c, v) for h, o, c, v in
             zip(hours, candles["open"], candles["close"], candles["volume"])}
    if ENTRY_HOUR in table and table[ENTRY_HOUR][2] > 0:
        return table[ENTRY_HOUR][0], "open@16"
    for h in (15, 14, 13, 12):                    # last trade before 16:00
        if h in table and table[h][2] > 0:
            return table[h][1], f"close@{h}"
    for h in (17, 18):                            # first trade after 16:00
        if h in table and table[h][2] > 0:
            return table[h][0], f"open@{h}"
    return None


def load_quotes(cur: str) -> list[dict]:
    """Per week: all candidate contracts with entry price, IV and delta."""
    weeks = []
    for fp in sorted(f for f in (RAW_DIR / "weekly" / cur).glob("*.json")
                     if not f.name.startswith("._")):
        try:
            rec = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue  # file still being written by the downloader
        if not rec.get("contracts"):
            continue  # week without a listed Sunday expiry
        s = rec["spot"]
        quotes = []
        for name, candles in rec["contracts"].items():
            _, expiry_s, strike_s, side = name.split("-")
            ep = entry_price(candles)
            if ep is None:
                continue
            price, src = ep
            iv = invert_iv(price * s, s, float(strike_s), side)
            if iv is None:
                continue
            quotes.append({"name": name, "strike": float(strike_s), "side": side,
                           "price": price, "iv": iv,
                           "delta": bs_delta(s, float(strike_s), iv, side), "src": src})
        if quotes:
            weeks.append({"friday": rec["friday"], "sunday": rec["sunday"],
                          "spot": s, "quotes": quotes})
    return weeks


# ---------------------------------------------------------------- backtest

def run_backtest(cur: str, target_delta: float, weeks: list[dict],
                 delivery: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    trades = []
    for wk in weeks:
        s0 = wk["spot"]
        s_t = delivery.get(wk["sunday"])
        if s_t is None:
            continue
        legs, ok = {}, True
        for side, tgt in (("C", target_delta), ("P", -target_delta)):
            pool = [q for q in wk["quotes"] if q["side"] == side]
            if not pool:
                ok = False
                break
            q = min(pool, key=lambda q: abs(q["delta"] - tgt))
            legs[side] = q
        if not ok:
            continue

        pnl = premium_sum = fees = 0.0
        rec = {"friday": wk["friday"], "sunday": wk["sunday"],
               "entry_spot": s0, "settle": s_t, "spot_ret": s_t / s0 - 1}
        for side in ("C", "P"):
            q = legs[side]
            k = q["strike"]
            payoff = (max(s_t - k, 0.0) if side == "C" else max(k - s_t, 0.0)) / s_t
            open_fee = min(OPEN_FEE_CAP * q["price"], OPEN_FEE_RATE)
            settle_fee = min(SETTLE_FEE_CAP * payoff, SETTLE_FEE_RATE) if payoff > 0 else 0.0
            leg_pnl = q["price"] - payoff - open_fee - settle_fee
            pnl += leg_pnl
            premium_sum += q["price"]
            fees += open_fee + settle_fee
            rec.update({f"{side}_name": q["name"], f"{side}_strike": k,
                        f"{side}_delta": q["delta"], f"{side}_iv": q["iv"],
                        f"{side}_premium": q["price"], f"{side}_payoff": payoff,
                        f"{side}_src": q["src"]})
        rec.update(pnl=pnl, premium_sum=premium_sum, fees=fees,
                   pnl_usd=pnl * s0, pnl_pct=pnl)  # pnl in base == pct of 1-unit notional
        trades.append(rec)

    t = pd.DataFrame(trades)
    if t.empty:
        return t, {}
    cum = t["pnl"].cumsum()
    ret = t["pnl_pct"]
    summary = {
        "weeks": len(t),
        "total_pnl_base": t["pnl"].sum(),
        "total_pnl_usd": t["pnl_usd"].sum(),
        "ret_on_notional": t["pnl"].sum() / len(t),
        "win_rate": (t["pnl"] > 0).mean(),
        "sharpe_w": ret.mean() / ret.std() * np.sqrt(52) if ret.std() > 0 else np.nan,
        "max_dd_base": (cum - cum.cummax()).min(),
        "premium_sum_base": t["premium_sum"].sum(),
        "retained": t["pnl"].sum() / t["premium_sum"].sum() if t["premium_sum"].sum() else np.nan,
        "first": t["friday"].iloc[0], "last": t["friday"].iloc[-1],
    }
    return t, summary


def print_grid(cur: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["delta", "weeks", "total_pnl_base", "ret_on_notional", "win_rate",
            "sharpe_w", "max_dd_base", "retained"]
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(f"\n===== Deribit {cur} delta grid "
          f"({rows[0]['first']} .. {rows[0]['last']}) =====")
    print(df[cols].to_string(index=False))
    best = df.loc[df["total_pnl_base"].idxmax()]
    print(f"--> best by total PnL: delta={best['delta']:.2f} "
          f"(PnL {best['total_pnl_base']:.4f} {cur}, Sharpe {best['sharpe_w']:.2f})")
    return df


def plot_grid(cur: str, curves: dict[float, pd.Series], path: Path) -> None:
    plt.figure(figsize=(11, 5.5))
    for d, s in curves.items():
        plt.plot(pd.to_datetime(s.index), s.values, label=f"{d:.2f}", lw=1)
    plt.axhline(0, color="k", lw=0.5)
    plt.title(f"Deribit {cur} Friday short strangle (Sunday expiry): cumulative PnL ({cur})")
    plt.xlabel("date")
    plt.ylabel(f"cum PnL ({cur})")
    plt.legend(title="delta", ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all", choices=["BTC", "ETH", "all"])
    ap.add_argument("--delta", type=float, default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    curs = ["BTC", "ETH"] if args.underlying == "all" else [args.underlying]
    for cur in curs:
        delivery = load_delivery(cur)
        weeks = load_quotes(cur)
        print(f"{cur}: {len(weeks)} tradeable weeks loaded")
        deltas = [args.delta] if args.delta else DELTA_GRID
        rows, curves = [], {}
        for d in deltas:
            trades, summ = run_backtest(cur, d, weeks, delivery)
            if trades.empty:
                continue
            tag = f"deribit_{cur}_d{int(round(d * 100)):02d}"
            trades.to_csv(RESULTS_DIR / f"trades_{tag}.csv", index=False)
            rows.append({"delta": d, **summ})
            curves[d] = trades.set_index("friday")["pnl"].cumsum()
        if not rows:
            continue
        grid_df = print_grid(cur, rows)
        grid_df.to_csv(RESULTS_DIR / f"grid_deribit_{cur}.csv", index=False)
        plot_grid(cur, curves, RESULTS_DIR / f"equity_deribit_{cur}.png")
        print(f"results: results/grid_deribit_{cur}.csv, results/equity_deribit_{cur}.png")


if __name__ == "__main__":
    main()
