"""Download Deribit data for the weekend short-strangle backtest.

Fetches (cached under RAW_DIR):
  1. delivery prices (daily settlement prices, full history)
  2. PERPETUAL 1h candles (underlying reference for IV inversion / vol estimate)
  3. per Friday: the listed Sunday-expiry option strikes near the money (discovered by
     probing instrument names) + their hourly candles around the 16:00 UTC entry.

Deribit naming: BTC-27AUG23-26000-C (day without leading zero, 3-letter month).
Probing a nonexistent name returns a clean "instrument not found" error, which is how
the historical strike universe is reconstructed (get_instruments?expired=true only
covers the last ~day, and historical trades are purged).

Only depends on requests + the standard library, so it runs in a bare Colab runtime:
  python download_data.py
"""

import csv
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

RAW_DIR = Path(os.environ.get("TRADE_DATA_ROOT", "/Volumes/trade/data")) / "deribit"
API = "https://www.deribit.com/api/v2/public"

START = date(2022, 9, 1)          # Deribit daily expiries exist from ~Sep 2022
END = date(2026, 8, 23)           # last fully-expired Sunday
UNDERLYINGS = {
    "BTC": {"step": 250, "perp": "BTC-PERPETUAL", "index": "btc_usd"},
    "ETH": {"step": 25, "perp": "ETH-PERPETUAL", "index": "eth_usd"},
}
ENTRY_HOUR = 16
CANDLE_WINDOW = range(12, 18)     # fetch Friday 12:00-17:00 candles for fallback

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# global token bucket: ~10 requests/second across all worker threads
_rate_lock = threading.Lock()
_rate_next = 0.0


def call(endpoint: str, retries: int = 6) -> dict:
    """Rate-limited JSON-RPC call with retry on 429/5xx/transient errors."""
    global _rate_next
    url = f"{API}/{endpoint}"
    for attempt in range(retries):
        with _rate_lock:
            wait = _rate_next - time.monotonic()
            _rate_next = max(time.monotonic(), _rate_next) + 0.10
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        try:  # 200 payload, or JSON-RPC error body (e.g. 400 "instrument not found")
            return r.json()
        except ValueError:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            raise
    raise RuntimeError("unreachable")


def ts_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def instrument_name(cur: str, expiry: date, strike: float, side: str) -> str:
    k = int(strike) if strike == int(strike) else strike
    return f"{cur}-{expiry.day}{MONTHS[expiry.month - 1]}{str(expiry.year)[2:]}-{k}-{side}"


# ---------------------------------------------------------------- 1. delivery prices

def fetch_delivery_prices(cur_cfg: dict) -> None:
    out = RAW_DIR / f"delivery_{cur_cfg['index']}.csv"
    if out.exists():
        return
    rows, offset = [], 0
    while True:  # server caps pages at 100 rows regardless of count
        d = call(f"get_delivery_prices?index_name={cur_cfg['index']}&count=100&offset={offset}")["result"]
        rows += d["data"]
        if len(rows) >= d["records_total"] or not d["data"]:
            break
        offset += 100
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 2. perp candles
# perp is a list of (ts: aware datetime, open: float, close: float), sorted by ts.

def fetch_perp_candles(perp: str) -> list[tuple[datetime, float, float]]:
    out = RAW_DIR / f"perp_1h_{perp}.csv"
    if out.exists():
        rows = []
        with out.open() as f:
            for r in csv.DictReader(f):
                rows.append((datetime.fromisoformat(r["ts"]),
                             float(r["open"]), float(r["close"])))
        return rows
    rows, seen = [], set()
    start = datetime(START.year, START.month, START.day, tzinfo=timezone.utc)
    end = datetime(END.year, END.month, END.day, tzinfo=timezone.utc) + timedelta(days=1)
    chunk = timedelta(days=90)
    while start < end:
        stop = min(start + chunk, end)
        d = call(f"get_tradingview_chart_data?instrument_name={perp}"
                 f"&start_timestamp={ts_ms(start)}&end_timestamp={ts_ms(stop)}&resolution=60")["result"]
        if d["status"] == "ok" and d["ticks"]:
            for t, o, c in zip(d["ticks"], d["open"], d["close"]):
                if t not in seen:
                    seen.add(t)
                    rows.append((datetime.fromtimestamp(t / 1000, tz=timezone.utc),
                                 float(o), float(c)))
        start = stop
    rows.sort(key=lambda r: r[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "close"])
        for ts, o, c in rows:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S+00:00"), o, c])
    return rows


# ---------------------------------------------------------------- 3. strikes + candles

def probe_exists(name: str, hint_ms: int) -> bool:
    d = call(f"get_tradingview_chart_data?instrument_name={name}"
             f"&start_timestamp={hint_ms}&end_timestamp={hint_ms + 3600000}&resolution=60")
    return "error" not in d


def fetch_candles(name: str, start_ms: int, end_ms: int) -> dict:
    d = call(f"get_tradingview_chart_data?instrument_name={name}"
             f"&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution=60")
    if "error" in d:
        return {}
    r = d["result"]
    if r["status"] != "ok":
        return {}
    return {"ticks": r["ticks"], "open": r["open"], "close": r["close"], "volume": r["volume"]}


def realized_vol(perp: list, at: datetime) -> float:
    """Annualized vol from trailing 72h of 1h log returns."""
    win = [c for ts, _, c in perp if ts < at][-73:]
    if len(win) < 10:
        return 0.6
    ret = [math.log(win[i] / win[i - 1]) for i in range(1, len(win))]
    sd = statistics.stdev(ret) if len(ret) > 1 else 0.0
    return sd * math.sqrt(24 * 365) or 0.6


def download_week(cur: str, cfg: dict, friday: date, perp: list) -> dict:
    """Discover listed Sunday strikes near the money and fetch their candles."""
    sunday = friday + timedelta(days=2)
    cache = RAW_DIR / "weekly" / cur / f"{friday}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    def save(rec: dict) -> dict:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec))
        return rec

    entry_dt = datetime(friday.year, friday.month, friday.day, ENTRY_HOUR, tzinfo=timezone.utc)
    entry_ms = ts_ms(entry_dt)
    spot_row = [o for ts, o, _ in perp if ts == entry_dt]
    if not spot_row:
        return save({"friday": str(friday), "sunday": str(sunday), "contracts": {}})
    spot = spot_row[0]
    sigma = realized_vol(perp, entry_dt)
    t_years = 40 / (365 * 24)
    band = max(0.12, 3.2 * sigma * math.sqrt(t_years))

    step = cfg["step"]
    lo = int(math.floor(spot * (1 - band) / step) * step)
    hi = int(math.ceil(spot * (1 + band) / step) * step)

    strikes = []
    k = lo
    while k <= hi:
        name = instrument_name(cur, sunday, k, "C")
        if probe_exists(name, entry_ms):
            strikes.append(k)
        k += step
    if not strikes:
        # Sunday expiry not listed this week
        return save({"friday": str(friday), "sunday": str(sunday), "spot": spot, "contracts": {}})

    c0, c1 = ts_ms(datetime(friday.year, friday.month, friday.day, min(CANDLE_WINDOW))), \
             ts_ms(datetime(friday.year, friday.month, friday.day, max(CANDLE_WINDOW) + 1))
    contracts = {}
    for k in strikes:
        for side in ("C", "P"):
            name = instrument_name(cur, sunday, k, side)
            candles = fetch_candles(name, c0, c1)
            if candles:
                contracts[name] = candles

    return save({"friday": str(friday), "sunday": str(sunday), "spot": spot,
                 "sigma_rv": sigma, "contracts": contracts})


def main() -> None:
    fridays = []
    d = START
    while d <= END:
        if d.weekday() == 4 and d + timedelta(days=2) <= END:
            fridays.append(d)
        d += timedelta(days=1)
    print(f"{len(fridays)} candidate Fridays x {len(UNDERLYINGS)} underlyings", flush=True)

    for cur, cfg in UNDERLYINGS.items():
        fetch_delivery_prices(cfg)
        print(f"{cur}: delivery prices cached", flush=True)
        perp = fetch_perp_candles(cfg["perp"])
        print(f"{cur}: {len(perp)} perp candles", flush=True)
        done = failed = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(download_week, cur, cfg, f, perp): f for f in fridays}
            for fut in futures:
                try:
                    fut.result()
                except Exception as e:
                    failed += 1
                    print(f"{cur} {futures[fut]}: FAILED ({type(e).__name__}: {e})", flush=True)
                done += 1
                if done % 25 == 0:
                    print(f"{cur}: {done}/{len(fridays)} weeks ({failed} failed)", flush=True)
        print(f"{cur} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
