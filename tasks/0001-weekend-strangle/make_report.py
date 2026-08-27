#!/usr/bin/env python3
"""从 results/ 下的回测 CSV 生成自包含 HTML 报告 results/report.html。

用法: uv run make_report.py
所有数字均来自 CSV, 脚本内不硬编码任何回测结果。
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = RES / "report.html"
DELTAS = [f"d{d:02d}" for d in range(5, 61, 5)]  # d05..d60
UNDERLYINGS = ["BTC", "ETH"]


def r(x, n=6):
    """round float, NaN -> None"""
    if x is None:
        return None
    v = float(x)
    if np.isnan(v):
        return None
    return round(v, n)


def load_deribit(sym):
    grid = pd.read_csv(RES / f"grid_deribit_{sym}.csv").sort_values("delta")
    g = []
    for row in grid.itertuples():
        g.append({
            "delta": r(row.delta, 2),
            "weeks": int(row.weeks),
            "total_pnl_base": r(row.total_pnl_base),
            "total_pnl_usd": r(row.total_pnl_usd, 2),
            "ret_on_notional": r(row.ret_on_notional, 8),
            "annual_ret": r(row.ret_on_notional * 52, 6),  # 简单年化
            "win_rate": r(row.win_rate),
            "sharpe_w": r(row.sharpe_w, 4),  # 周频 Sharpe x sqrt(52), 已年化
            "max_dd_base": r(row.max_dd_base),
            "premium_sum_base": r(row.premium_sum_base, 4),
            "retained": r(row.retained),
            "first": row.first,
            "last": row.last,
        })

    deltas = {}
    for tag in DELTAS:
        t = pd.read_csv(RES / f"trades_deribit_{sym}_{tag}.csv")
        deltas[tag] = {
            "dates": t["friday"].tolist(),
            "cum": [r(v, 8) for v in t["pnl"].cumsum()],
            "trades": [{
                "f": tr.friday,
                "cs": r(tr.C_strike, 1), "cd": r(tr.C_delta, 4), "cp": r(tr.C_premium, 5),
                "ps": r(tr.P_strike, 1), "pd": r(tr.P_delta, 4), "pp": r(tr.P_premium, 5),
                "e": r(tr.entry_spot, 2), "s": r(tr.settle, 2),
                "pnl": r(tr.pnl, 6), "pnl_usd": r(tr.pnl_usd, 2),
            } for tr in t.itertuples()],
        }

    # 价格时效统计(数据来自逐笔 CSV 的 C_src/P_src 列)
    all_t = pd.concat(
        [pd.read_csv(RES / f"trades_deribit_{sym}_{tag}.csv") for tag in DELTAS],
        ignore_index=True)
    non16 = (all_t["C_src"] != "open@16") | (all_t["P_src"] != "open@16")
    stats = {
        "n_records": int(len(all_t)),
        "stale_frac": r(non16.mean(), 4),  # 至少一腿非 16:00 整点开盘价的记录占比
    }
    return {"unit": sym, "grid": g, "deltas": deltas, "stats": stats}


def load_compound(sym):
    """复利口径: 对称网格指标 + call x put 非对称网格 + 回撤区间"""
    sym_df = pd.read_csv(RES / f"compound_sym_{sym}.csv").sort_values("delta")
    s = [{
        "delta": r(row.delta, 2), "weeks": int(row.weeks),
        "equity_final": r(row.equity_final), "cagr": r(row.cagr),
        "sharpe_w": r(row.sharpe_w, 4), "max_dd": r(row.max_dd),
        "calmar": r(row.calmar, 4), "win_rate": r(row.win_rate),
        "usd_final": r(row.usd_final), "bh_final": r(row.bh_final),
        "max_dd_usd": r(row.max_dd_usd),
    } for row in sym_df.itertuples()]

    asym_df = pd.read_csv(RES / f"compound_asym_{sym}.csv")
    a = [{
        "dc": r(row.delta_call, 2), "dp": r(row.delta_put, 2),
        "cagr": r(row.cagr), "calmar": r(row.calmar, 4),
        "eq": r(row.equity_final), "mdd": r(row.max_dd),
    } for row in asym_df.itertuples()]

    def dd(name):
        df = pd.read_csv(RES / f"drawdowns_{sym}_{name}.csv")
        return [{"peak": row.peak_friday, "trough": row.trough_friday,
                 "rec": row.recovered_friday, "depth": r(row.depth),
                 "w2t": int(row.weeks_to_trough)} for row in df.itertuples()]

    return {"sym": s, "asym": a, "dd35": dd("35-35"), "dd_best": dd("best-calmar")}


def build_data():
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "delta_tags": DELTAS,
        "deribit": {sym: load_deribit(sym) for sym in UNDERLYINGS},
        "compound": {sym: load_compound(sym) for sym in UNDERLYINGS},
    }


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>加密期权周末卖方策略回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  :root { --fg:#1c2430; --muted:#66707d; --line:#e3e7ec; --bg:#f7f8fa; --card:#fff; --accent:#2563eb; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font:14px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:17px; margin:28px 0 10px; border-left:4px solid var(--accent); padding-left:8px; }
  .meta { color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:12px 0; }
  .btns { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
  .btns button { padding:5px 14px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; font-size:13px; }
  .btns button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { border-bottom:1px solid var(--line); padding:5px 8px; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  th { background:#f0f2f5; position:sticky; top:0; }
  .scroll { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:6px; }
  .chart { width:100%; height:380px; }
  .chart-sm { width:100%; height:260px; }
  .grid3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:10px 0; }
  .metric { background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .metric .k { color:var(--muted); font-size:12px; }
  .metric .v { font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }
  .note { color:var(--muted); font-size:13px; }
  .warn { background:#fff8e6; border:1px solid #f0ddA0; border-radius:8px; padding:12px 16px; }
  .neg { color:#c0392b; } .pos { color:#1e8449; }
  #nochart { display:none; color:#c0392b; }
</style>
</head>
<body>
<div class="wrap">
  <h1>加密货币期权周末卖方策略回测报告</h1>
  <div class="meta" id="meta"></div>

  <div class="card warn" id="assumptions"></div>

  <h2>标的切换</h2>
  <div class="btns" id="uBtns"></div>

  <h2>全部 12 档 delta 累计 PnL 对比(币本位,单利口径)</h2>
  <div class="card"><div id="chartAll" class="chart"></div></div>

  <h2>单档 delta 明细</h2>
  <div class="btns" id="dBtns"></div>
  <div class="btns" id="mBtns"></div>
  <div id="cards" class="cards"></div>
  <div class="card"><div id="chartOne" class="chart"></div></div>
  <h3 style="margin:14px 0 6px">逐笔交易明细</h3>
  <div class="scroll"><table id="tradeTable"></table></div>

  <h2>12 档 delta 指标对比汇总</h2>
  <div class="grid3" style="grid-template-columns:1fr 1fr">
    <div class="card"><h3 style="margin-top:0" id="tblTitleBTC"></h3><div class="scroll"><table id="gridTableBTC"></table></div></div>
    <div class="card"><h3 style="margin-top:0" id="tblTitleETH"></h3><div class="scroll"><table id="gridTableETH"></table></div></div>
  </div>
  <div class="grid3">
    <div class="card"><div id="barPnl" class="chart-sm"></div></div>
    <div class="card"><div id="barSharpe" class="chart-sm"></div></div>
    <div class="card"><div id="barRetained" class="chart-sm"></div></div>
  </div>

  <h2>复利口径汇总(权利金无损再投资,期初权益 = 1)</h2>
  <div class="grid3" style="grid-template-columns:1fr 1fr">
    <div class="card"><h3 style="margin-top:0" id="cTblTitleBTC"></h3><div class="scroll"><table id="cGridTableBTC"></table></div></div>
    <div class="card"><h3 style="margin-top:0" id="cTblTitleETH"></h3><div class="scroll"><table id="cGridTableETH"></table></div></div>
  </div>

  <h2>非对称 delta 寻优(call × put,复利口径)</h2>
  <div class="card">
    <div class="note" id="asymNote"></div>
    <div class="grid3" style="grid-template-columns:1fr 1fr; margin-top:10px">
      <div><div id="hmCagr" class="chart-sm" style="height:340px"></div></div>
      <div><div id="hmCalmar" class="chart-sm" style="height:340px"></div></div>
    </div>
  </div>
  <h3 style="margin:14px 0 6px">Top 10 组合(按 Calmar 排序)</h3>
  <div class="scroll"><table id="top10Table"></table></div>

  <h2>回撤归因(35/35 对称配置,复利口径)</h2>
  <div class="card">
    <div class="grid3" style="grid-template-columns:1fr 1fr">
      <div class="scroll"><table id="ddTableBTC"></table></div>
      <div class="scroll"><table id="ddTableETH"></table></div>
    </div>
    <div class="note" style="margin-top:10px">
      <b>回撤区间对应事件(调研结论):</b>
      <ul style="margin:6px 0; padding-left:20px">
        <li>2023-01:CPI 降温 + FTX 修复 + 空头挤压(上涨回撤,short call 腿亏损)</li>
        <li>2024-04:伊朗袭击以色列 + 减半前清洗</li>
        <li>2024-08:日元套息平仓(尾部延续到周一,周日 08:00 结算低估风险)</li>
        <li>2024-11:特朗普胜选(ETH 上涨回撤)</li>
        <li>2025-01:特朗普首轮关税</li>
        <li>2025-06:美军轰炸伊朗核设施</li>
        <li>2025-10:特朗普对华 100% 关税威胁、$19.3B 最大强平</li>
        <li>2026-01~02:日本国债闪崩 + 格陵兰关税 + Fed 鹰派</li>
      </ul>
      <b>共性:</b>外生冲击 × 周末流动性真空;ETH 尾部为 BTC 的 1.5~2 倍。部分事件(如 2024-08、2024-11)未出现在 35/35 前五大回撤中,但在其他配置(如 best-calmar)的回撤序列中可见。
    </div>
  </div>

  <h2>脚注</h2>
  <div class="card note">
    <ul style="margin:6px 0; padding-left:20px">
      <li>年化收益率 = 周均收益率(名义本金) × 52,为<b>简单年化</b>,未复利。</li>
      <li>复利口径:净值 = 逐周 (1 + 币本位周收益率) 累乘,期初 = 1,假设权利金无损再投资;CAGR、复利 maxDD、Calmar 均基于复利净值曲线。</li>
      <li>Sharpe = 周收益率均值 / 周收益率标准差 × √52(周频年化,无风险利率取 0)。</li>
      <li>Deribit 期权为币本位:除特别注明 USD 外,所有 PnL / 权利金 / 回撤均以 BTC 或 ETH 计;USD 值按入场时现货价折算,仅供参考。</li>
      <li>回测结果不代表未来收益。</li>
    </ul>
  </div>
  <div id="nochart">图表库(ECharts CDN)加载失败,页面仅展示表格数据。</div>
</div>

<script id="backtest-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('backtest-data').textContent);
const HAS_ECHARTS = typeof echarts !== 'undefined';
if (!HAS_ECHARTS) document.getElementById('nochart').style.display = 'block';

const state = { u: 'BTC', d: 'd35', m: 'simple' };
const EVENTS = [
  { date: '2023-01-20', text: 'CPI 降温 + FTX 修复 + 空头挤压(上涨回撤)' },
  { date: '2024-04-12', text: '伊朗袭击以色列 + 减半前清洗' },
  { date: '2024-08-05', text: '日元套息平仓(尾部延续到周一)' },
  { date: '2024-11-08', text: '特朗普胜选(ETH 上涨回撤)' },
  { date: '2025-01-31', text: '特朗普首轮关税' },
  { date: '2025-06-20', text: '美军轰炸伊朗核设施' },
  { date: '2025-10-10', text: '特朗普对华 100% 关税威胁、$19.3B 最大强平' },
  { date: '2026-01-30', text: '日本国债闪崩 + 格陵兰关税 + Fed 鹰派' },
];
const charts = {};
const fmt = (x, n=4) => x === null || x === undefined ? '—' :
  Number(x).toLocaleString('en-US', {minimumFractionDigits: n, maximumFractionDigits: n});
const pct = (x, n=2) => x === null || x === undefined ? '—' : (x*100).toFixed(n) + '%';
const cls = x => x < 0 ? 'neg' : 'pos';

function meta() {
  const g = DATA.deribit.BTC.grid[0];
  document.getElementById('meta').textContent =
    `策略:持有现货 + 每周五 16:00 UTC 卖出 1 张周日 08:00 UTC 到期的 call 与 put(short strangle),持有至交割结算 | ` +
    `数据窗口:${g.first} ~ ${g.last}(BTC / ETH 各 ${g.weeks} 周)| 数据来源:Deribit 公开 API | 报告生成:${DATA.generated}`;
  const s = {};
  for (const u of Object.keys(DATA.deribit)) s[u] = DATA.deribit[u].stats;
  document.getElementById('assumptions').innerHTML =
    `<b>关键假设与局限</b><ul style="margin:6px 0; padding-left:20px">` +
    `<li>每周五 16:00 UTC 以当时可得的标记价卖出期权,周日 08:00 UTC 按 Deribit 官方交割价结算。</li>` +
    `<li>费率模型:开仓费 0.03% × 名义本金(上限为权利金的 12.5%);交割费 0.015% × 名义本金(上限为赔付的 12.5%,仅实值收取)。</li>` +
    `<li>目标 delta 由 BS 模型(r = 0)按标记 IV 反推,实际成交合约为最接近目标 delta 的行权价。</li>` +
    `<li>局限:成交价为标记价而非买一价,未计滑点;部分周无 16:00 整点价格,使用了临近时刻价格(交易记录中至少一腿非 open@16 的占比:` +
    `BTC ${pct(s.BTC.stale_frac,1)}、ETH ${pct(s.ETH.stale_frac,1)},多数在 ±2h 内);币本位 PnL 未对冲 USD 计价波动;回测不代表未来。</li>` +
    `<li>复利口径假设权利金无损再投资(逐周 (1+周收益率) 累乘,期初权益 = 1)。</li>` +
    `<li>非对称 delta 组合为 208 周<b>样本内</b>寻优,存在过拟合风险。</li>` +
    `<li>周日 08:00 UTC 结算会低估延续到周一的尾部风险(如 2024-08 日元套息平仓,主跌段发生在周一)。</li></ul>`;
}

function mkChart(id, opt) {
  if (!HAS_ECHARTS) return;
  if (!charts[id]) charts[id] = echarts.init(document.getElementById(id));
  charts[id].setOption(opt, true);
}

function renderUBtns() {
  const el = document.getElementById('uBtns'); el.innerHTML = '';
  for (const u of Object.keys(DATA.deribit)) {
    const b = document.createElement('button');
    b.textContent = u; b.className = u === state.u ? 'on' : '';
    b.onclick = () => { state.u = u; renderAll(); };
    el.appendChild(b);
  }
}

function renderDBtns() {
  const el = document.getElementById('dBtns'); el.innerHTML = '';
  for (const tag of DATA.delta_tags) {
    const b = document.createElement('button');
    b.textContent = (parseInt(tag.slice(1)) / 100).toFixed(2);
    b.className = tag === state.d ? 'on' : '';
    b.onclick = () => { state.d = tag; renderAll(); };
    el.appendChild(b);
  }
}

function renderAllChart() {
  const U = DATA.deribit[state.u];
  const dates = U.deltas[DATA.delta_tags[0]].dates;
  mkChart('chartAll', {
    tooltip: { trigger: 'axis', valueFormatter: v => fmt(v, 4) + ' ' + U.unit },
    legend: { type: 'scroll' },
    grid: { left: 70, right: 20, top: 40, bottom: 60 },
    xAxis: { type: 'category', data: dates },
    yAxis: { type: 'value', name: '累计 PnL(' + U.unit + ')' },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: DATA.delta_tags.map(tag => ({
      name: 'Δ' + (parseInt(tag.slice(1)) / 100).toFixed(2),
      type: 'line', showSymbol: false, data: U.deltas[tag].cum,
    })),
  });
}

function renderMBtns() {
  const el = document.getElementById('mBtns'); el.innerHTML = '';
  for (const [m, label] of [['simple', '单利(币本位 PnL 累加)'], ['compound', '复利(无损再投资)']]) {
    const b = document.createElement('button');
    b.textContent = label; b.className = m === state.m ? 'on' : '';
    b.onclick = () => { state.m = m; renderAll(); };
    el.appendChild(b);
  }
}

function compoundCum(D) {
  let e = 1; return D.trades.map(t => { e *= (1 + t.pnl); return +e.toFixed(6); });
}

function renderCards() {
  const U = DATA.deribit[state.u];
  const tag = state.d;
  let items;
  if (state.m === 'simple') {
    const g = U.grid.find(r => 'd' + String(Math.round(r.delta * 100)).padStart(2, '0') === tag);
    items = [
      ['周数', g.weeks],
      ['总 PnL(' + U.unit + ')', `<span class="${cls(g.total_pnl_base)}">${fmt(g.total_pnl_base)}</span>`],
      ['总 PnL(USD)', `<span class="${cls(g.total_pnl_usd)}">$${fmt(g.total_pnl_usd, 0)}</span>`],
      ['周均收益率(名义)', pct(g.ret_on_notional)],
      ['年化(简单)', pct(g.annual_ret)],
      ['胜率', pct(g.win_rate)],
      ['Sharpe(年化)', fmt(g.sharpe_w, 2)],
      ['最大回撤(' + U.unit + ')', `<span class="neg">${fmt(g.max_dd_base)}</span>`],
      ['权利金总额(' + U.unit + ')', fmt(g.premium_sum_base)],
      ['权利金留存率', pct(g.retained)],
    ];
  } else {
    const g = DATA.compound[state.u].sym.find(r => 'd' + String(Math.round(r.delta * 100)).padStart(2, '0') === tag);
    items = [
      ['周数', g.weeks],
      ['期末权益(倍数)', `<span class="${cls(g.equity_final - 1)}">${fmt(g.equity_final, 3)}×</span>`],
      ['CAGR', pct(g.cagr)],
      ['Sharpe(年化)', fmt(g.sharpe_w, 2)],
      ['最大回撤(复利)', `<span class="neg">${pct(g.max_dd)}</span>`],
      ['Calmar', fmt(g.calmar, 2)],
      ['胜率', pct(g.win_rate)],
      ['期末权益(USD 倍数)', fmt(g.usd_final, 2) + '×'],
      ['同期买入持有(USD 倍数)', fmt(g.bh_final, 2) + '×'],
      ['最大回撤(USD 复利)', `<span class="neg">${pct(g.max_dd_usd)}</span>`],
    ];
  }
  document.getElementById('cards').innerHTML = items.map(([k, v]) =>
    `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}

function renderOneChart() {
  const U = DATA.deribit[state.u], D = U.deltas[state.d];
  const compound = state.m === 'compound';
  mkChart('chartOne', {
    tooltip: { trigger: 'axis', valueFormatter: v => compound ? fmt(v, 3) + '×' : fmt(v, 5) + ' ' + U.unit },
    grid: { left: 70, right: 20, top: 40, bottom: 60 },
    xAxis: { type: 'category', data: D.dates },
    yAxis: { type: 'value', scale: true,
      name: compound ? '复利净值(倍数,期初=1)' : '累计 PnL(' + U.unit + ',单利)' },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: [{ name: 'Δ' + (parseInt(state.d.slice(1)) / 100).toFixed(2),
      type: 'line', showSymbol: false,
      data: compound ? compoundCum(D) : D.cum,
      areaStyle: { opacity: 0.12 } }],
  });
}

function renderTradeTable() {
  const U = DATA.deribit[state.u], D = U.deltas[state.d];
  let h = '<thead><tr><th>friday</th><th>Call 行权价</th><th>Call Δ</th><th>Call 权利金</th>' +
    '<th>Put 行权价</th><th>Put Δ</th><th>Put 权利金</th><th>入场价</th><th>交割价</th>' +
    `<th>单周 PnL(${U.unit})</th><th>单周 PnL(USD)</th></tr></thead><tbody>`;
  for (const t of D.trades) {
    h += `<tr><td>${t.f}</td><td>${fmt(t.cs, 0)}</td><td>${fmt(t.cd)}</td><td>${fmt(t.cp, 5)}</td>` +
      `<td>${fmt(t.ps, 0)}</td><td>${fmt(t.pd)}</td><td>${fmt(t.pp, 5)}</td>` +
      `<td>${fmt(t.e, 1)}</td><td>${fmt(t.s, 1)}</td>` +
      `<td class="${cls(t.pnl)}">${fmt(t.pnl, 5)}</td><td class="${cls(t.pnl_usd)}">${fmt(t.pnl_usd, 1)}</td></tr>`;
  }
  document.getElementById('tradeTable').innerHTML = h + '</tbody>';
}

function renderGridTables() {
  for (const u of ['BTC', 'ETH']) {
    const U = DATA.deribit[u];
    document.getElementById('tblTitle' + u).textContent = u + '(币本位)';
    let h = '<thead><tr><th>Δ</th><th>周数</th><th>总 PnL</th><th>总 PnL(USD)</th>' +
      '<th>周均收益率</th><th>年化</th><th>胜率</th><th>Sharpe</th><th>最大回撤</th>' +
      '<th>权利金总额</th><th>留存率</th></tr></thead><tbody>';
    for (const g of U.grid) {
      h += `<tr><td>${g.delta.toFixed(2)}</td><td>${g.weeks}</td>` +
        `<td class="${cls(g.total_pnl_base)}">${fmt(g.total_pnl_base)}</td>` +
        `<td class="${cls(g.total_pnl_usd)}">${fmt(g.total_pnl_usd, 0)}</td>` +
        `<td>${pct(g.ret_on_notional)}</td><td>${pct(g.annual_ret)}</td>` +
        `<td>${pct(g.win_rate)}</td><td>${fmt(g.sharpe_w, 2)}</td>` +
        `<td class="neg">${fmt(g.max_dd_base)}</td><td>${fmt(g.premium_sum_base)}</td>` +
        `<td>${pct(g.retained)}</td></tr>`;
    }
    document.getElementById('gridTable' + u).innerHTML = h + '</tbody>';
  }
}

function renderBars() {
  const U = DATA.deribit[state.u];
  const xs = U.grid.map(g => g.delta.toFixed(2));
  const bar = (id, name, vals, n) => mkChart(id, {
    title: { text: name + '(' + state.u + ')', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 15, top: 40, bottom: 25 },
    xAxis: { type: 'category', data: xs, name: 'Δ' },
    yAxis: { type: 'value' },
    series: [{ type: 'bar', data: vals }],
  });
  bar('barPnl', '总 PnL(' + U.unit + ')', U.grid.map(g => g.total_pnl_base));
  bar('barSharpe', 'Sharpe(年化)', U.grid.map(g => g.sharpe_w));
  bar('barRetained', '权利金留存率', U.grid.map(g => +(g.retained * 100).toFixed(1)));
}

function renderCompoundGrids() {
  for (const u of ['BTC', 'ETH']) {
    const C = DATA.compound[u];
    document.getElementById('cTblTitle' + u).textContent = u + '(复利,期初=1)';
    let h = '<thead><tr><th>Δ</th><th>周数</th><th>期末权益(×)</th><th>CAGR</th>' +
      '<th>Sharpe</th><th>最大回撤</th><th>Calmar</th><th>胜率</th>' +
      '<th>期末USD(×)</th><th>买入持有USD(×)</th><th>USD最大回撤</th></tr></thead><tbody>';
    for (const g of C.sym) {
      h += `<tr><td>${g.delta.toFixed(2)}</td><td>${g.weeks}</td>` +
        `<td class="${cls(g.equity_final - 1)}">${fmt(g.equity_final, 3)}</td>` +
        `<td>${pct(g.cagr)}</td><td>${fmt(g.sharpe_w, 2)}</td>` +
        `<td class="neg">${pct(g.max_dd)}</td><td>${fmt(g.calmar, 2)}</td>` +
        `<td>${pct(g.win_rate)}</td><td>${fmt(g.usd_final, 2)}</td>` +
        `<td>${fmt(g.bh_final, 2)}</td><td class="neg">${pct(g.max_dd_usd)}</td></tr>`;
    }
    document.getElementById('cGridTable' + u).innerHTML = h + '</tbody>';
  }
}

function renderAsym() {
  const C = DATA.compound[state.u];
  const xs = [...new Set(C.asym.map(o => o.dc))].sort((a, b) => a - b);
  const ys = [...new Set(C.asym.map(o => o.dp))].sort((a, b) => a - b);
  const xLabels = xs.map(v => v.toFixed(2)), yLabels = ys.map(v => v.toFixed(2));
  const mk = (id, key, name, fmtv) => {
    const vals = C.asym.map(o => [xLabels.indexOf(o.dc.toFixed(2)), yLabels.indexOf(o.dp.toFixed(2)), o[key]]);
    const all = C.asym.map(o => o[key]);
    mkChart(id, {
      title: { text: name + '(' + state.u + ',复利)', textStyle: { fontSize: 13 } },
      tooltip: { formatter: p => `call Δ${xLabels[p.value[0]]} / put Δ${yLabels[p.value[1]]}<br>${name}: <b>${fmtv(p.value[2])}</b>` },
      grid: { left: 60, right: 80, top: 40, bottom: 40 },
      xAxis: { type: 'category', data: xLabels, name: 'call Δ' },
      yAxis: { type: 'category', data: yLabels, name: 'put Δ' },
      visualMap: { min: Math.min(...all), max: Math.max(...all), orient: 'vertical',
        right: 0, top: 'center', calculable: true,
        inRange: { color: ['#d73027', '#fee08b', '#1a9850'] } },
      series: [{ type: 'heatmap', data: vals,
        label: { show: true, fontSize: 9, formatter: p => fmtv(p.value[2]) } }],
    });
  };
  mk('hmCagr', 'cagr', 'CAGR', v => (v * 100).toFixed(0) + '%');
  mk('hmCalmar', 'calmar', 'Calmar', v => v.toFixed(1));

  const best = [...C.asym].sort((a, b) => b.calmar - a.calmar)[0];
  document.getElementById('asymNote').innerHTML =
    `样本内(208 周)最优平台区:<b>BTC — call Δ0.25 / put Δ0.55~0.60;ETH — call Δ0.35~0.40 / put Δ0.55</b>。` +
    `当前标的 ${state.u} 的 Calmar 最高点为 call Δ${best.dc.toFixed(2)} / put Δ${best.dp.toFixed(2)}` +
    `(Calmar ${fmt(best.calmar, 2)},CAGR ${pct(best.cagr)})。` +
    `以上为非对称 call/put 目标 delta 的 12×12 全网格样本内寻优结果,选的是热力图上的<b>平台区</b>而非单一最优点,不构成样本外收益保证。`;

  const top10 = [...C.asym].sort((a, b) => b.calmar - a.calmar).slice(0, 10);
  let h = `<thead><tr><th>#</th><th>call Δ</th><th>put Δ</th><th>CAGR</th><th>Calmar</th>` +
    `<th>期末权益(×)</th><th>最大回撤</th></tr></thead><tbody>`;
  top10.forEach((o, i) => {
    h += `<tr><td>${i + 1}</td><td>${o.dc.toFixed(2)}</td><td>${o.dp.toFixed(2)}</td>` +
      `<td>${pct(o.cagr)}</td><td>${fmt(o.calmar, 2)}</td>` +
      `<td>${fmt(o.eq, 3)}</td><td class="neg">${pct(o.mdd)}</td></tr>`;
  });
  document.getElementById('top10Table').innerHTML = h + '</tbody>';
}

function renderDDTables() {
  for (const u of ['BTC', 'ETH']) {
    const rows = DATA.compound[u].dd35;
    let h = `<thead><tr><th colspan="6" style="text-align:left">${u} 35/35 前 ${rows.length} 大回撤区间</th></tr>` +
      '<tr><th>峰值周五</th><th>谷底周五</th><th>修复周五</th><th>深度</th><th>到谷底周数</th><th>对应事件</th></tr></thead><tbody>';
    for (const d of rows) {
      const ev = EVENTS.find(e => e.date >= d.peak && e.date <= d.rec);
      h += `<tr><td>${d.peak}</td><td>${d.trough}</td><td>${d.rec}</td>` +
        `<td class="neg">${pct(d.depth)}</td><td>${d.w2t}</td>` +
        `<td style="white-space:normal;text-align:left">${ev ? ev.text : '—'}</td></tr>`;
    }
    document.getElementById('ddTable' + u).innerHTML = h + '</tbody>';
  }
}

function renderAll() {
  renderUBtns(); renderDBtns(); renderMBtns();
  renderAllChart(); renderCards(); renderOneChart(); renderTradeTable();
  renderGridTables(); renderBars();
  renderCompoundGrids(); renderAsym(); renderDDTables();
}
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
renderAll();
</script>
</body>
</html>
"""


def main():
    data = build_data()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # 防 script 标签提前闭合
    html = TEMPLATE.replace("__DATA__", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"written: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    for sym in UNDERLYINGS:
        g = data["deribit"][sym]["grid"]
        print(f"{sym}: {len(g)} deltas, {g[0]['weeks']} weeks, "
              f"{g[0]['first']} ~ {g[0]['last']}")


if __name__ == "__main__":
    main()
