import os
import json
import uuid
import numpy as np
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import timedelta, date, datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

def _yf(ticker: str) -> yf.Ticker:
    """Return a yfinance Ticker with a browser-like session to avoid cloud rate limits."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return yf.Ticker(ticker, session=s)

# ─── Strategy Persistence ──────────────────────────────────────────────────────

STRATEGIES_FILE = "strategies.json"

def _load_strategies() -> list:
    if not os.path.exists(STRATEGIES_FILE):
        return []
    try:
        with open(STRATEGIES_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save_strategies(strategies: list):
    with open(STRATEGIES_FILE, "w") as f:
        json.dump(strategies, f, indent=2, default=str)

def add_strategy(strategy: dict):
    strategies = _load_strategies()
    strategies.insert(0, strategy)          # newest first
    _save_strategies(strategies)

def remove_strategy(sid: str):
    strategies = [s for s in _load_strategies() if s["id"] != sid]
    _save_strategies(strategies)

st.set_page_config(page_title="Recommendation Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Recommendation Analyzer")
st.markdown("Upload your recommendations CSV and see if trades hit the profit target or stop loss first.")

# ─── Apply pending strategy load BEFORE any widgets are created ───────────────
if "pending_load" in st.session_state:
    p = st.session_state.pop("pending_load")
    st.session_state["profit_pct"]      = p["profit_pct"]
    st.session_state["loss_pct"]        = p["loss_pct"]
    st.session_state["position_pct"]    = p["position_pct"]
    st.session_state["initial_capital"] = p["initial_capital"]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Recommendations CSV", type=["csv"])
    st.divider()
    st.subheader("Trade Parameters")
    profit_pct = st.slider(
        "Profit Target (%)", 1.0, 20.0, 4.0, 0.5,
        key="profit_pct",
        help="How much % gain before taking profit"
    )
    loss_pct = st.slider(
        "Stop Loss (%)", 1.0, 20.0, 4.0, 0.5,
        key="loss_pct",
        help="How much % loss before stopping out"
    )
    st.divider()
    st.subheader("Session Window")
    extended_hours = st.checkbox(
        "Include extended hours",
        key="extended_hours",
        help=(
            "Track after-hours (4–8 PM) + pre-market (4–9:30 AM) in addition to the regular session.\n\n"
            "Window: 4 PM on recommendation day → 4 PM on next trading day.\n\n"
            "⚠️ Extended data is available only for the past ~60 days at 5-min resolution. "
            "Older dates will show as 'No ext. data'."
        ),
    )
    st.divider()
    st.subheader("Portfolio Simulation")
    initial_capital = st.number_input(
        "Starting Capital ($)", min_value=100, max_value=10_000_000,
        value=5000, step=500,
        key="initial_capital",
    )
    position_pct = st.slider(
        "Position Size (% of portfolio per trade)", 1, 100, 20, 1,
        key="position_pct",
        help="How much of your current portfolio you put into each trade"
    )
    st.divider()
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    st.caption(
        "First click fetches data from Yahoo Finance.\n"
        "After that, adjusting the sliders re-analyzes instantly.\n"
        "Toggling extended hours requires clicking Analyze again."
    )

    # ── Saved Strategies Panel ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Saved Strategies")
    strategies = _load_strategies()
    if not strategies:
        st.caption("No saved strategies yet.\nRun an analysis and click 'Save Strategy'.")
    for s in strategies:
        with st.container(border=True):
            st.markdown(f"**{s['name']}**")
            if s.get("description"):
                st.caption(s["description"])
            st.caption(
                f"🎯 TP {s['profit_pct']}% / SL {s['loss_pct']}%  |  "
                f"📊 {s.get('n_trades', '?')} trades  |  "
                f"🏆 Win {s.get('win_rate', 0):.0f}%"
            )
            if s.get("total_return_a") is not None:
                st.caption(
                    f"Sim A: **{s['total_return_a']:+.1f}%**  |  "
                    f"Sim B: **{s['total_return_b']:+.1f}%**"
                )
            btn_col, del_col = st.columns([3, 1])
            if btn_col.button("⬆ Load", key=f"load_{s['id']}", use_container_width=True):
                st.session_state["pending_load"] = {
                    "profit_pct":      float(s["profit_pct"]),
                    "loss_pct":        float(s["loss_pct"]),
                    "position_pct":    int(s["position_pct"]),
                    "initial_capital": float(s["initial_capital"]),
                }
                st.rerun()
            if del_col.button("🗑", key=f"del_{s['id']}", use_container_width=True, help="Delete"):
                remove_strategy(s["id"])
                st.rerun()


# ─── Data Helpers ──────────────────────────────────────────────────────────────

def parse_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw.decode("utf-8")), sep=";", quotechar='"')
    df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date
    return df


@st.cache_data(show_spinner=False)
def get_trading_days(ticker: str, start: date, end: date) -> list[date]:
    """Return NYSE trading days in [start, end] — no API call needed."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start, end_date=end)
    return [d.date() for d in sched.index]


@st.cache_data(show_spinner=False)
def load_local_data(ticker: str) -> pd.DataFrame | None:
    """Load the locally saved Alpaca 1-min file if it exists."""
    path = f"data/{ticker}_1min.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df


def get_extended_window_local(
    local_df: pd.DataFrame,
    rec_date: date,
    next_day: date,
) -> tuple[pd.DataFrame, str]:
    """Slice 4 PM rec_date → 4 PM next_day from the local Alpaca dataset."""
    start_dt = pd.Timestamp(rec_date, tz="America/New_York").replace(hour=16, minute=0, second=0)
    end_dt   = pd.Timestamp(next_day,  tz="America/New_York").replace(hour=16, minute=0, second=0)
    # Use strictly < end_dt so we don't include the 16:00 after-hours bar of next_day
    window = local_df[(local_df["timestamp"] > start_dt) & (local_df["timestamp"] < end_dt)].copy()
    if window.empty:
        return pd.DataFrame(), "no local data"
    window = window.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    return window.set_index("timestamp"), "1m+ext"


@st.cache_data(show_spinner=False)
def fetch_extended_window(ticker: str, rec_date: date, next_day: date) -> tuple[pd.DataFrame, str]:
    """
    Fetch data from 4 PM on rec_date through 4 PM on next_day.
    Uses prepost=True to include after-hours and pre-market bars.
    Falls back gracefully if extended data is unavailable.
    """
    days_ago = (date.today() - rec_date).days

    if days_ago <= 7:
        res_order = ["1m", "5m", "1h"]
    elif days_ago <= 59:
        res_order = ["5m", "1h"]
    else:
        return pd.DataFrame(), "no ext. data (>60 days)"

    t = _yf(ticker)
    for res in res_order:
        try:
            data = t.history(
                start=rec_date,
                end=next_day + timedelta(days=1),
                interval=res,
                prepost=True,
                auto_adjust=True,
            )
            if data.empty:
                continue

            if data.index.tz is not None:
                data.index = data.index.tz_convert("America/New_York")
            else:
                data.index = data.index.tz_localize("UTC").tz_convert("America/New_York")

            # Keep only: strictly after 4 PM on rec_date through 4 PM on next_day
            start_dt = pd.Timestamp(rec_date, tz="America/New_York").replace(hour=16, minute=0, second=0)
            end_dt   = pd.Timestamp(next_day,  tz="America/New_York").replace(hour=16, minute=0, second=0)
            data = data[(data.index > start_dt) & (data.index < end_dt)]

            if not data.empty:
                return data, f"{res}+ext"
        except Exception:
            continue

    return pd.DataFrame(), "no ext. data"


@st.cache_data(show_spinner=False)
def fetch_day_data(ticker: str, target: date) -> tuple[pd.DataFrame, str]:
    """
    Fetch the best available intraday data for a single trading day.
    Tries higher resolution first; falls back as needed.
    Returns (dataframe, resolution_string).
    """
    days_ago = (date.today() - target).days

    if days_ago <= 7:
        res_order = ["1m", "5m", "1h", "1d"]
    elif days_ago <= 59:
        res_order = ["5m", "1h", "1d"]
    elif days_ago <= 729:
        res_order = ["1h", "1d"]
    else:
        res_order = ["1d"]

    t = _yf(ticker)
    for res in res_order:
        try:
            data = t.history(
                start=target,
                end=target + timedelta(days=1),
                interval=res,
                prepost=False,
                auto_adjust=True,
            )
            if data.empty:
                continue

            if res != "1d":
                # Normalize to Eastern Time and keep regular hours only
                if data.index.tz is not None:
                    data.index = data.index.tz_convert("America/New_York")
                else:
                    data.index = data.index.tz_localize("UTC").tz_convert("America/New_York")
                data = data.between_time("09:30", "16:00")

            if not data.empty:
                return data, res
        except Exception:
            continue

    return pd.DataFrame(), "none"


@st.cache_data(show_spinner=False)
def fetch_market_regimes(start: date, end: date) -> pd.DataFrame:
    """Returns DataFrame indexed by date with columns: spy_ret (%), vix_close."""
    spy = _yf("SPY").history(start=start, end=end + timedelta(days=2), interval="1d", auto_adjust=True)
    vix = _yf("^VIX").history(start=start, end=end + timedelta(days=2), interval="1d", auto_adjust=True)
    for _h in [spy, vix]:
        if _h.index.tz is not None:
            _h.index = _h.index.tz_localize(None)
    spy_ret = (spy["Close"].pct_change() * 100).rename("spy_ret")
    vix_cls = vix["Close"].rename("vix")
    out = pd.concat([spy_ret, vix_cls], axis=1)
    out.index = pd.to_datetime(out.index).date
    return out


@st.cache_data(show_spinner=False)
def fetch_buy_hold(ticker: str, start: date, end: date, initial_capital: float):
    """Daily close prices scaled to initial_capital from start date."""
    hist = _yf(ticker).history(
        start=start, end=end + timedelta(days=2), interval="1d", auto_adjust=True
    )
    if hist.empty:
        return None
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist.index = pd.to_datetime(hist.index)
    base = hist["Close"].iloc[0]
    return pd.DataFrame({"date": hist.index, "portfolio": hist["Close"] / base * initial_capital})


# ─── Analysis Logic ────────────────────────────────────────────────────────────

def analyze_trade(
    entry: float,
    data: pd.DataFrame,
    res: str,
    profit_pct: float,
    loss_pct: float,
) -> tuple[str, float | None, str]:
    """
    Determine the outcome of a trade.
    Returns (outcome, exit_price, note).
    outcome: 'success' | 'failure' | 'neutral' | 'ambiguous' | 'no_data'
    """
    if data.empty:
        return "no_data", None, "No data available"

    pt = entry * (1 + profit_pct / 100)  # profit target price
    lt = entry * (1 - loss_pct / 100)    # loss target price

    # Normalize resolution: "1m+ext", "regular (1d) — ...", "1h" → base like "1m","1d","1h"
    base_res = res.split("+")[0].strip()          # "1m+ext" → "1m"
    if base_res.startswith("regular") or base_res.startswith("no"):
        # e.g. "regular (1d) — no ext. data" → extract the part inside parens
        import re
        m = re.search(r"\((\w+)\)", res)
        base_res = m.group(1) if m else "1d"      # default to daily if we can't parse

    # If the first bar already opened beyond a threshold, exit immediately at that price.
    # In regular session this is an overnight gap; in extended hours it's an immediate post-close move.
    open_p = data["Open"].iloc[0]
    is_ext = res.endswith("+ext")
    gap_label = "Immediate post-close move" if is_ext else "Overnight gap"
    if open_p >= pt:
        return "success", open_p, f"{gap_label}: opened at {open_p:.2f}"
    if open_p <= lt:
        return "failure", open_p, f"{gap_label}: opened at {open_p:.2f}"

    if base_res == "1d":
        # With daily bars we know High and Low but not which came first
        h, l = data["High"].iloc[0], data["Low"].iloc[0]
        if h >= pt and l <= lt:
            return "ambiguous", data["Close"].iloc[0], "Both thresholds reached (order unknown — daily data only)"
        if h >= pt:
            return "success", pt, None
        if l <= lt:
            return "failure", lt, None
        return "neutral", data["Close"].iloc[0], None

    # Intraday: walk bar by bar to find which threshold is hit first
    for _, bar in data.iterrows():
        hit_p = bar["High"] >= pt
        hit_l = bar["Low"] <= lt
        if hit_p and hit_l:
            # Both in the same bar — use bar's open direction as tiebreaker
            if bar["Open"] >= entry:
                return "success", pt, "Both in same bar (approx.)"
            return "failure", lt, "Both in same bar (approx.)"
        if hit_p:
            return "success", pt, None
        if hit_l:
            return "failure", lt, None

    return "neutral", data["Close"].iloc[-1], None


def run_analysis(
    buy_recs: pd.DataFrame,
    trading_days: list[date],
    profit_pct: float,
    loss_pct: float,
    extended_hours: bool = False,
) -> pd.DataFrame:
    ticker = buy_recs["symbol"].iloc[0]
    today = date.today()
    local_df = load_local_data(ticker) if extended_hours else None
    rows = []

    for _, rec in buy_recs.iterrows():
        rec_date = rec["date_only"]
        entry = float(rec["close"])
        next_day = next((d for d in trading_days if d > rec_date), None)

        if next_day is None or next_day > today:
            rows.append(dict(
                rec_date=rec_date, entry=entry, next_day=next_day,
                outcome="future", exit_price=None, pnl_pct=None,
                next_open_pnl_pct=None,
                note="Future date — no data yet", resolution=None,
            ))
            continue

        if extended_hours:
            if local_df is not None:
                # Use local Alpaca file (full history, 1-min resolution)
                data, res = get_extended_window_local(local_df, rec_date, next_day)
            else:
                # Local file missing — try live API
                data, res = fetch_extended_window(ticker, rec_date, next_day)
            # If still empty, fall back to regular session
            if data.empty:
                data, res = fetch_day_data(ticker, next_day)
        else:
            data, res = fetch_day_data(ticker, next_day)

        outcome, exit_price, note = analyze_trade(entry, data, res, profit_pct, loss_pct)
        pnl = (exit_price - entry) / entry * 100 if exit_price is not None else None

        # For neutral trades: fetch the open of the next trading day after next_day
        next_open_pnl = None
        if outcome == "neutral":
            day_after = next((d for d in trading_days if d > next_day), None)
            if day_after and day_after <= today:
                next_open_data, _ = fetch_day_data(ticker, day_after)
                if not next_open_data.empty:
                    next_open_p = next_open_data["Open"].iloc[0]
                    next_open_pnl = (next_open_p - entry) / entry * 100

        rows.append(dict(
            rec_date=rec_date, entry=entry, next_day=next_day,
            outcome=outcome, exit_price=exit_price, pnl_pct=pnl,
            next_open_pnl_pct=next_open_pnl,
            note=note or "", resolution=res,
        ))

    return pd.DataFrame(rows)


# ─── Parameter Optimization ───────────────────────────────────────────────────

OPT_VALUES = [round(x * 0.5, 1) for x in range(1, 21)]  # 0.5 … 10.0


def preload_trade_data(buy_recs, trading_days, extended, local_df, ticker):
    """Return {rec_date: (entry, DataFrame, resolution)} for every analyzable trade."""
    today = date.today()
    out = {}
    for _, rec in buy_recs.iterrows():
        rec_date = rec["date_only"]
        entry    = float(rec["close"])
        next_day = next((d for d in trading_days if d > rec_date), None)
        if next_day is None or next_day > today:
            continue
        if extended and local_df is not None:
            data, res = get_extended_window_local(local_df, rec_date, next_day)
        elif extended:
            data, res = fetch_extended_window(ticker, rec_date, next_day)
        else:
            data, res = fetch_day_data(ticker, next_day)
        if data.empty and extended:
            data, res = fetch_day_data(ticker, next_day)
        if not data.empty:
            # Pre-fetch next-day open for Sim A (neutral → exit at next open)
            day_after = next((d for d in trading_days if d > next_day), None)
            next_open_pnl = None
            if day_after and day_after <= today:
                next_open_data, _ = fetch_day_data(ticker, day_after)
                if not next_open_data.empty:
                    next_open_p = next_open_data["Open"].iloc[0]
                    next_open_pnl = (next_open_p - entry) / entry * 100
            out[rec_date] = (entry, data, res, next_open_pnl)
    return out


def run_optimization(trade_data, initial_capital, position_pct):
    """
    Grid-search all (profit_pct, loss_pct) pairs in OPT_VALUES.
    Returns two 20×20 DataFrames (Sim A / Sim B) of total return %,
    plus dicts describing the best cell in each.
    """
    sorted_dates = sorted(trade_data)
    n = len(OPT_VALUES)
    mat_z = pd.DataFrame(index=OPT_VALUES, columns=OPT_VALUES, dtype=float)
    mat_c = pd.DataFrame(index=OPT_VALUES, columns=OPT_VALUES, dtype=float)

    for profit_pct in OPT_VALUES:
        for loss_pct in OPT_VALUES:
            port_z = initial_capital
            port_c = initial_capital
            for rec_date in sorted_dates:
                entry, data, res, next_open_pnl = trade_data[rec_date]
                outcome, exit_price, _ = analyze_trade(
                    entry, data, res, profit_pct, loss_pct
                )
                pnl = (exit_price - entry) / entry if exit_price is not None else 0.0

                pos_z = port_z * (position_pct / 100)
                if outcome in ("neutral", "ambiguous"):
                    ret_z = (next_open_pnl / 100) if next_open_pnl is not None else 0.0
                else:
                    ret_z = pnl
                port_z += pos_z * ret_z

                pos_c = port_c * (position_pct / 100)
                port_c += pos_c * pnl

            mat_z.loc[profit_pct, loss_pct] = (port_z - initial_capital) / initial_capital * 100
            mat_c.loc[profit_pct, loss_pct] = (port_c - initial_capital) / initial_capital * 100

    def best_cell(mat):
        idx = mat.stack().idxmax()
        return {"profit_pct": idx[0], "loss_pct": idx[1],
                "return_pct": mat.loc[idx[0], idx[1]]}

    return mat_z, mat_c, best_cell(mat_z), best_cell(mat_c)


def make_heatmap(matrix, best, title):
    z   = matrix.values.tolist()
    xs  = [f"{v:.1f}%" for v in OPT_VALUES]   # stop loss  (x axis)
    ys  = [f"{v:.1f}%" for v in OPT_VALUES]   # profit target (y axis)

    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ys,
        colorscale="RdYlGn",
        hovertemplate="TP: %{y}  SL: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %"),
    ))
    # Star on the best cell
    fig.add_trace(go.Scatter(
        x=[f"{best['loss_pct']:.1f}%"],
        y=[f"{best['profit_pct']:.1f}%"],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="white",
                    line=dict(color="black", width=1)),
        text=[f"  TP {best['profit_pct']}% / SL {best['loss_pct']}%"],
        textposition="middle right",
        textfont=dict(color="white", size=11),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Stop Loss %",
        yaxis_title="Profit Target %",
        height=520,
    )
    return fig


# ─── Portfolio Simulation ──────────────────────────────────────────────────────

def simulate_portfolio(
    results: pd.DataFrame,
    initial_capital: float,
    position_pct: float,
    neutral_mode: str = "next_open",   # "next_open" | "close"
) -> pd.DataFrame:
    """
    Walk through trades chronologically and compound profits/losses.
    neutral_mode="next_open" → neutral days exit at the open of the following trading day
    neutral_mode="close"     → neutral days exit at the day's closing price
    """
    trades = results[results["outcome"].isin(["success", "failure", "neutral", "ambiguous"])].copy()
    trades = trades.sort_values("rec_date").reset_index(drop=True)

    portfolio = initial_capital
    rows = [{"date": None, "label": "Start", "outcome": None,
              "position": 0, "trade_pnl_pct": 0, "trade_gain": 0,
              "portfolio": portfolio}]

    for _, trade in trades.iterrows():
        outcome = trade["outcome"]

        if outcome in ("neutral", "ambiguous") and neutral_mode == "next_open":
            raw = trade["next_open_pnl_pct"] if "next_open_pnl_pct" in trade.index else None
            ret = (raw or 0) / 100
        else:
            ret = (trade["pnl_pct"] or 0) / 100

        position   = portfolio * (position_pct / 100)
        gain       = position * ret
        portfolio += gain
        rows.append({
            "date":           trade["rec_date"],
            "label":          outcome,
            "outcome":        outcome,
            "position":       position,
            "trade_pnl_pct":  ret * 100,
            "trade_gain":     gain,
            "portfolio":      portfolio,
        })

    return pd.DataFrame(rows)


# ─── Display Helpers ───────────────────────────────────────────────────────────

def fmt_money(v, signed=False):
    """Format a dollar amount compactly: $1.23M, $45.6K, $789."""
    prefix = "+" if signed and v > 0 else ("-" if v < 0 else "")
    a = abs(v)
    if a >= 1_000_000:
        return f"{prefix}${a/1_000_000:.2f}M"
    if a >= 1_000:
        return f"{prefix}${a/1_000:.1f}K"
    return f"{prefix}${a:,.0f}"


def compute_drawdown(sim):
    """Returns (max_drawdown_pct, trough_date, recovery_days or None)."""
    df = sim[sim["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return 0.0, None, None
    port = df["portfolio"].values
    dates = df["date"].values
    running_max = port[0]
    running_max_date = dates[0]
    max_dd = 0.0
    max_dd_peak = port[0]
    max_dd_trough_date = dates[0]
    for i in range(len(port)):
        if port[i] > running_max:
            running_max = port[i]
            running_max_date = dates[i]
        dd = (running_max - port[i]) / running_max
        if dd > max_dd:
            max_dd = dd
            max_dd_peak = running_max
            max_dd_trough_date = dates[i]
    recovery_days = None
    if max_dd > 0:
        after = df[df["date"] > pd.Timestamp(max_dd_trough_date)]
        recovered = after[after["portfolio"] >= max_dd_peak]
        if not recovered.empty:
            recovery_days = (recovered["date"].iloc[0] - pd.Timestamp(max_dd_trough_date)).days
    return max_dd * 100, pd.Timestamp(max_dd_trough_date).date(), recovery_days


def compute_streaks(analyzed_df):
    """Returns (max_win_streak, max_loss_streak, current_streak_str)."""
    trades = analyzed_df[analyzed_df["outcome"].isin(["success", "failure"])].sort_values("rec_date")
    if trades.empty:
        return 0, 0, "—"
    max_w = max_l = cur = 0
    cur_type = None
    for outcome in trades["outcome"]:
        if outcome == "success":
            cur = cur + 1 if cur_type == "success" else 1
            cur_type = "success"
            max_w = max(max_w, cur)
        else:
            cur = cur + 1 if cur_type == "failure" else 1
            cur_type = "failure"
            max_l = max(max_l, cur)
    icon = "✅" if cur_type == "success" else "❌"
    return max_w, max_l, f"{icon} {cur} in a row"


OUTCOME_LABEL = {
    "success":   "✅ Success",
    "failure":   "❌ Failure",
    "neutral":   "⚡ Neutral",
    "ambiguous": "❓ Ambiguous",
    "future":    "🔮 Future",
    "no_data":   "🚫 No Data",
}
COLORS = {
    "✅ Success":   "#2ecc71",
    "❌ Failure":   "#e74c3c",
    "⚡ Neutral":   "#95a5a6",
    "❓ Ambiguous": "#f39c12",
}


# ─── Main ──────────────────────────────────────────────────────────────────────

if not uploaded_file:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    with st.expander("Expected CSV format (semicolon-delimited)"):
        st.code(
            'symbol;date;buy_sell;open;close;high;low;model;prediction\n'
            '"SOXL";"2026-04-22 00:00:00";"buy";102.94;105.64;106.05;99.6;"model_id";"buy"\n'
            '"SOXL";"2026-04-21 00:00:00";"buy";97.97;98.09;99.95;95.32;"model_id";"no buy"'
        )
    st.stop()

try:
    df = parse_csv(uploaded_file.read())
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

MIN_DATE = date(2020, 8, 1)  # Alpaca 1-min data reliable from this date

buy_recs_all = df[df["prediction"] == "buy"].copy()
buy_recs = buy_recs_all[buy_recs_all["date_only"] >= MIN_DATE].copy()
excluded = len(buy_recs_all) - len(buy_recs)
ticker = df["symbol"].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Ticker", ticker)
c2.metric("Total rows in CSV", len(df))
c3.metric("Buy signals", len(buy_recs),
          f"−{excluded} before {MIN_DATE}" if excluded else None)

if buy_recs.empty:
    st.warning("No rows with prediction='buy' found.")
    st.stop()

# ── Step 1: Fetch data (on button click, results cached) ───────────────────────
if analyze_btn:
    with st.spinner("Fetching trading calendar..."):
        min_d = buy_recs["date_only"].min()
        max_d = min(buy_recs["date_only"].max() + timedelta(days=10), date.today())
        trading_days = get_trading_days(ticker, min_d, max_d)
        st.session_state.update(
            analyzed=True, ticker=ticker,
            trading_days=trading_days, analyzed_extended=extended_hours,
        )

    has_local = os.path.exists(f"data/{ticker}_1min.parquet")
    label = "extended hours" if extended_hours else "intraday"
    progress = st.progress(0, text=f"Fetching {label} data...")
    for i, (_, rec) in enumerate(buy_recs.iterrows()):
        nd = next((d for d in trading_days if d > rec["date_only"]), None)
        if nd and nd <= date.today():
            if extended_hours and not has_local:
                # Only hit live API if no local Alpaca file
                fetch_extended_window(ticker, rec["date_only"], nd)
            else:
                # Always warm regular session cache (used as fallback)
                fetch_day_data(ticker, nd)
        progress.progress((i + 1) / len(buy_recs), text=f"Fetching {nd} ...")
    progress.empty()

# ── Step 2: Analyze (runs on every slider change once data is cached) ──────────
if not (st.session_state.get("analyzed") and st.session_state.get("ticker") == ticker):
    st.stop()

trading_days = st.session_state["trading_days"]
active_extended = st.session_state.get("analyzed_extended", False)

if extended_hours != active_extended:
    st.warning("Extended hours setting changed — click **Analyze** again to fetch the new data.")
    st.stop()

results = run_analysis(buy_recs, trading_days, profit_pct, loss_pct, extended_hours)

# Attach model info so later sections can group by it
if "model" in buy_recs.columns:
    model_map = buy_recs.set_index("date_only")["model"].to_dict()
    results["model"] = results["rec_date"].map(model_map)

analyzed = results[results["outcome"].isin(["success", "failure", "neutral", "ambiguous"])]
total = len(analyzed)

if total == 0:
    st.warning("No completed trades to analyze (all dates may be in the future).")
    st.stop()

counts = analyzed["outcome"].value_counts()
s = counts.get("success", 0)
f = counts.get("failure", 0)
n = counts.get("neutral", 0)
a = counts.get("ambiguous", 0)

# ── Summary metrics ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Results Summary")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Analyzed Trades", total)
m2.metric("✅ Successes", s, f"{s/total*100:.1f}%")
m3.metric("❌ Failures", f, f"{f/total*100:.1f}%")
m4.metric("⚡ Neutral", n, f"{n/total*100:.1f}%")
m5.metric("❓ Ambiguous", a, f"{a/total*100:.1f}%")

win_rate = s / (s + f) * 100 if (s + f) > 0 else 0.0

neutral_pnls = analyzed[analyzed["outcome"] == "neutral"]["pnl_pct"].dropna()
avg_neutral = neutral_pnls.mean() if len(neutral_pnls) > 0 else None

if (s + f) > 0:
    neutral_str = f"  |  ⚡ Avg neutral close: **{avg_neutral:+.2f}%**" if avg_neutral is not None else ""
    st.info(
        f"**Win rate (excluding neutral):** {win_rate:.1f}%  |  "
        f"Expected value per trade: **{(s * profit_pct - f * loss_pct) / total:.2f}%**"
        f"{neutral_str}"
    )

# ── Streak & Kelly ─────────────────────────────────────────────────────────────
max_win_streak, max_loss_streak, current_streak = compute_streaks(analyzed)

wins_pnl  = analyzed[analyzed["outcome"] == "success"]["pnl_pct"].dropna()
losses_pnl = analyzed[analyzed["outcome"] == "failure"]["pnl_pct"].dropna()
kelly_str = "—"
half_kelly = None
if len(wins_pnl) > 0 and len(losses_pnl) > 0:
    W = len(wins_pnl) / (len(wins_pnl) + len(losses_pnl))
    R = wins_pnl.mean() / abs(losses_pnl.mean())
    kelly_raw = (W - (1 - W) / R) * 100
    half_kelly = kelly_raw / 2
    kelly_str = f"{kelly_raw:.1f}%  (half-Kelly: {half_kelly:.1f}%)"

sk1, sk2, sk3, sk4 = st.columns(4)
sk1.metric("🏆 Max Win Streak",  max_win_streak)
sk2.metric("💀 Max Loss Streak", max_loss_streak)
sk3.metric("📍 Current Streak",  current_streak)
sk4.metric("🎲 Kelly Criterion",  kelly_str,
           delta=f"You use: {position_pct}%  ({'✅ safe' if half_kelly is None or position_pct <= half_kelly else '⚠️ above half-Kelly'})",
           delta_color="off")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.divider()
ch1, ch2 = st.columns(2)

with ch1:
    labels = [OUTCOME_LABEL.get(k, k) for k in counts.index]
    fig_pie = px.pie(
        values=counts.values, names=labels,
        color=labels, color_discrete_map=COLORS,
        title=f"Outcome Distribution  (TP +{profit_pct}% / SL -{loss_pct}%)",
    )
    fig_pie.update_traces(textinfo="percent+value")
    st.plotly_chart(fig_pie, use_container_width=True)

with ch2:
    tl = analyzed.copy()
    tl["label"] = tl["outcome"].map(OUTCOME_LABEL)
    tl["rec_date"] = pd.to_datetime(tl["rec_date"])
    fig_scatter = px.scatter(
        tl, x="rec_date", y="pnl_pct",
        color="label", color_discrete_map=COLORS,
        title="P&L % per Trade Over Time",
        labels={"rec_date": "Recommendation Date", "pnl_pct": "P&L %", "label": "Outcome"},
        size_max=10,
    )
    fig_scatter.add_hline(y=profit_pct, line_dash="dash", line_color="#2ecc71",
                          annotation_text=f"TP +{profit_pct}%", annotation_position="top right")
    fig_scatter.add_hline(y=-loss_pct, line_dash="dash", line_color="#e74c3c",
                          annotation_text=f"SL -{loss_pct}%", annotation_position="bottom right")
    fig_scatter.add_hline(y=0, line_color="gray", line_width=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Monthly Win Rate Heatmap ───────────────────────────────────────────────────
st.divider()
st.subheader("📅 Monthly Performance")
_sf = analyzed[analyzed["outcome"].isin(["success", "failure"])].copy()
_sf["rec_date"] = pd.to_datetime(_sf["rec_date"])
_sf["year"]  = _sf["rec_date"].dt.year
_sf["month"] = _sf["rec_date"].dt.month
if not _sf.empty:
    _monthly = _sf.groupby(["year", "month"]).agg(
        wins=("outcome", lambda x: (x == "success").sum()),
        total=("outcome", "count"),
    ).reset_index()
    _monthly["win_rate"] = (_monthly["wins"] / _monthly["total"] * 100).round(1)
    _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _years  = sorted(_monthly["year"].unique())
    _z, _txt = [], []
    for yr in _years:
        row_z, row_t = [], []
        for mo in range(1, 13):
            cell = _monthly[(_monthly["year"] == yr) & (_monthly["month"] == mo)]
            if cell.empty:
                row_z.append(None); row_t.append("")
            else:
                wr = cell["win_rate"].iloc[0]
                cnt = cell["total"].iloc[0]
                row_z.append(wr); row_t.append(f"{wr:.0f}%<br>({cnt})")
        _z.append(row_z); _txt.append(row_t)
    fig_monthly = go.Figure(go.Heatmap(
        z=_z, x=_MONTHS, y=[str(y) for y in _years],
        colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#2ecc71"]],
        zmid=50, zmin=0, zmax=100,
        text=_txt, texttemplate="%{text}",
        hovertemplate="Month: %{x}  Year: %{y}<br>Win Rate: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Win Rate %"),
    ))
    fig_monthly.update_layout(
        title="Monthly Win Rate % (number of trades in parentheses)",
        xaxis_title="Month", yaxis_title="Year",
        height=max(300, len(_years) * 55 + 120),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

# ── Detailed table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Detailed Results")

disp = results.copy()
disp["Outcome"] = disp["outcome"].map(OUTCOME_LABEL)
disp["Entry Price"] = disp["entry"].apply(lambda x: f"${x:.2f}")
disp["Exit Price"] = disp["exit_price"].apply(lambda x: f"${x:.2f}" if x is not None else "—")
disp["P&L %"] = disp["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if x is not None else "—")
disp["Data Res."] = disp["resolution"].fillna("—")

st.dataframe(
    disp[["rec_date", "Entry Price", "next_day", "Outcome", "P&L %", "note", "Data Res."]].rename(columns={
        "rec_date": "Recommendation Date",
        "next_day": "Trade Day",
        "note": "Note",
    }),
    use_container_width=True,
    hide_index=True,
)

if active_extended:
    st.caption(
        "Extended hours window: 4 PM on recommendation day → 4 PM on next trading day  |  "
        "5m+ext = 5-min bars with pre/post-market (≤60 days)  |  "
        "Falls back to regular session when extended data is unavailable."
    )
else:
    st.caption(
        "Regular session only: 9:30 AM – 4:00 PM ET on next trading day  |  "
        "1m (≤7d) | 5m (≤60d) | 1h (≤2yr) | 1d (>2yr, order of High/Low unknown)."
    )

# ── Model Comparison ───────────────────────────────────────────────────────────
if "model" in results.columns and results["model"].nunique() > 1:
    st.divider()
    st.subheader("🤖 Model Comparison")
    _model_rows = []
    for _mid, _grp in results.groupby("model"):
        _ga = _grp[_grp["outcome"].isin(["success", "failure", "neutral", "ambiguous"])]
        _gs = (_ga["outcome"] == "success").sum()
        _gf = (_ga["outcome"] == "failure").sum()
        _gn = (_ga["outcome"] == "neutral").sum()
        _wr = _gs / (_gs + _gf) * 100 if (_gs + _gf) > 0 else 0
        _avg = _ga["pnl_pct"].mean()
        _model_rows.append({
            "Model":       _mid,
            "Trades":      len(_ga),
            "✅ Wins":     _gs,
            "❌ Losses":   _gf,
            "⚡ Neutral":  _gn,
            "Win Rate":    f"{_wr:.1f}%",
            "Avg P&L":     f"{_avg:+.2f}%" if pd.notna(_avg) else "—",
        })
    st.dataframe(pd.DataFrame(_model_rows), hide_index=True, use_container_width=True)

    _fig_models = px.bar(
        pd.DataFrame(_model_rows), x="Model", y="Win Rate",
        title="Win Rate by Model",
        text="Win Rate",
        color="Win Rate",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
    )
    _fig_models.update_traces(textposition="outside")
    _fig_models.update_layout(showlegend=False, yaxis_title="Win Rate (%)", height=350)
    st.plotly_chart(_fig_models, use_container_width=True)

# ── Portfolio Simulation ───────────────────────────────────────────────────────
st.divider()
st.subheader("💰 Portfolio Simulation")
st.caption(
    f"Starting with **${initial_capital:,.0f}**, investing **{position_pct}%** of the portfolio on each trade "
    f"(TP +{profit_pct}% / SL -{loss_pct}%). Trades applied in chronological order with compounding."
)

sim = simulate_portfolio(results, initial_capital, position_pct, neutral_mode="close")

def _prep_chart(sim_df, start_date):
    df = sim_df.copy()
    df["date"] = df["date"].apply(
        lambda d: pd.Timestamp(d) if d is not None else pd.Timestamp(start_date)
    )
    return df

def _sim_metrics(sim_df, cap):
    rows  = sim_df.iloc[1:]
    final = sim_df["portfolio"].iloc[-1]
    ret   = (final - cap) / cap * 100
    return final, ret, rows["trade_gain"].max(), rows["trade_gain"].min()

def _milestone_data(sim_df, cap, t0):
    df = sim_df[sim_df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if df.empty:
        return [], None
    t0 = pd.Timestamp(t0)
    milestones, prev = [], t0
    for mult in range(2, 200):
        hit = df[df["portfolio"] >= cap * mult]
        if hit.empty:
            break
        d = hit["date"].iloc[0]
        milestones.append({"mult": mult, "date": d.date(),
                           "days_total": (d - t0).days, "days_delta": (d - prev).days})
        prev = d
    exp, dbl = 1, [t0]
    while True:
        hit = df[df["portfolio"] >= cap * (2 ** exp)]
        if hit.empty:
            break
        dbl.append(hit["date"].iloc[0])
        exp += 1
    avg = (sum((dbl[i+1]-dbl[i]).days for i in range(len(dbl)-1)) / (len(dbl)-1)
           if len(dbl) > 1 else None)
    return milestones, avg

ret_sim = None
if len(sim) <= 1:
    st.info("No completed trades to simulate yet.")
else:
    start_date = results["rec_date"].min()
    fv, ret_sim, best_t, worst_t = _sim_metrics(sim, initial_capital)
    dd_sim, dd_sim_date, rec_sim  = compute_drawdown(sim)

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Final Value",    fmt_money(fv),                 f"{ret_sim:+.1f}%")
    sm2.metric("Best trade",     fmt_money(best_t, signed=True))
    sm3.metric("Worst trade",    fmt_money(worst_t, signed=True))
    sm4.metric("📉 Max Drawdown", f"{dd_sim:.1f}%", f"low: {dd_sim_date}", delta_color="off")
    sm5.metric("⏳ Recovery",
               f"{rec_sim}d" if rec_sim is not None else "Not yet", delta_color="off")

    # Portfolio growth chart
    chart_c = _prep_chart(sim, start_date)[["date", "portfolio"]]
    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(
        x=chart_c["date"], y=chart_c["portfolio"],
        name="Strategy", line=dict(color="#e67e22", width=2.5),
        hovertemplate="%{x}<br>Portfolio: $%{y:,.0f}<extra></extra>",
    ))
    _bnh_end = results["next_day"].dropna().max()
    if isinstance(_bnh_end, date):
        _bnh = fetch_buy_hold(ticker, start_date, _bnh_end, initial_capital)
        if _bnh is not None:
            fig_port.add_trace(go.Scatter(
                x=_bnh["date"], y=_bnh["portfolio"],
                name=f"Buy & Hold {ticker}", line=dict(color="#9b59b6", width=2, dash="dot"),
                hovertemplate="%{x}<br>Buy & Hold: $%{y:,.0f}<extra></extra>",
            ))
    fig_port.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                       annotation_text="Starting Capital", annotation_position="top left")
    fig_port.update_layout(
        title="Portfolio Growth — neutral days exit at close (compounded)",
        xaxis_title="Date", yaxis_title="Portfolio ($)",
        yaxis=dict(tickformat="$,.3s"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_port, use_container_width=True)

    # ── Doubling Analysis ─────────────────────────────────────────────────────
    st.markdown("#### ⏱ Time to Double")
    ms_c, avg_c = _milestone_data(sim, initial_capital, start_date)
    if avg_c is not None:
        st.metric("Avg time to double", f"{avg_c:.0f} days",
                  f"≈ {avg_c/30.44:.1f} months", delta_color="off")
    else:
        st.info("Portfolio hasn't doubled yet.")
    if ms_c:
        tbl_ms = pd.DataFrame([{
            "Milestone": f"×{m['mult']}", "Target": fmt_money(initial_capital * m["mult"]),
            "Date": str(m["date"]), "Days from start": m["days_total"],
            "Days since ×prev": m["days_delta"],
        } for m in ms_c])
        col_tbl, col_bar = st.columns([1, 2])
        with col_tbl:
            st.dataframe(tbl_ms, hide_index=True, use_container_width=True)
        with col_bar:
            fig_dbl = go.Figure(go.Bar(
                x=[f"×{m['mult']}" for m in ms_c], y=[m["days_total"] for m in ms_c],
                marker_color="#e67e22",
                hovertemplate="Milestone: %{x}<br>Days: %{y}<extra></extra>",
            ))
            fig_dbl.update_layout(title="Days to reach each milestone",
                                  xaxis_title="Multiple", yaxis_title="Days", height=300)
            st.plotly_chart(fig_dbl, use_container_width=True)

    # ── Per-trade breakdown ───────────────────────────────────────────────────
    with st.expander("Trade-by-trade breakdown"):
        def _trade_table(sim_df):
            d = sim_df.iloc[1:].copy()
            d["Outcome"]     = d["label"].map(OUTCOME_LABEL)
            d["Position"]    = d["position"].apply(fmt_money)
            d["Trade P&L"]   = d["trade_gain"].apply(lambda x: fmt_money(x, signed=True))
            d["Trade %"]     = d["trade_pnl_pct"].apply(lambda x: f"{x:+.2f}%")
            d["Portfolio →"] = d["portfolio"].apply(fmt_money)
            st.dataframe(
                d[["date", "Outcome", "Position", "Trade %", "Trade P&L", "Portfolio →"]].rename(
                    columns={"date": "Recommendation Date"}),
                use_container_width=True, hide_index=True,
            )
        _trade_table(sim)

# ── Advanced Risk Analytics ────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Advanced Risk Analytics")

# ── Computed values used across all tabs ──────────────────────────────────────
_wins_r  = analyzed[analyzed["outcome"] == "success"]["pnl_pct"].dropna()
_losses_r = analyzed[analyzed["outcome"] == "failure"]["pnl_pct"].dropna()
_all_rets = analyzed["pnl_pct"].dropna().values / 100

_first_dt   = pd.to_datetime(results["rec_date"].min())
_last_nd    = results["next_day"].dropna()
_last_dt    = pd.to_datetime(_last_nd.max()) if not _last_nd.empty else _first_dt
_years_span = max((_last_dt - _first_dt).days / 365.25, 0.01)
_n_per_yr   = total / _years_span

# Sharpe (annualized, risk-free = 0)
_mean_r  = np.mean(_all_rets) if len(_all_rets) > 0 else 0.0
_std_r   = np.std(_all_rets, ddof=1) if len(_all_rets) > 1 else 0.0
_sharpe  = (_mean_r / _std_r) * np.sqrt(_n_per_yr) if _std_r > 0 else 0.0

# Sortino (downside deviation only)
_neg_r    = _all_rets[_all_rets < 0]
_down_std = np.std(_neg_r, ddof=1) if len(_neg_r) > 1 else 0.0
_sortino  = (_mean_r / _down_std) * np.sqrt(_n_per_yr) if _down_std > 0 else 0.0

# Profit Factor
_pf = _wins_r.sum() / abs(_losses_r.sum()) if len(_losses_r) > 0 and _losses_r.sum() != 0 else float("inf")

# Drawdown + Calmar
_dd_adv, _dd_adv_date, _rec_adv = compute_drawdown(sim)
_final_adv  = sim["portfolio"].iloc[-1] if len(sim) > 1 else initial_capital
_ann_ret_adv = (((_final_adv / initial_capital) ** (1 / _years_span)) - 1) * 100
_calmar = _ann_ret_adv / _dd_adv if _dd_adv > 0 else float("inf")

# Expectancy
_w_rt = len(_wins_r) / (len(_wins_r) + len(_losses_r)) if (len(_wins_r) + len(_losses_r)) > 0 else 0
_exp_pct = _w_rt * (_wins_r.mean() if len(_wins_r) > 0 else 0) - (1 - _w_rt) * (abs(_losses_r.mean()) if len(_losses_r) > 0 else 0)
_exp_usd = _exp_pct / 100 * initial_capital * (position_pct / 100)

# ── Tabs ──────────────────────────────────────────────────────────────────────
_tab_risk, _tab_regime, _tab_time = st.tabs(["🛡️ Risk Metrics", "🌍 Market Regimes", "⏰ Exit Timing"])

with _tab_risk:
    rr1, rr2, rr3, rr4 = st.columns(4)
    rr1.metric("📉 Max Drawdown", f"{_dd_adv:.1f}%",
               f"low: {_dd_adv_date}  •  {'recovered in ' + str(_rec_adv) + 'd' if _rec_adv else 'not recovered'}",
               delta_color="off")
    rr2.metric("⚡ Sharpe Ratio", f"{_sharpe:.2f}",
               "🟢 Excellent" if _sharpe > 2 else ("🟡 Good" if _sharpe > 1 else "🔴 Below avg"),
               delta_color="off")
    rr3.metric("🎯 Sortino Ratio", f"{_sortino:.2f}",
               "🟢 Excellent" if _sortino > 2 else ("🟡 Good" if _sortino > 1 else "🔴 Below avg"),
               delta_color="off")
    rr4.metric("💰 Profit Factor", f"{_pf:.2f}" if _pf < 1000 else "∞",
               "🟢 Strong (≥1.5)" if _pf >= 1.5 else ("🟡 OK (≥1.0)" if _pf >= 1 else "🔴 Losing"),
               delta_color="off")

    rr5, rr6, rr7, rr8 = st.columns(4)
    rr5.metric("📈 Calmar Ratio", f"{_calmar:.2f}" if _calmar < 1000 else "∞",
               "Annualized return ÷ Max drawdown", delta_color="off")
    rr6.metric("🎲 Expectancy", f"{_exp_pct:+.2f}%",
               f"≈ {fmt_money(_exp_usd, signed=True)} per trade", delta_color="off")
    rr7.metric("💀 Max Loss Streak", max_loss_streak,
               "consecutive losing trades", delta_color="off")
    rr8.metric("🏆 Max Win Streak", max_win_streak,
               "consecutive winning trades", delta_color="off")

    # Return distribution histogram
    st.markdown("##### Return Distribution per Trade")
    _pos_r = analyzed[analyzed["pnl_pct"] >= 0]["pnl_pct"].dropna()
    _neg_r2 = analyzed[analyzed["pnl_pct"] < 0]["pnl_pct"].dropna()
    fig_dist = go.Figure()
    if not _neg_r2.empty:
        fig_dist.add_trace(go.Histogram(x=_neg_r2, nbinsx=25, name="Loss",
                                        marker_color="#e74c3c", opacity=0.85))
    if not _pos_r.empty:
        fig_dist.add_trace(go.Histogram(x=_pos_r, nbinsx=25, name="Win",
                                        marker_color="#2ecc71", opacity=0.85))
    fig_dist.add_vline(x=0, line_color="white", line_width=1, line_dash="dash")
    fig_dist.add_vline(x=float(analyzed["pnl_pct"].mean()), line_color="#f39c12",
                       line_dash="dot",
                       annotation_text=f" Avg {analyzed['pnl_pct'].mean():+.2f}%",
                       annotation_position="top right")
    fig_dist.update_layout(barmode="overlay", height=350,
                           xaxis_title="P&L %", yaxis_title="Count",
                           legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_dist, use_container_width=True)

with _tab_regime:
    _reg_end_nd = results["next_day"].dropna().max()
    _reg_end_d  = _reg_end_nd if isinstance(_reg_end_nd, date) else pd.Timestamp(_reg_end_nd).date()
    _reg_df = fetch_market_regimes(results["rec_date"].min(), _reg_end_d)

    _ana_reg = analyzed.copy()
    _ana_reg["_nd"] = pd.to_datetime(_ana_reg["next_day"]).dt.date
    _ana_reg = _ana_reg.merge(_reg_df.reset_index().rename(columns={"index": "_nd"}),
                              on="_nd", how="left")

    _ana_reg["spy_dir"] = _ana_reg["spy_ret"].apply(
        lambda x: "📈 SPY Up" if pd.notna(x) and x > 0 else ("📉 SPY Down" if pd.notna(x) and x < 0 else "❓ Unknown")
    )
    _ana_reg["vix_reg"] = _ana_reg["vix"].apply(
        lambda v: "🟢 Low VIX (<20)" if pd.notna(v) and v < 20
        else ("🟡 Med VIX (20–30)" if pd.notna(v) and v < 30
              else ("🔴 High VIX (>30)" if pd.notna(v) else "❓ Unknown"))
    )
    _sf_r = _ana_reg[_ana_reg["outcome"].isin(["success", "failure"])]

    def _regime_table(grp_col):
        g = _sf_r.groupby(grp_col).agg(
            Trades=("outcome", "count"),
            Wins=("outcome", lambda x: (x == "success").sum()),
            Avg_PnL=("pnl_pct", "mean"),
        ).reset_index()
        g["Win %"]   = (g["Wins"] / g["Trades"] * 100).round(1)
        g["Avg P&L"] = g["Avg_PnL"].round(2)
        return g[[grp_col, "Trades", "Wins", "Win %", "Avg P&L"]].rename(columns={grp_col: "Regime"})

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("##### Win rate by SPY direction on trade day")
        st.dataframe(_regime_table("spy_dir"), hide_index=True, use_container_width=True)
    with rc2:
        st.markdown("##### Win rate by VIX level on trade day")
        st.dataframe(_regime_table("vix_reg"), hide_index=True, use_container_width=True)

    # Alpha & correlation
    _alpha_df = _ana_reg[["pnl_pct", "spy_ret"]].dropna()
    if not _alpha_df.empty:
        _corr  = _alpha_df["pnl_pct"].corr(_alpha_df["spy_ret"])
        _alpha = _alpha_df["pnl_pct"].mean() - _alpha_df["spy_ret"].mean()
        ac1, ac2 = st.columns(2)
        ac1.metric("Correlation to SPY", f"{_corr:.3f}",
                   "Low = more independent from market" if abs(_corr) < 0.3 else
                   ("Medium correlation" if abs(_corr) < 0.6 else "High correlation"),
                   delta_color="off")
        ac2.metric("Alpha vs SPY (avg per trade)", f"{_alpha:+.2f}%",
                   "Strategy avg P&L minus SPY avg return on same day",
                   delta_color="off")
        st.markdown("##### Strategy P&L vs SPY Return (each dot = one trade)")
        fig_alpha = px.scatter(
            _alpha_df, x="spy_ret", y="pnl_pct",
            labels={"spy_ret": "SPY Return %", "pnl_pct": "Strategy P&L %"},
            color_discrete_sequence=["#3498db"],
            opacity=0.6,
        )
        fig_alpha.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_alpha.add_vline(x=0, line_color="gray", line_width=0.5)
        fig_alpha.update_layout(height=380)
        st.plotly_chart(fig_alpha, use_container_width=True)

with _tab_time:
    st.markdown("##### How Each Trade Was Resolved")

    def _exit_type(row):
        note = str(row.get("note", "")).lower()
        res  = str(row.get("resolution", "")).lower()
        if "gap" in note or "post-close" in note:
            return "🌙 Overnight gap / Pre-market"
        if row["outcome"] in ("neutral", "ambiguous"):
            return "⏰ Held to end of day"
        if "1m" in res or "5m" in res:
            return "⚡ Intraday (high-res)"
        if "1h" in res:
            return "🕐 Intraday (hourly)"
        if "1d" in res:
            return "📊 Daily bar (order unknown)"
        return "❓ Other"

    _t_ana = analyzed.copy()
    _t_ana["Exit Type"] = _t_ana.apply(_exit_type, axis=1)
    _ec = _t_ana["Exit Type"].value_counts().reset_index()
    _ec.columns = ["Exit Type", "Count"]
    _ec["% of Trades"] = (_ec["Count"] / len(_t_ana) * 100).round(1)

    et1, et2 = st.columns([1, 2])
    with et1:
        st.dataframe(_ec, hide_index=True, use_container_width=True)
        st.caption(
            "**Overnight gap / Pre-market:** TP or SL triggered before regular session opened — "
            "highest slippage risk.\n\n"
            "**Held to end of day:** neither TP nor SL hit (neutral/ambiguous).\n\n"
            "**Avg holding time:** ~1 trading day by design (entry at close, check next day)."
        )
    with et2:
        fig_exit = px.pie(_ec, values="Count", names="Exit Type",
                          title="Exit Timing Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_exit.update_traces(textinfo="percent+label")
        fig_exit.update_layout(height=350)
        st.plotly_chart(fig_exit, use_container_width=True)

    st.info(
        "💡 **Slippage note:** trades that exit via overnight gap have the highest slippage risk "
        "— the actual fill price may differ significantly from the theoretical TP/SL price shown. "
        "If a large % of your wins come from gaps, factor in 0.1–0.3% slippage per gap trade."
    )

# ── Save Strategy ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("💾 Save this strategy", expanded=False):
    model_id = df["model"].iloc[0] if "model" in df.columns else "unknown"
    with st.form("save_strategy_form", clear_on_submit=True):
        st.caption(
            f"Current settings: TP **{profit_pct}%** / SL **{loss_pct}%** / "
            f"Position **{position_pct}%** / Capital **${initial_capital:,.0f}** / "
            f"Win rate **{win_rate:.1f}%** / Model `{model_id}`"
        )
        s_name = st.text_input("Strategy name *", placeholder="e.g. SOXL Bull — Conservative v1")
        s_desc = st.text_area("Description (optional)",
                              placeholder="When does this work? Any conditions or notes…",
                              height=80)
        submitted = st.form_submit_button("💾 Save", type="primary")
        if submitted:
            if not s_name.strip():
                st.error("Please enter a strategy name.")
            else:
                add_strategy({
                    "id":             str(uuid.uuid4()),
                    "name":           s_name.strip(),
                    "description":    s_desc.strip(),
                    "model":          model_id,
                    "ticker":         ticker,
                    "profit_pct":     profit_pct,
                    "loss_pct":       loss_pct,
                    "position_pct":   position_pct,
                    "initial_capital": initial_capital,
                    "extended_hours": active_extended,
                    "win_rate":       round(win_rate, 1),
                    "n_trades":       total,
                    "total_return_a": round(ret_sim, 2) if ret_sim is not None else None,
                    "total_return_b": round(ret_sim, 2) if ret_sim is not None else None,
                    "saved_at":       datetime.now().isoformat(timespec="seconds"),
                })
                st.success(f"✅ '{s_name}' saved! Check the sidebar.")
                st.rerun()

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📥 Export Report")
if st.button("📊 Generate Excel Report", type="secondary"):
    import io as _io
    try:
        import openpyxl  # noqa: F401
        _buf = _io.BytesIO()
        with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
            # Summary sheet
            _summary = pd.DataFrame([{
                "Ticker": ticker,
                "Total Trades": total,
                "Successes": s, "Failures": f, "Neutral": n, "Ambiguous": a,
                "Win Rate (%)": round(win_rate, 1),
                "TP (%)": profit_pct, "SL (%)": loss_pct,
                "Position Size (%)": position_pct,
                "Initial Capital ($)": initial_capital,
                "Max Win Streak": max_win_streak,
                "Max Loss Streak": max_loss_streak,
            }])
            _summary.to_excel(_xw, sheet_name="Summary", index=False)
            # Full results
            results.to_excel(_xw, sheet_name="All Trades", index=False)
            # Simulations
            sim_zero.iloc[1:].to_excel(_xw, sheet_name="Sim A (exit next open)", index=False)
            sim_close.iloc[1:].to_excel(_xw, sheet_name="Sim B (exit at close)", index=False)
        st.download_button(
            label="⬇️ Download Excel",
            data=_buf.getvalue(),
            file_name=f"{ticker}_analysis_{date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.error("openpyxl is not installed. Run: `pip install openpyxl`")

# ── Parameter Optimization ─────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Parameter Optimization")
st.caption(
    "Grid search: tests every combination of Profit Target × Stop Loss "
    "(0.5%–10%, step 0.5%) → **400 combinations**. "
    "Shows total return % for each pair. Star = best combination."
)

opt_btn = st.button("▶ Run Optimization", type="secondary")

# Invalidate cached results if a new Analyze was run
opt_key = f"{ticker}_{active_extended}_{len(buy_recs)}_{initial_capital}_{position_pct}"
if st.session_state.get("opt_key") != opt_key:
    st.session_state.pop("opt_results", None)

if opt_btn:
    local_df_opt = load_local_data(ticker) if active_extended else None
    with st.spinner("Running 400 combinations… (usually < 5 seconds)"):
        trade_data = preload_trade_data(
            buy_recs, trading_days, active_extended, local_df_opt, ticker
        )
        mat_z, mat_c, best_z, best_c = run_optimization(
            trade_data, initial_capital, position_pct
        )
    st.session_state["opt_results"] = (mat_z, mat_c, best_z, best_c)
    st.session_state["opt_key"] = opt_key

if "opt_results" in st.session_state:
    mat_z, mat_c, best_z, best_c = st.session_state["opt_results"]
    st.success(
        f"**🎯 Best parameters (exit at close)**\n\n"
        f"Profit Target: **{best_c['profit_pct']}%** &nbsp;|&nbsp; "
        f"Stop Loss: **{best_c['loss_pct']}%**\n\n"
        f"Total return: **{best_c['return_pct']:+.2f}%**"
    )
    if True:
        st.plotly_chart(
            make_heatmap(mat_c, best_c, "Total Return % (neutral days exit at close)"),
            use_container_width=True,
        )

# ── Robustness Analysis ────────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Robustness Analysis")
st.caption(
    "A good strategy should perform consistently across a **range** of nearby parameters, "
    "not just at a single optimized peak. Tests 25 TP/SL combinations in a ±1% neighborhood "
    "around your current settings."
)

rob_btn = st.button("▶ Run Robustness Check", type="secondary", key="rob_btn")
rob_key = f"rob_{ticker}_{active_extended}_{len(buy_recs)}_{profit_pct}_{loss_pct}_{initial_capital}_{position_pct}"
if st.session_state.get("rob_key") != rob_key:
    st.session_state.pop("rob_results", None)

if rob_btn:
    _step = 0.5
    _tp_vals = sorted({max(0.5, round(profit_pct + i * _step, 1)) for i in range(-2, 3)})
    _sl_vals = sorted({max(0.5, round(loss_pct  + i * _step, 1)) for i in range(-2, 3)})
    _local_r = load_local_data(ticker) if active_extended else None
    with st.spinner("Running 25 nearby parameter combinations..."):
        _rob_data = preload_trade_data(buy_recs, trading_days, active_extended, _local_r, ticker)
        _rob_mat  = pd.DataFrame(index=_tp_vals, columns=_sl_vals, dtype=float)
        for _tp in _tp_vals:
            for _sl in _sl_vals:
                _port = initial_capital
                for _rd in sorted(_rob_data):
                    _e, _d, _r, _nop = _rob_data[_rd]
                    _oc, _xp, _ = analyze_trade(_e, _d, _r, _tp, _sl)
                    _pnl = (_xp - _e) / _e if _xp is not None else 0.0
                    _ret = (_nop / 100 if _nop is not None else 0.0) if _oc in ("neutral", "ambiguous") else _pnl
                    _port += _port * (position_pct / 100) * _ret
                _rob_mat.loc[_tp, _sl] = (_port - initial_capital) / initial_capital * 100
    st.session_state["rob_results"] = _rob_mat
    st.session_state["rob_key"] = rob_key

if "rob_results" in st.session_state:
    _rm = st.session_state["rob_results"]
    _rv = _rm.values.flatten().astype(float)
    _center_val = float(_rm.loc[profit_pct, loss_pct]) if (profit_pct in _rm.index and loss_pct in _rm.columns) else float(_rv[len(_rv)//2])
    _pct_pos = (_rv > 0).sum() / len(_rv) * 100
    _cv = abs(_rv.std() / _rv.mean()) * 100 if _rv.mean() != 0 else 999.0

    if _pct_pos >= 80 and _cv < 50:
        _rob_verdict = "🟢 **Robust** — strategy delivers positive returns across a wide parameter range"
    elif _pct_pos >= 50:
        _rob_verdict = "🟡 **Moderate** — works for most nearby values, some fragility present"
    else:
        _rob_verdict = "🔴 **Fragile** — performance collapses with small parameter changes (likely overfit)"

    st.info(_rob_verdict)
    rmc1, rmc2, rmc3 = st.columns(3)
    rmc1.metric("Current TP/SL return",     f"{_center_val:+.1f}%")
    rmc2.metric("Neighborhood positive %",  f"{_pct_pos:.0f}%",
                help="% of the 25 nearby param combos that yield positive total return")
    rmc3.metric("Variation (CV)",           f"{_cv:.0f}%",
                help="std / mean × 100. Lower = flatter plateau = more robust strategy")

    _rx = [f"SL {v}%" for v in _rm.columns.tolist()]
    _ry = [f"TP {v}%" for v in _rm.index.tolist()]
    fig_rob = go.Figure(go.Heatmap(
        z=_rm.values.tolist(), x=_rx, y=_ry,
        colorscale="RdYlGn",
        hovertemplate="TP: %{y}  SL: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %"),
    ))
    fig_rob.add_trace(go.Scatter(
        x=[f"SL {loss_pct}%"], y=[f"TP {profit_pct}%"],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="white", line=dict(color="black", width=1)),
        text=["  Current"], textposition="middle right",
        textfont=dict(color="white", size=11),
        showlegend=False, hoverinfo="skip",
    ))
    fig_rob.update_layout(
        title=f"Neighborhood Heatmap — TP={profit_pct}% / SL={loss_pct}% ± 1%",
        xaxis_title="Stop Loss %", yaxis_title="Profit Target %", height=420,
    )
    st.plotly_chart(fig_rob, use_container_width=True)

# ── Out-of-Sample Test ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🧪 Out-of-Sample Test")
st.caption(
    "Split your data: optimize on **In-Sample** only, then validate on the **Out-of-Sample** period "
    "you never touched. A large performance drop signals overfitting."
)

_d_all   = sorted(buy_recs["date_only"].unique())
_d_min   = _d_all[0]
_d_max   = _d_all[-1]
_default_cut = _d_min + timedelta(days=int((_d_max - _d_min).days * 0.8))
_default_cut = min(_default_cut, _d_max - timedelta(days=30))

_cut_col, _cnt_col = st.columns([2, 1])
with _cut_col:
    oos_cut = st.date_input(
        "In-Sample ends on (Out-of-Sample starts the next day)",
        value=_default_cut,
        min_value=_d_min + timedelta(days=30),
        max_value=_d_max - timedelta(days=30),
        key="oos_cut",
    )
_is_recs  = buy_recs[buy_recs["date_only"] <= oos_cut]
_oos_recs = buy_recs[buy_recs["date_only"] >  oos_cut]
with _cnt_col:
    st.metric("In-Sample trades",     len(_is_recs),  f"up to {oos_cut}")
    st.metric("Out-of-Sample trades", len(_oos_recs), f"from {oos_cut + timedelta(days=1)}")

oos_btn = st.button("▶ Run Out-of-Sample Test", type="secondary", key="oos_btn")
oos_key  = f"oos_{ticker}_{active_extended}_{len(_is_recs)}_{len(_oos_recs)}_{initial_capital}_{position_pct}"
if st.session_state.get("oos_key") != oos_key:
    st.session_state.pop("oos_results", None)

if oos_btn:
    if len(_is_recs) < 5 or len(_oos_recs) < 3:
        st.error("Not enough trades in one of the periods — adjust the split date.")
    else:
        _local_oos = load_local_data(ticker) if active_extended else None
        with st.spinner("Step 1/2: Optimizing on In-Sample data (400 combinations)..."):
            _is_data = preload_trade_data(_is_recs, trading_days, active_extended, _local_oos, ticker)
            _is_mz, _is_mc, _is_bz, _is_bc = run_optimization(_is_data, initial_capital, position_pct)
        with st.spinner("Step 2/2: Validating on Out-of-Sample data..."):
            _oos_res_c = run_analysis(_oos_recs, trading_days, _is_bc["profit_pct"], _is_bc["loss_pct"], active_extended)
            _oos_sim_c = simulate_portfolio(_oos_res_c, initial_capital, position_pct, "close")
        st.session_state["oos_results"] = {
            "is_bc": _is_bc,
            "oos_res_c": _oos_res_c, "oos_sim_c": _oos_sim_c,
        }
        st.session_state["oos_key"] = oos_key

if "oos_results" in st.session_state:
    _oos  = st.session_state["oos_results"]
    _best = _oos["is_bc"]
    _oos_sim  = _oos["oos_sim_c"]
    _oos_res  = _oos["oos_res_c"]
    st.markdown(
        f"In-Sample best params: TP **{_best['profit_pct']}%** / SL **{_best['loss_pct']}%**  "
        f"→ In-Sample return: **{_best['return_pct']:+.2f}%**"
    )
    _oa = _oos_res[_oos_res["outcome"].isin(["success","failure","neutral","ambiguous"])]
    if _oa.empty:
        st.warning("No Out-of-Sample trades to evaluate.")
    else:
        _of  = _oos_sim["portfolio"].iloc[-1] if len(_oos_sim) > 1 else initial_capital
        _or  = (_of - initial_capital) / initial_capital * 100
        _os  = (_oa["outcome"] == "success").sum()
        _ofl = (_oa["outcome"] == "failure").sum()
        _owr = _os / (_os + _ofl) * 100 if (_os + _ofl) > 0 else 0
        _is_r = _best["return_pct"]
        _deg  = (_is_r - _or) / abs(_is_r) * 100 if _is_r != 0 else (0 if _or >= 0 else 100)
        _fit  = ("🟢 No significant overfitting" if _deg < 30 else
                 "🟡 Moderate degradation — possible overfitting" if _deg < 70 else
                 "🔴 Severe degradation — likely overfit")
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("IS Return",    f"{_is_r:+.1f}%", "In-Sample (optimized)")
        oc2.metric("OOS Return",   f"{_or:+.1f}%",   "Out-of-Sample (unseen)")
        oc3.metric("OOS Win Rate", f"{_owr:.1f}%",   f"{int(_os)}W / {int(_ofl)}L")
        oc4.metric("Degradation",  f"{_deg:.0f}%",   _fit, delta_color="off")

# ── Monte Carlo Simulation ─────────────────────────────────────────────────────
st.divider()
st.subheader("🎲 Monte Carlo Simulation")
st.caption(
    "Bootstrap resampling: randomly reshuffles the historical trade returns N times "
    "to estimate the distribution of possible outcomes. "
    "Shows the realistic best, median, and worst-case scenarios."
)

mc_n   = st.slider("Number of simulations", 200, 2000, 1000, 100, key="mc_n")
mc_btn = st.button("▶ Run Monte Carlo", type="secondary", key="mc_btn")
mc_key = f"mc_{ticker}_{len(buy_recs)}_{profit_pct}_{loss_pct}_{initial_capital}_{position_pct}_{mc_n}"
if st.session_state.get("mc_key") != mc_key:
    st.session_state.pop("mc_results", None)

if mc_btn:
    _mc_rets = analyzed["pnl_pct"].dropna().values / 100
    if len(_mc_rets) < 3:
        st.warning("Not enough trades for Monte Carlo.")
    else:
        _n_trades_mc = len(_mc_rets)
        _finals = np.empty(mc_n)
        _curves  = []
        rng = np.random.default_rng(42)
        for _i in range(mc_n):
            _sampled = rng.choice(_mc_rets, size=_n_trades_mc, replace=True)
            _port = initial_capital
            _curve = [_port]
            for _r in _sampled:
                _port += _port * (position_pct / 100) * _r
                _curve.append(_port)
            _finals[_i] = _port
            if _i < 300:
                _curves.append(_curve)
        st.session_state["mc_results"] = {"finals": _finals, "curves": _curves, "n": _n_trades_mc}
        st.session_state["mc_key"] = mc_key

if "mc_results" in st.session_state:
    _mc = st.session_state["mc_results"]
    _finals = _mc["finals"]
    _p10  = float(np.percentile(_finals, 10))
    _p50  = float(np.percentile(_finals, 50))
    _p90  = float(np.percentile(_finals, 90))
    _pct_profit = (_finals > initial_capital).sum() / len(_finals) * 100
    _pct_double = (_finals > initial_capital * 2).sum() / len(_finals) * 100

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("10th Pct (worst likely)",  fmt_money(_p10),
               f"{(_p10 - initial_capital) / initial_capital * 100:+.1f}%", delta_color="off")
    mc2.metric("50th Pct (median)",        fmt_money(_p50),
               f"{(_p50 - initial_capital) / initial_capital * 100:+.1f}%", delta_color="off")
    mc3.metric("90th Pct (best likely)",   fmt_money(_p90),
               f"{(_p90 - initial_capital) / initial_capital * 100:+.1f}%", delta_color="off")
    mc4.metric("% sims profitable",        f"{_pct_profit:.0f}%")
    mc5.metric("% sims doubled",           f"{_pct_double:.0f}%")

    # Fan chart
    fig_mc = go.Figure()
    for _curve in _mc["curves"]:
        fig_mc.add_trace(go.Scatter(
            y=_curve, mode="lines",
            line=dict(color="rgba(230,126,34,0.06)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Percentile curves
    _all_curves = np.array([_mc["curves"][i] for i in range(len(_mc["curves"]))])
    _p10_curve  = np.percentile(_all_curves, 10, axis=0)
    _p50_curve  = np.percentile(_all_curves, 50, axis=0)
    _p90_curve  = np.percentile(_all_curves, 90, axis=0)
    _xs = list(range(len(_p50_curve)))
    fig_mc.add_trace(go.Scatter(y=_p10_curve, x=_xs, name="10th pct (worst likely)",
                                line=dict(color="#e74c3c", width=2.5)))
    fig_mc.add_trace(go.Scatter(y=_p50_curve, x=_xs, name="50th pct (median)",
                                line=dict(color="#f39c12", width=3)))
    fig_mc.add_trace(go.Scatter(y=_p90_curve, x=_xs, name="90th pct (best likely)",
                                line=dict(color="#2ecc71", width=2.5)))
    fig_mc.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                     annotation_text="Starting Capital")
    fig_mc.update_layout(
        title=f"Monte Carlo: {mc_n} simulations of {_mc['n']} trades (bootstrap resampling)",
        xaxis_title="Trade #", yaxis_title="Portfolio ($)",
        yaxis=dict(tickformat="$,.3s"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=440,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Final value distribution histogram
    fig_mc_hist = go.Figure()
    fig_mc_hist.add_trace(go.Histogram(x=_finals, nbinsx=60, name="Final Values",
                                        marker_color="#3498db", opacity=0.8))
    fig_mc_hist.add_vline(x=_p10, line_color="#e74c3c", line_dash="dash",
                          annotation_text=f" P10: {fmt_money(_p10)}", annotation_position="top right")
    fig_mc_hist.add_vline(x=_p50, line_color="#f39c12", line_width=2.5,
                          annotation_text=f" Median: {fmt_money(_p50)}", annotation_position="top right")
    fig_mc_hist.add_vline(x=_p90, line_color="#2ecc71", line_dash="dash",
                          annotation_text=f" P90: {fmt_money(_p90)}", annotation_position="top right")
    fig_mc_hist.add_vline(x=initial_capital, line_color="gray", line_dash="dot",
                          annotation_text=" Start", annotation_position="top left")
    fig_mc_hist.update_layout(title="Distribution of Final Portfolio Values",
                               xaxis_title="Final Value ($)", yaxis_title="Count",
                               xaxis=dict(tickformat="$,.0f"), height=320)
    st.plotly_chart(fig_mc_hist, use_container_width=True)

# ── Quarterly Returns ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📆 Quarterly Returns")

_qa = analyzed.copy()
_qa["rec_date"] = pd.to_datetime(_qa["rec_date"])
_qa["quarter"]  = _qa["rec_date"].dt.to_period("Q").astype(str)

_q_groups = _qa.groupby("quarter")
_q_rows = []
for _q, _grp in _q_groups:
    _qsf = _grp[_grp["outcome"].isin(["success", "failure"])]
    _qws = (_qsf["outcome"] == "success").sum()
    _qls = (_qsf["outcome"] == "failure").sum()
    _qwr = _qws / (_qws + _qls) * 100 if (_qws + _qls) > 0 else None
    # Compounded return for the quarter (using sim logic)
    _qport = 1.0
    for _, _row in _grp.sort_values("rec_date").iterrows():
        if _row["outcome"] in ("success", "failure", "neutral", "ambiguous"):
            _r = (_row["pnl_pct"] or 0) / 100
            _qport += _qport * (position_pct / 100) * _r
    _q_ret = (_qport - 1) * 100
    _q_rows.append({
        "Quarter":       _q,
        "Trades":        len(_grp[_grp["outcome"].isin(["success","failure","neutral","ambiguous"])]),
        "Win Rate":      round(_qwr, 1) if _qwr is not None else None,
        "Return (%)":    round(_q_ret, 2),
        "Avg P&L/trade": round(_grp["pnl_pct"].mean(), 2) if _grp["pnl_pct"].notna().any() else None,
        "Std Dev":       round(_grp["pnl_pct"].std(), 2) if _grp["pnl_pct"].notna().sum() > 1 else None,
    })

_q_df = pd.DataFrame(_q_rows)
if not _q_df.empty:
    _summary_row = {
        "Quarter":       "📊 Average",
        "Trades":        round(_q_df["Trades"].mean(), 1),
        "Win Rate":      round(_q_df["Win Rate"].dropna().mean(), 1),
        "Return (%)":    round(_q_df["Return (%)"].mean(), 2),
        "Avg P&L/trade": round(_q_df["Avg P&L/trade"].dropna().mean(), 2),
        "Std Dev":       round(_q_df["Std Dev"].dropna().mean(), 2),
    }
    _std_row = {
        "Quarter":       "📉 Std Dev (across quarters)",
        "Trades":        round(_q_df["Trades"].std(), 1),
        "Win Rate":      round(_q_df["Win Rate"].dropna().std(), 1),
        "Return (%)":    round(_q_df["Return (%)"].std(), 2),
        "Avg P&L/trade": round(_q_df["Avg P&L/trade"].dropna().std(), 2),
        "Std Dev":       None,
    }
    _q_display = pd.concat([_q_df, pd.DataFrame([_summary_row, _std_row])], ignore_index=True)

    def _color_return(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ""
        return "color: #2ecc71" if val > 0 else "color: #e74c3c"

    st.dataframe(
        _q_display.style.applymap(_color_return, subset=["Return (%)", "Avg P&L/trade"]),
        hide_index=True, use_container_width=True,
    )

    # Bar chart of quarterly returns
    _q_plot = _q_df.dropna(subset=["Return (%)"])
    fig_q = go.Figure(go.Bar(
        x=_q_plot["Quarter"], y=_q_plot["Return (%)"],
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in _q_plot["Return (%)"]],
        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_q.add_hline(y=0, line_color="gray", line_width=0.8)
    fig_q.update_layout(title="Compounded Return per Quarter",
                        xaxis_title="Quarter", yaxis_title="Return (%)", height=350)
    st.plotly_chart(fig_q, use_container_width=True)

# ── AI Analyst ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 AI Strategy Analyst")
st.caption(
    "Claude reads all the computed metrics and gives a detailed opinion on the strategy's "
    "viability for the next 5 years, with full reasoning."
)

if st.button("🧠 Generate AI Analysis", type="primary", key="ai_btn"):
    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic()

        # Build context from all computed metrics
        _q_summary = _q_df[["Quarter", "Return (%)", "Win Rate"]].to_string(index=False) if not _q_df.empty else "N/A"
        _mc_summary = (
            f"10th pct: {fmt_money(_p10)} ({(_p10 - initial_capital)/initial_capital*100:+.1f}%), "
            f"Median: {fmt_money(_p50)} ({(_p50 - initial_capital)/initial_capital*100:+.1f}%), "
            f"90th pct: {fmt_money(_p90)} ({(_p90 - initial_capital)/initial_capital*100:+.1f}%), "
            f"% profitable sims: {_pct_profit:.0f}%"
        ) if "mc_results" in st.session_state else "Monte Carlo not yet run"

        _prompt = f"""You are a professional quantitative trading analyst reviewing a systematic trading strategy.
Analyze ALL the data below and provide a detailed, honest assessment of whether this strategy is likely to remain viable over the next 5 years. Be specific, rigorous, and don't hedge everything — give a clear opinion.

=== STRATEGY OVERVIEW ===
Ticker: {ticker}
Signal type: Algorithm buy recommendations (prediction='buy')
Trade mechanics: Enter at closing price on recommendation day, exit the following trading day.
Neutral/ambiguous trades: exit at same-day close.
Profit Target: {profit_pct}% | Stop Loss: {loss_pct}%
Position size: {position_pct}% of portfolio per trade
Data range analyzed: {results["rec_date"].min()} to {results["rec_date"].max()}

=== PERFORMANCE SUMMARY ===
Total analyzed trades: {total}
Win rate (excl. neutral): {win_rate:.1f}%
Successes: {s} | Failures: {f} | Neutral: {n} | Ambiguous: {a}
Expectancy per trade: {_exp_pct:+.2f}% (≈ {fmt_money(_exp_usd, signed=True)})
Profit Factor: {_pf:.2f}
Max consecutive losses: {max_loss_streak}
Max consecutive wins: {max_win_streak}

=== RISK METRICS ===
Max Drawdown: {_dd_adv:.1f}% (trough: {_dd_adv_date}, recovery: {str(_rec_adv) + ' days' if _rec_adv else 'not recovered'})
Sharpe Ratio (annualized): {_sharpe:.2f}
Sortino Ratio (annualized): {_sortino:.2f}
Calmar Ratio: {_calmar:.2f}
Final portfolio value: {fmt_money(_final_adv)} ({_ann_ret_adv:+.2f}% annualized)

=== MONTE CARLO (bootstrap resampling) ===
{_mc_summary}

=== QUARTERLY RETURNS ===
{_q_summary}

=== MARKET REGIME SENSITIVITY ===
(from SPY/VIX analysis if available — comment on concentration risk if regimes show large divergence)

=== YOUR TASK ===
1. Is the strategy genuinely profitable or does the data suggest luck/overfitting? Give your honest assessment.
2. What are the 2-3 biggest risks to this strategy continuing to work in 2025-2030?
3. What market conditions would break this strategy?
4. Is the position sizing ({position_pct}% per trade) appropriate given the max drawdown and loss streak?
5. Overall verdict: Would you trade this strategy with real money? Scale of 1-10 confidence, with reasoning.

Write in clear, direct English. Be an honest analyst, not a cheerleader."""

        with st.spinner("Claude is analyzing your strategy..."):
            _msg = _client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1500,
                messages=[{"role": "user", "content": _prompt}],
            )
            _analysis = _msg.content[0].text

        st.session_state["ai_analysis"] = _analysis

    except ImportError:
        st.error("anthropic package not installed. Run: `pip install anthropic`")
    except Exception as _e:
        st.error(f"API error: {_e}. Make sure ANTHROPIC_API_KEY is set as an environment variable.")

if "ai_analysis" in st.session_state:
    st.markdown(st.session_state["ai_analysis"])
