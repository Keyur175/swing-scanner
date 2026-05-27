"""
VCP (Volatility Contraction Pattern) — Mark Minervini SEPA
Relaxed detection: catches 2C and 3C patterns on daily chart
"""

import numpy as np
import pandas as pd


def find_swing_highs(series: pd.Series, window: int = 10):
    """Find meaningful swing highs using larger window."""
    highs = []
    for i in range(window, len(series) - window):
        seg = series.iloc[i - window: i + window + 1]
        if series.iloc[i] == seg.max():
            highs.append((series.index[i], float(series.iloc[i])))
    return highs


def find_swing_lows(series: pd.Series, window: int = 10):
    """Find meaningful swing lows using larger window."""
    lows = []
    for i in range(window, len(series) - window):
        seg = series.iloc[i - window: i + window + 1]
        if series.iloc[i] == seg.min():
            lows.append((series.index[i], float(series.iloc[i])))
    return lows


def detect_vcp(df: pd.DataFrame, symbol: str) -> dict | None:
    """
    VCP Detection:
    1. Stock in uptrend (price > EMA150 > EMA200)
    2. Within 30% of 52-week high
    3. Identify base: look back up to 52 weeks
    4. Find 2-3 contracting pullbacks within the base
       - Each correction shallower than previous (depth contracting)
       - Volume drying up in each successive contraction
    5. Price near the pivot (tightest area = breakout zone)
    """
    if len(df) < 200:
        return None

    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    # ── Trend filter ──
    ema150 = c.ewm(span=150, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()

    price  = float(c.iloc[-1])
    e150   = float(ema150.iloc[-1])
    e200   = float(ema200.iloc[-1])
    e50    = float(ema50.iloc[-1])

    # Must be in uptrend — relaxed: price > EMA200 at minimum
    if price < e200:
        return None

    # 52-week metrics
    lookback_52 = min(252, len(df))
    high52 = float(h.iloc[-lookback_52:].max())
    low52  = float(l.iloc[-lookback_52:].min())
    vol_avg= float(v.rolling(20).mean().iloc[-1])

    # Must be within 30% of 52-week high
    if price < high52 * 0.70:
        return None

    rs_rank = round((price - low52) / (high52 - low52) * 100, 1) if high52 != low52 else 50

    # ── Find base: look at last 200 days ──
    base_days = min(200, len(df))
    base_df   = df.iloc[-base_days:].copy()
    bc = base_df["Close"].squeeze()
    bh = base_df["High"].squeeze()
    bl = base_df["Low"].squeeze()
    bv = base_df["Volume"].squeeze()

    # Use window=10 to find meaningful swings
    swing_highs = find_swing_highs(bc, window=10)
    swing_lows  = find_swing_lows(bc,  window=10)

    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return None

    # ── Measure pullbacks between consecutive swing highs ──
    pullbacks = []
    for i in range(len(swing_highs) - 1):
        ph_date, ph_val = swing_highs[i]
        nh_date, nh_val = swing_highs[i + 1]

        # Lowest point between the two highs
        between_lows = [(d, val) for d, val in swing_lows if ph_date < d < nh_date]
        if not between_lows:
            # Use min close between the two dates
            mask = (base_df.index > ph_date) & (base_df.index < nh_date)
            if mask.sum() == 0:
                continue
            seg_close = bc[mask]
            trough_val = float(seg_close.min())
            trough_date= seg_close.idxmin()
        else:
            trough_date, trough_val = min(between_lows, key=lambda x: x[1])

        depth_pct = (ph_val - trough_val) / ph_val * 100
        if depth_pct <= 0:
            continue

        duration = max((trough_date - ph_date).days, 1)

        # Volume during contraction vs average
        try:
            mask = (base_df.index >= ph_date) & (base_df.index <= trough_date)
            avg_vol_c = float(bv[mask].mean()) if mask.sum() > 0 else vol_avg
            vol_ratio = avg_vol_c / vol_avg if vol_avg > 0 else 1.0
        except:
            vol_ratio = 1.0

        pullbacks.append({
            "peak":      round(ph_val, 2),
            "trough":    round(trough_val, 2),
            "depth_pct": round(depth_pct, 1),
            "duration":  duration,
            "vol_ratio": round(vol_ratio, 2),
        })

    if len(pullbacks) < 2:
        return None

    # ── Find best contracting sequence ──
    # Look for 3C first, then 2C
    best_pattern = None

    # Try 3C: any 3 consecutive pullbacks that contract
    for i in range(len(pullbacks) - 2):
        p1, p2, p3 = pullbacks[i], pullbacks[i+1], pullbacks[i+2]
        # Depths contracting (relaxed: allow up to 3% violation)
        d_ok = (p1["depth_pct"] > p2["depth_pct"] * 0.80 and
                p2["depth_pct"] > p3["depth_pct"] * 0.80 and
                p3["depth_pct"] < 25)  # last pullback must be < 25%
        if d_ok:
            best_pattern = ("3C-VCP", [p1, p2, p3])

    # Try 2C if no 3C found
    if not best_pattern:
        for i in range(len(pullbacks) - 1):
            p1, p2 = pullbacks[i], pullbacks[i+1]
            d_ok = (p1["depth_pct"] > p2["depth_pct"] * 0.80 and
                    p2["depth_pct"] < 20)
            if d_ok:
                best_pattern = ("2C-VCP", [p1, p2])

    if not best_pattern:
        return None

    pattern_type, pattern_pbs = best_pattern
    last_pb = pattern_pbs[-1]

    # ── Pivot = most recent swing high ──
    pivot = float(swing_highs[-1][1])

    # Distance from current price to pivot
    dist_from_pivot = round(abs(pivot - price) / pivot * 100, 1)
    near_pivot = dist_from_pivot <= 7  # within 7% above OR below pivot  # within 7% for relaxed

    # Volume dry-up in final contraction
    vol_drying = last_pb["vol_ratio"] < 0.85

    # Tightness: how tight is the last contraction
    tightness = round(max(1, min(10, 10 - last_pb["depth_pct"] / 3)), 1)

    # ── VCP Score ──
    vcp_score = 0
    if pattern_type == "3C-VCP": vcp_score += 4
    else:                         vcp_score += 2
    if near_pivot:                vcp_score += 2
    if vol_drying:                vcp_score += 2
    if rs_rank > 70:              vcp_score += 1
    if price > e150:              vcp_score += 1
    # Extra: each subsequent pullback meaningfully smaller
    if len(pattern_pbs) >= 2:
        meaningful = all(
            pattern_pbs[i]["depth_pct"] > pattern_pbs[i+1]["depth_pct"] + 2
            for i in range(len(pattern_pbs)-1)
        )
        if meaningful: vcp_score += 1
    vcp_score = min(vcp_score, 10)

    # ── ATR for SL/Target ──
    tr  = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # SL = below last trough of final contraction
    sl      = round(last_pb["trough"] - 0.5 * atr, 2)
    risk    = abs(price - sl)
    target1 = round(pivot + last_pb["depth_pct"] / 100 * pivot, 2)
    target2 = round(pivot + 2 * last_pb["depth_pct"] / 100 * pivot, 2)

    return {
        "symbol":       symbol,
        "setup":        "VCP",
        "pattern_type": pattern_type,
        "price":        round(price, 2),
        "pivot":        round(pivot, 2),
        "dist_pivot":   dist_from_pivot,
        "near_pivot":   near_pivot,
        "contractions": len(pattern_pbs),
        "depths":       [p["depth_pct"] for p in pattern_pbs],
        "durations":    [p["duration"]  for p in pattern_pbs],
        "vol_drying":   vol_drying,
        "vol_ratio":    last_pb["vol_ratio"],
        "rs_rank":      rs_rank,
        "tightness":    tightness,
        "vcp_score":    vcp_score,
        "ema50":        round(e50, 2),
        "ema150":       round(e150, 2),
        "ema200":       round(e200, 2),
        "atr":          round(atr, 2),
        "sl":           sl,
        "target1":      target1,
        "target2":      target2,
        "rr1":          round(abs(target1 - price) / risk, 1) if risk > 0 else 0,
        "rr2":          round(abs(target2 - price) / risk, 1) if risk > 0 else 0,
        "direction":    "BULL",
        "date":         df.index[-1].strftime("%Y-%m-%d"),
    }
