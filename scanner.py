"""
Swing Trading Scanner + Backtester
- 3 setup types: Trend Follow, Breakout, Pullback
- Multi-indicator scoring: RSI, MACD, EMA, ADX, Volume
- F&O suitability filter
- 2-3 year backtest
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────
FNO_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC",
    "SBIN","BHARTIARTL","KOTAKBANK","LT","AXISBANK","MARUTI","ASIANPAINT",
    "BAJFINANCE","TITAN","SUNPHARMA","WIPRO","HCLTECH","ULTRACEMCO",
    "NESTLEIND","POWERGRID","NTPC","TATAMOTORS","INDUSINDBK","JSWSTEEL",
    "ADANIENT","ADANIPORTS","HINDALCO","BPCL","ONGC","COALINDIA",
    "CIPLA","DRREDDY","DIVISLAB","APOLLOHOSP","EICHERMOT","BAJAJFINSV",
    "HEROMOTOCO","TECHM","GRASIM","TATACONSUM","BAJAJ-AUTO","BRITANNIA",
    "HDFCLIFE","SBILIFE","PIDILITIND","HAVELLS","DMART","MCDOWELL-N",
]

NIFTY500_EXTRA = [
    "AUROPHARMA","BALKRISIND","BANDHANBNK","BERGEPAINT","BIOCON",
    "CANBK","CHOLAFIN","COFORGE","CONCOR","CROMPTON","DABUR",
    "FEDERALBNK","GODREJCP","GODREJPROP","GUJGASLTD","IDFCFIRSTB",
    "INDUSTOWER","IRCTC","JKCEMENT","JUBLFOOD","LICHSGFIN","LUPIN",
    "MARICO","MPHASIS","MRF","MUTHOOTFIN","NAUKRI","OBEROIRLTY",
    "PAGEIND","PERSISTENT","PETRONET","PFC","PNB","POLYCAB",
    "PVRINOX","RAMCOCEM","RECLTD","SAIL","SHREECEM","SIEMENS",
    "SRF","TATACOMM","TATACHEM","TATAELXSI","TATAPOWER","TORNTPHARM",
    "TORNTPOWER","TRENT","UBL","UNIONBANK","UPL","VEDL",
    "VOLTAS","ZOMATO","ZYDUSLIFE",
]

ALL_SYMBOLS  = list(set(FNO_STOCKS + NIFTY500_EXTRA))
FNO_SET      = set(FNO_STOCKS)

# ─────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    # EMAs
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    df["macd"]  = ema12 - ema26
    df["signal"]= df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal"]

    # ADX
    tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    up   = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    pdi  = 100 * up.rolling(14).mean()  / atr.replace(0, np.nan)
    ndi  = 100 * down.rolling(14).mean()/ atr.replace(0, np.nan)
    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    df["adx"] = dx.rolling(14).mean()
    df["atr"] = atr

    # Volume
    df["vol_avg20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_avg20"].replace(0, np.nan)

    # 52-week high/low
    df["high52"] = h.rolling(252).max()
    df["low52"]  = l.rolling(252).min()

    # Bollinger Bands
    sma20        = c.rolling(20).mean()
    std20        = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    return df


# ─────────────────────────────────────────────────────────────
#  SETUP DETECTION
# ─────────────────────────────────────────────────────────────
def detect_setups(df: pd.DataFrame, symbol: str) -> dict | None:
    if len(df) < 220:
        return None

    row   = df.iloc[-1]   # latest day
    prev  = df.iloc[-2]
    c     = float(row["Close"])
    o     = float(row["Open"])
    h     = float(row["High"])
    l     = float(row["Low"])

    rsi      = float(row["rsi"])
    macd     = float(row["macd"])
    sig      = float(row["signal"])
    macd_h   = float(row["macd_hist"])
    adx      = float(row["adx"])
    ema20    = float(row["ema20"])
    ema50    = float(row["ema50"])
    ema200   = float(row["ema200"])
    atr      = float(row["atr"])
    vol_r    = float(row["vol_ratio"])
    high52   = float(row["high52"])
    low52    = float(row["low52"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])

    setups   = []
    score    = 0

    # ── SETUP 1: TREND FOLLOW ──
    # Price above EMA20 > EMA50 > EMA200, ADX > 25, RSI 50-70
    trend_bull = (c > ema20 > ema50 > ema200)
    trend_bear = (c < ema20 < ema50 < ema200)

    if trend_bull and adx > 25 and 50 < rsi < 75 and macd > sig:
        setups.append("TREND_BULL")
        score += 3
    if trend_bear and adx > 25 and 25 < rsi < 50 and macd < sig:
        setups.append("TREND_BEAR")
        score += 3

    # ── SETUP 2: BREAKOUT ──
    # Price breaking 52-week high with volume surge
    near_52high = c >= high52 * 0.98
    near_52low  = c <= low52  * 1.02
    vol_surge   = vol_r > 1.5

    if near_52high and vol_surge and rsi > 55 and adx > 20:
        setups.append("BREAKOUT_BULL")
        score += 4
    if near_52low and vol_surge and rsi < 45 and adx > 20:
        setups.append("BREAKOUT_BEAR")
        score += 4

    # ── SETUP 3: PULLBACK IN TREND ──
    # In uptrend, price pulls back to EMA20/50, RSI cools to 40-55, then bounces
    in_uptrend   = ema20 > ema50 > ema200
    in_downtrend = ema20 < ema50 < ema200

    near_ema20 = abs(c - ema20) / ema20 < 0.02
    near_ema50 = abs(c - ema50) / ema50 < 0.02

    prev_rsi = float(prev["rsi"])
    rsi_bounce   = rsi > prev_rsi and rsi > 45   # RSI turning up
    rsi_rollover = rsi < prev_rsi and rsi < 55   # RSI turning down

    if in_uptrend and (near_ema20 or near_ema50) and rsi_bounce and macd_h > float(prev["macd_hist"]):
        setups.append("PULLBACK_BULL")
        score += 3
    if in_downtrend and (near_ema20 or near_ema50) and rsi_rollover and macd_h < float(prev["macd_hist"]):
        setups.append("PULLBACK_BEAR")
        score += 3

    if not setups:
        return None

    # ── INDICATOR SCORE (out of 10) ──
    ind_score = 0
    # RSI confirmation
    if any("BULL" in s for s in setups):
        if 50 < rsi < 70:  ind_score += 2
        elif rsi >= 70:    ind_score += 1
    else:
        if 30 < rsi < 50:  ind_score += 2
        elif rsi <= 30:    ind_score += 1

    # MACD
    if macd_h > 0 and any("BULL" in s for s in setups): ind_score += 2
    if macd_h < 0 and any("BEAR" in s for s in setups): ind_score += 2

    # ADX (trend strength)
    if adx > 30:   ind_score += 2
    elif adx > 20: ind_score += 1

    # Volume
    if vol_r > 2:    ind_score += 2
    elif vol_r > 1.5: ind_score += 1

    # EMA alignment bonus
    if trend_bull or trend_bear: ind_score += 2

    ind_score = min(ind_score, 10)

    # ── F&O SUITABILITY ──
    is_fno     = symbol in FNO_SET
    fno_score  = 0
    fno_reason = []

    if is_fno:
        if adx > 30:
            fno_score += 3
            fno_reason.append(f"Strong trend ADX={adx:.0f}")
        if vol_r > 2:
            fno_score += 2
            fno_reason.append(f"High volume {vol_r:.1f}x")
        if near_52high or near_52low:
            fno_score += 2
            fno_reason.append("Near 52W extreme")
        if "BREAKOUT" in " ".join(setups):
            fno_score += 2
            fno_reason.append("Breakout setup")
        if ind_score >= 7:
            fno_score += 1
            fno_reason.append("High indicator confluence")

    fno_tag = ""
    if is_fno and fno_score >= 7:
        fno_tag = "🔥 STRONG F&O"
    elif is_fno and fno_score >= 4:
        fno_tag = "⚡ F&O POSSIBLE"
    elif is_fno:
        fno_tag = "📋 F&O (weak setup)"

    # ── SL & TARGET ──
    direction = "BULL" if any("BULL" in s for s in setups) else "BEAR"
    sl     = round(c - 1.5 * atr, 2) if direction == "BULL" else round(c + 1.5 * atr, 2)
    tgt1   = round(c + 2 * atr,   2) if direction == "BULL" else round(c - 2 * atr, 2)
    tgt2   = round(c + 3.5 * atr, 2) if direction == "BULL" else round(c - 3.5 * atr, 2)
    risk   = abs(c - sl)

    return {
        "symbol":      symbol,
        "date":        df.index[-1].strftime("%Y-%m-%d"),
        "price":       round(c, 2),
        "setups":      " + ".join(setups),
        "direction":   direction,
        "score":       ind_score,
        "fno_tag":     fno_tag,
        "fno_reason":  ", ".join(fno_reason) if fno_reason else "—",
        "rsi":         round(rsi, 1),
        "macd_hist":   round(macd_h, 3),
        "adx":         round(adx, 1),
        "vol_ratio":   round(vol_r, 2),
        "ema20":       round(ema20, 2),
        "ema50":       round(ema50, 2),
        "atr":         round(atr, 2),
        "sl":          sl,
        "target1":     tgt1,
        "target2":     tgt2,
        "risk_pts":    round(risk, 2),
        "rr_t1":       round(abs(tgt1 - c) / risk, 2) if risk > 0 else 0,
        "rr_t2":       round(abs(tgt2 - c) / risk, 2) if risk > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────
#  FETCH DATA
# ─────────────────────────────────────────────────────────────
def fetch_data(symbol: str, years: int = 3) -> pd.DataFrame | None:
    try:
        df = yf.download(
            f"{symbol}.NS",
            period=f"{years}y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < 220:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None


# ─────────────────────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────────────────────
def backtest_symbol(df: pd.DataFrame, symbol: str) -> list:
    trades = []
    if len(df) < 250:
        return trades

    df = compute_indicators(df.copy())

    for i in range(220, len(df) - 1):
        window = df.iloc[:i+1]
        setup  = detect_setups(window, symbol)
        if setup is None:
            continue

        # Entry next day open
        entry_row = df.iloc[i + 1]
        entry     = float(entry_row["Open"])
        atr       = float(window.iloc[-1]["atr"])
        direction = setup["direction"]

        sl   = round(entry - 1.5 * atr, 2) if direction == "BULL" else round(entry + 1.5 * atr, 2)
        tgt1 = round(entry + 2 * atr,   2) if direction == "BULL" else round(entry - 2 * atr, 2)
        tgt2 = round(entry + 3.5 * atr, 2) if direction == "BULL" else round(entry - 3.5 * atr, 2)
        risk = abs(entry - sl)

        if risk <= 0:
            continue

        # Walk forward max 20 days
        result   = "OPEN"
        exit_px  = None
        exit_day = None
        hit_tgt1 = False

        for j in range(i + 2, min(i + 22, len(df))):
            row  = df.iloc[j]
            high = float(row["High"])
            low  = float(row["Low"])

            if direction == "BULL":
                if low <= sl:
                    result   = "SL"
                    exit_px  = sl
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break
                if not hit_tgt1 and high >= tgt1:
                    hit_tgt1 = True
                if hit_tgt1 and high >= tgt2:
                    result   = "TGT2"
                    exit_px  = tgt2
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break
                if hit_tgt1 and low <= tgt1:
                    result   = "TGT1"
                    exit_px  = tgt1
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break
            else:
                if high >= sl:
                    result   = "SL"
                    exit_px  = sl
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break
                if not hit_tgt1 and low <= tgt1:
                    hit_tgt1 = True
                if hit_tgt1 and low <= tgt2:
                    result   = "TGT2"
                    exit_px  = tgt2
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break
                if hit_tgt1 and high >= tgt1:
                    result   = "TGT1"
                    exit_px  = tgt1
                    exit_day = df.index[j].strftime("%Y-%m-%d")
                    break

        if result == "OPEN":
            exit_px  = float(df.iloc[min(i+21, len(df)-1)]["Close"])
            exit_day = df.index[min(i+21, len(df)-1)].strftime("%Y-%m-%d")

        pnl    = round(exit_px - entry if direction == "BULL" else entry - exit_px, 2)
        pnl_pct= round(pnl / entry * 100, 2)

        trades.append({
            "symbol":    symbol,
            "setup":     setup["setups"],
            "direction": direction,
            "entry_date":df.index[i+1].strftime("%Y-%m-%d"),
            "exit_date": exit_day,
            "entry":     round(entry, 2),
            "exit":      round(exit_px, 2),
            "sl":        sl,
            "tgt1":      tgt1,
            "tgt2":      tgt2,
            "result":    result,
            "pnl_pts":   pnl,
            "pnl_pct":   pnl_pct,
            "score":     setup["score"],
            "fno_tag":   setup["fno_tag"],
            "win":       1 if pnl > 0 else 0,
        })

    return trades


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_scanner():
    print("\n" + "="*60)
    print("  SWING SCANNER — Today's Setups")
    print("="*60)

    results = []
    for i, sym in enumerate(ALL_SYMBOLS):
        print(f"[{i+1}/{len(ALL_SYMBOLS)}] {sym}...", end=" ", flush=True)
        df = fetch_data(sym, years=3)
        if df is None:
            print("❌")
            continue
        df  = compute_indicators(df)
        res = detect_setups(df, sym)
        if res:
            results.append(res)
            print(f"✅ {res['setups']} | Score:{res['score']}/10 {res['fno_tag']}")
        else:
            print("—")

    if not results:
        print("No setups today.")
        return

    df_out = pd.DataFrame(results).sort_values(["score","fno_tag"], ascending=[False, True])
    df_out.to_csv(Path(__file__).parent / "swing_signals.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  {len(results)} SETUPS FOUND TODAY")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x["score"])[:10]:
        print(f"  {r['symbol']:15} {r['setups']:30} Score:{r['score']}/10  {r['fno_tag']}")
        print(f"  {'':15} Entry:₹{r['price']}  SL:₹{r['sl']}  T1:₹{r['target1']}  T2:₹{r['target2']}")
        print()
    print(f"  Full list saved to swing_signals.csv")


def run_backtest():
    print("\n" + "="*60)
    print("  SWING BACKTEST — 3 Years")
    print("="*60)

    all_trades = []
    for i, sym in enumerate(ALL_SYMBOLS):
        print(f"[{i+1}/{len(ALL_SYMBOLS)}] {sym}...", end=" ", flush=True)
        df = fetch_data(sym, years=3)
        if df is None:
            print("❌")
            continue
        trades = backtest_symbol(df, sym)
        all_trades.extend(trades)
        print(f"✅ {len(trades)} trades")

    if not all_trades:
        print("No trades found.")
        return

    df_t = pd.DataFrame(all_trades)
    total    = len(df_t)
    wins     = df_t["win"].sum()
    wr       = round(wins/total*100, 1)
    avg_win  = round(df_t[df_t["win"]==1]["pnl_pct"].mean(), 2)
    avg_loss = round(df_t[df_t["win"]==0]["pnl_pct"].mean(), 2)
    tgt2_hits= len(df_t[df_t["result"]=="TGT2"])
    tgt1_hits= len(df_t[df_t["result"]=="TGT1"])
    sl_hits  = len(df_t[df_t["result"]=="SL"])

    # by setup type
    for setup_type in ["TREND","BREAKOUT","PULLBACK"]:
        sub = df_t[df_t["setup"].str.contains(setup_type)]
        if len(sub):
            swr = round(sub["win"].sum()/len(sub)*100,1)
            print(f"  {setup_type}: {len(sub)} trades, WR={swr}%")

    print(f"\n{'='*60}")
    print(f"  BACKTEST SUMMARY (3 Years)")
    print(f"{'='*60}")
    print(f"  Total Trades  : {total}")
    print(f"  Win Rate      : {wr}%")
    print(f"  Avg Win       : +{avg_win}%")
    print(f"  Avg Loss      : {avg_loss}%")
    print(f"  TGT2 hits     : {tgt2_hits} ({round(tgt2_hits/total*100,1)}%)")
    print(f"  TGT1 hits     : {tgt1_hits} ({round(tgt1_hits/total*100,1)}%)")
    print(f"  SL hits       : {sl_hits}  ({round(sl_hits/total*100,1)}%)")
    print(f"{'='*60}")

    out = Path(__file__).parent
    df_t.to_csv(out / "swing_backtest_trades.csv", index=False)

    daily = df_t.groupby("entry_date").agg(
        trades=("symbol","count"), wins=("win","sum"), avg_pnl=("pnl_pct","mean")
    ).reset_index()
    daily.to_csv(out / "swing_backtest_daily.csv", index=False)

    stock = df_t.groupby("symbol").agg(
        trades=("win","count"), wins=("win","sum"), total_pnl=("pnl_pct","sum")
    ).reset_index()
    stock["wr"] = (stock["wins"]/stock["trades"]*100).round(1)
    stock.sort_values("total_pnl", ascending=False).to_csv(out / "swing_backtest_by_stock.csv", index=False)

    print(f"\n  Files saved:")
    print(f"  swing_backtest_trades.csv   — every trade")
    print(f"  swing_backtest_daily.csv    — day by day")
    print(f"  swing_backtest_by_stock.csv — best stocks\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "backtest":
        run_backtest()
    else:
        run_scanner()
