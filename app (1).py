"""
Swing Trading Scanner — Streamlit Dashboard
Nifty 500 + F&O | 4 Setups: Trend + Breakout + Pullback + VCP
"""

import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Swing Scanner", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID", "")

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
    "CANBK","CHOLAFIN","COFORGE","CONCUR","CROMPTON","DABUR",
    "FEDERALBNK","GODREJCP","GODREJPROP","GUJGASLTD","IDFCFIRSTB",
    "INDUSTOWER","IRCTC","JKCEMENT","JUBLFOOD","LICHSGFIN","LUPIN",
    "MARICO","MPHASIS","MRF","MUTHOOTFIN","NAUKRI","OBEROIRLTY",
    "PAGEIND","PERSISTENT","PETRONET","PFC","PNB","POLYCAB",
    "PVRINOX","RAMCOCEM","RECLTD","SAIL","SHREECEM","SIEMENS",
    "SRF","TATACOMM","TATACHEM","TATAELXSI","TATAPOWER","TORNTPHARM",
    "TORNTPOWER","TRENT","UBL","UNIONBANK","UPL","VEDL",
    "VOLTAS","ZOMATO","ZYDUSLIFE",
]
ALL_SYMBOLS = list(set(FNO_STOCKS + NIFTY500_EXTRA))
FNO_SET     = set(FNO_STOCKS)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Bebas+Neue&display=swap');
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0.5rem 0.8rem 2rem!important;max-width:500px;margin:auto}
.header{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:3px;
        background:linear-gradient(135deg,#00e676,#3b82f6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:2px}
.subhead{font-size:0.72rem;color:#64748b;margin-bottom:12px}
.card{background:#111520;border-radius:14px;padding:14px;margin-bottom:10px;border-left:4px solid}
.card.bull{border-color:#00e676}.card.bear{border-color:#ff3d5a}.card.vcp{border-color:#a855f7}
.card.fno-strong{border-color:#ff6d00}
.sym{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:1px}
.tag{border-radius:20px;padding:2px 10px;font-size:0.65rem;font-weight:700;margin-left:6px}
.tag-bull{background:rgba(0,230,118,0.12);color:#00e676;border:1px solid rgba(0,230,118,0.3)}
.tag-bear{background:rgba(255,61,90,0.12);color:#ff3d5a;border:1px solid rgba(255,61,90,0.3)}
.tag-vcp{background:rgba(168,85,247,0.12);color:#a855f7;border:1px solid rgba(168,85,247,0.3)}
.tag-fno{background:rgba(255,109,0,0.15);color:#ff6d00;border:1px solid rgba(255,109,0,0.3)}
.tag-fno2{background:rgba(255,179,0,0.12);color:#ffb300;border:1px solid rgba(255,179,0,0.3)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:8px 0}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin:5px 0}
.cell{background:#0a0c10;border-radius:8px;padding:7px 4px;text-align:center}
.cv{font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700}
.cl{font-size:0.57rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-top:1px}
.score-bar{height:6px;border-radius:3px;background:#1e2535;margin:6px 0}
.score-fill{height:6px;border-radius:3px}
.setup-tag{font-size:0.62rem;background:#1e2535;color:#94a3b8;border-radius:6px;
           padding:2px 8px;margin:2px;display:inline-block}
.stat-row{display:flex;gap:8px;margin-bottom:12px}
.stat-box{flex:1;background:#111520;border-radius:10px;padding:10px;text-align:center;border:1px solid #1e2535}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700}
.stat-lbl{font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=5)
    except: pass


def compute_indicators(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()
    df["ema20"]   = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]   = c.ewm(span=50,  adjust=False).mean()
    df["ema150"]  = c.ewm(span=150, adjust=False).mean()
    df["ema200"]  = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal"]
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up  = h.diff().clip(lower=0); dn = (-l.diff()).clip(lower=0)
    pdi = 100*up.rolling(14).mean()/atr.replace(0,np.nan)
    ndi = 100*dn.rolling(14).mean()/atr.replace(0,np.nan)
    df["adx"]      = (100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df["atr"]      = atr
    df["vol_avg"]  = v.rolling(20).mean()
    df["vol_ratio"]= v / df["vol_avg"].replace(0, np.nan)
    df["high52"]   = h.rolling(252).max()
    df["low52"]    = l.rolling(252).min()
    return df


# ─────────────────────────────────────────────────────────────
#  SETUP 1-3: TREND / BREAKOUT / PULLBACK
# ─────────────────────────────────────────────────────────────
def detect_swing(df, symbol):
    if len(df) < 220: return None
    row  = df.iloc[-1]; prev = df.iloc[-2]
    c    = float(row["Close"])
    rsi  = float(row["rsi"]);   mh   = float(row["macd_hist"])
    adx  = float(row["adx"]);   atr  = float(row["atr"])
    e20  = float(row["ema20"]); e50  = float(row["ema50"])
    e200 = float(row["ema200"]); sig = float(row["signal"])
    macd = float(row["macd"]);  vr   = float(row["vol_ratio"])
    h52  = float(row["high52"]); l52 = float(row["low52"])
    prsi = float(prev["rsi"]);  pmh  = float(prev["macd_hist"])

    setups = []; score = 0

    # Trend
    if c>e20>e50>e200 and adx>25 and 50<rsi<75 and macd>sig:
        setups.append("TREND_BULL"); score+=3
    if c<e20<e50<e200 and adx>25 and 25<rsi<50 and macd<sig:
        setups.append("TREND_BEAR"); score+=3

    # Breakout
    if c>=h52*0.98 and vr>1.5 and rsi>55 and adx>20:
        setups.append("BREAKOUT_BULL"); score+=4
    if c<=l52*1.02 and vr>1.5 and rsi<45 and adx>20:
        setups.append("BREAKOUT_BEAR"); score+=4

    # Pullback
    in_up = e20>e50>e200; in_dn = e20<e50<e200
    n20 = abs(c-e20)/e20<0.02; n50 = abs(c-e50)/e50<0.02
    if in_up and (n20 or n50) and rsi>prsi and rsi>45 and mh>pmh:
        setups.append("PULLBACK_BULL"); score+=3
    if in_dn and (n20 or n50) and rsi<prsi and rsi<55 and mh<pmh:
        setups.append("PULLBACK_BEAR"); score+=3

    if not setups: return None

    # Indicator score
    bull = any("BULL" in s for s in setups)
    ind  = 0
    if bull:
        if 50<rsi<70: ind+=2
        elif rsi>=70: ind+=1
    else:
        if 30<rsi<50: ind+=2
        elif rsi<=30: ind+=1
    if mh>0 and bull: ind+=2
    if mh<0 and not bull: ind+=2
    if adx>30: ind+=2
    elif adx>20: ind+=1
    if vr>2: ind+=2
    elif vr>1.5: ind+=1
    if (c>e20>e50>e200) or (c<e20<e50<e200): ind+=2
    ind = min(ind, 10)

    # F&O tag
    is_fno = symbol in FNO_SET; fs=0; fr=[]
    if is_fno:
        if adx>30: fs+=3; fr.append(f"ADX {adx:.0f}")
        if vr>2:   fs+=2; fr.append(f"Vol {vr:.1f}x")
        if c>=h52*0.98 or c<=l52*1.02: fs+=2; fr.append("52W extreme")
        if "BREAKOUT" in " ".join(setups): fs+=2; fr.append("Breakout")
        if ind>=7: fs+=1; fr.append("High confluence")
    fno_tag = ("🔥 STRONG F&O" if fs>=7 else "⚡ F&O POSSIBLE" if fs>=4 else "📋 F&O") if is_fno else ""

    direction = "BULL" if bull else "BEAR"
    sl  = round(c-1.5*atr,2) if bull else round(c+1.5*atr,2)
    t1  = round(c+2*atr,2)   if bull else round(c-2*atr,2)
    t2  = round(c+3.5*atr,2) if bull else round(c-3.5*atr,2)
    risk= abs(c-sl)

    return dict(symbol=symbol, price=round(c,2),
                setups=" + ".join(setups), setup_type="swing",
                direction=direction, score=ind, fno_tag=fno_tag,
                fno_reasons=", ".join(fr) if fr else "—",
                rsi=round(rsi,1), adx=round(adx,1), vol_ratio=round(vr,2),
                atr=round(atr,2), sl=sl, target1=t1, target2=t2,
                rr1=round(abs(t1-c)/risk,1) if risk>0 else 0,
                rr2=round(abs(t2-c)/risk,1) if risk>0 else 0,
                date=df.index[-1].strftime("%Y-%m-%d"))


# ─────────────────────────────────────────────────────────────
#  SETUP 4: VCP (built-in, no import needed)
# ─────────────────────────────────────────────────────────────
def find_swing_highs(series, window=10):
    highs = []
    for i in range(window, len(series)-window):
        seg = series.iloc[i-window:i+window+1]
        if series.iloc[i] == seg.max():
            highs.append((series.index[i], float(series.iloc[i])))
    return highs

def find_swing_lows(series, window=10):
    lows = []
    for i in range(window, len(series)-window):
        seg = series.iloc[i-window:i+window+1]
        if series.iloc[i] == seg.min():
            lows.append((series.index[i], float(series.iloc[i])))
    return lows

def detect_vcp(df, symbol):
    if len(df) < 200: return None
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    e150 = c.ewm(span=150, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    e50  = c.ewm(span=50,  adjust=False).mean()
    price = float(c.iloc[-1])

    if price < float(e200.iloc[-1]): return None

    lb    = min(252, len(df))
    h52   = float(h.iloc[-lb:].max())
    l52   = float(l.iloc[-lb:].min())
    if price < h52 * 0.70: return None

    rs_rank  = round((price-l52)/(h52-l52)*100,1) if h52!=l52 else 50
    vol_avg  = float(v.rolling(20).mean().iloc[-1])
    base_df  = df.iloc[-min(200,len(df)):]
    bc = base_df["Close"].squeeze()
    bv = base_df["Volume"].squeeze()

    swing_highs = find_swing_highs(bc, window=10)
    swing_lows  = find_swing_lows(bc,  window=10)
    if len(swing_highs)<2 or len(swing_lows)<1: return None

    pullbacks = []
    for i in range(len(swing_highs)-1):
        ph_date, ph_val = swing_highs[i]
        nh_date, nh_val = swing_highs[i+1]
        btw = [(d,val) for d,val in swing_lows if ph_date<d<nh_date]
        if not btw:
            mask = (base_df.index>ph_date)&(base_df.index<nh_date)
            if mask.sum()==0: continue
            seg = bc[mask]; tv=float(seg.min()); td=seg.idxmin()
        else:
            td,tv = min(btw, key=lambda x:x[1])
        depth = (ph_val-tv)/ph_val*100
        if depth<=0: continue
        dur = max((td-ph_date).days,1)
        try:
            mask = (base_df.index>=ph_date)&(base_df.index<=td)
            vr   = float(bv[mask].mean())/vol_avg if mask.sum()>0 and vol_avg>0 else 1.0
        except: vr=1.0
        pullbacks.append({"depth_pct":round(depth,1),"duration":dur,"vol_ratio":round(vr,2),"trough":round(tv,2)})

    if len(pullbacks)<2: return None

    best = None
    for i in range(len(pullbacks)-2):
        p1,p2,p3 = pullbacks[i],pullbacks[i+1],pullbacks[i+2]
        if p1["depth_pct"]>p2["depth_pct"]*0.80 and p2["depth_pct"]>p3["depth_pct"]*0.80 and p3["depth_pct"]<25:
            best = ("3C-VCP",[p1,p2,p3])
    if not best:
        for i in range(len(pullbacks)-1):
            p1,p2 = pullbacks[i],pullbacks[i+1]
            if p1["depth_pct"]>p2["depth_pct"]*0.80 and p2["depth_pct"]<20:
                best = ("2C-VCP",[p1,p2])
    if not best: return None

    ptype, pbs = best
    last_pb = pbs[-1]
    pivot   = float(swing_highs[-1][1])
    dist    = round(abs(pivot-price)/pivot*100,1)
    near_pv = dist<=7
    vol_dry = last_pb["vol_ratio"]<0.85
    tight   = round(max(1,min(10,10-last_pb["depth_pct"]/3)),1)

    vs = 0
    vs += 4 if ptype=="3C-VCP" else 2
    if near_pv:  vs+=2
    if vol_dry:  vs+=2
    if rs_rank>70: vs+=1
    if price>float(e150.iloc[-1]): vs+=1
    vs = min(vs,10)
    if vs<4: return None

    tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    sl  = round(last_pb["trough"]-0.5*atr,2)
    risk= abs(price-sl)
    t1  = round(pivot+last_pb["depth_pct"]/100*pivot,2)
    t2  = round(pivot+2*last_pb["depth_pct"]/100*pivot,2)

    is_fno  = symbol in FNO_SET
    fno_tag = ""
    if is_fno:
        fno_tag = "🔥 STRONG F&O" if vs>=7 else "⚡ F&O POSSIBLE" if vs>=5 else "📋 F&O"
    fno_r = f"Near pivot {dist}% | RS {rs_rank} | {'Vol dry ✅' if vol_dry else 'Vol normal'}"

    depths_str = "→".join(str(p["depth_pct"])+"%" for p in pbs)

    return dict(symbol=symbol, price=round(price,2),
                setups=f"{ptype} ({depths_str})", setup_type="vcp",
                direction="BULL", score=vs, fno_tag=fno_tag,
                fno_reasons=fno_r, rsi="—", adx="—",
                vol_ratio=round(last_pb["vol_ratio"],2),
                atr=round(atr,2), sl=sl, target1=t1, target2=t2,
                rr1=round(abs(t1-price)/risk,1) if risk>0 else 0,
                rr2=round(abs(t2-price)/risk,1) if risk>0 else 0,
                date=df.index[-1].strftime("%Y-%m-%d"))


# ─────────────────────────────────────────────────────────────
#  SCAN ONE STOCK
# ─────────────────────────────────────────────────────────────
def clean_df(df):
    """Flatten yfinance multi-level columns and ensure clean numeric data."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Keep only OHLCV columns
    needed = ["Open","High","Low","Close","Volume"]
    df = df[[c for c in needed if c in df.columns]]
    # Convert all to numeric, drop rows with NaN close
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    return df

def scan_symbol(symbol):
    try:
        raw = yf.download(f"{symbol}.NS", period="2y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty: return []
        df = clean_df(raw)
        if len(df) < 220: return []

        results = []

        # Swing setups
        df_ind = compute_indicators(df.copy())
        r = detect_swing(df_ind, symbol)
        if r: results.append(r)

        # VCP
        v = detect_vcp(df.copy(), symbol)
        if v: results.append(v)

        return results
    except:
        return []


# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="header">SWING SCANNER</div>', unsafe_allow_html=True)
st.markdown('<div class="subhead">Nifty 500 + F&O  •  EOD  •  Trend + Breakout + Pullback + VCP</div>', unsafe_allow_html=True)

if "signals"   not in st.session_state: st.session_state.signals   = []
if "scanned"   not in st.session_state: st.session_state.scanned   = False
if "last_scan" not in st.session_state: st.session_state.last_scan = ""
if "alerted"   not in st.session_state: st.session_state.alerted   = set()

col1, col2 = st.columns([2,1])
with col1: fno_only = st.toggle("F&O setups only", value=False)
with col2: scan_btn = st.button("🔍 Scan Now", use_container_width=True)

min_score = st.slider("Min score", 1, 10, 4)

if scan_btn:
    st.session_state.signals = []
    prog = st.progress(0, text="Starting scan...")
    total = len(ALL_SYMBOLS)

    for i, sym in enumerate(ALL_SYMBOLS):
        prog.progress((i+1)/total, text=f"Scanning {sym}... ({i+1}/{total})")
        hits = scan_symbol(sym)
        for r in hits:
            if r["score"] >= min_score:
                st.session_state.signals.append(r)
                key = f"{sym}_{r['direction']}_{r['setup_type']}"
                if key not in st.session_state.alerted:
                    emoji = "🟣" if r["setup_type"]=="vcp" else ("🟢" if r["direction"]=="BULL" else "🔴")
                    send_telegram(
                        f"{emoji} <b>SWING — {sym}</b>  [{r['setups']}]\n"
                        f"Score: {r['score']}/10  |  {r.get('fno_tag','')}\n"
                        f"💰 Price: ₹{r['price']}\n"
                        f"🛑 SL: ₹{r['sl']}  |  T1: ₹{r['target1']}  |  T2: ₹{r['target2']}\n"
                        f"R:R  T1={r['rr1']}R  T2={r['rr2']}R"
                    )
                    st.session_state.alerted.add(key)

    prog.empty()
    st.session_state.scanned   = True
    st.session_state.last_scan = datetime.now().strftime("%d %b %Y %H:%M")
    st.rerun()

if st.session_state.last_scan:
    st.caption(f"Last scan: {st.session_state.last_scan}")

sigs = st.session_state.signals
if fno_only: sigs = [s for s in sigs if s["fno_tag"]]
sigs = sorted(sigs, key=lambda x: -x["score"])

bulls = [s for s in sigs if s["direction"]=="BULL"]
bears = [s for s in sigs if s["direction"]=="BEAR"]
vcps  = [s for s in sigs if s["setup_type"]=="vcp"]
fnos  = [s for s in sigs if "STRONG" in s.get("fno_tag","")]

st.markdown(f"""
<div class="stat-row">
  <div class="stat-box"><div class="stat-val" style="color:#3b82f6">{len(sigs)}</div><div class="stat-lbl">Total</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#00e676">{len(bulls)}</div><div class="stat-lbl">Bullish</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#ff3d5a">{len(bears)}</div><div class="stat-lbl">Bearish</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#a855f7">{len(vcps)}</div><div class="stat-lbl">VCP</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#ff6d00">{len(fnos)}</div><div class="stat-lbl">Strong F&O</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

if not sigs and st.session_state.scanned:
    st.markdown('<div style="text-align:center;padding:40px;color:#64748b">No setups found.<br>Try lowering the min score slider.</div>', unsafe_allow_html=True)
elif not st.session_state.scanned:
    st.markdown('<div style="text-align:center;padding:40px;color:#64748b">📊 Tap <b>Scan Now</b> after 3:30 PM<br>to get today\'s swing setups.</div>', unsafe_allow_html=True)

for s in sigs:
    is_vcp = s["setup_type"] == "vcp"
    bull   = s["direction"]  == "BULL"
    card_c = "vcp" if is_vcp else ("fno-strong" if "STRONG" in s.get("fno_tag","") else ("bull" if bull else "bear"))
    dir_tag= f'<span class="tag {"tag-vcp" if is_vcp else ("tag-bull" if bull else "tag-bear")}">{"🔮 VCP" if is_vcp else ("▲ BULLISH" if bull else "▼ BEARISH")}</span>'
    fno_html=""
    if s["fno_tag"]:
        tc = "tag-fno" if "STRONG" in s["fno_tag"] else "tag-fno2"
        fno_html = f'<span class="tag {tc}">{s["fno_tag"]}</span>'
    setup_tags = "".join([f'<span class="setup-tag">{x}</span>' for x in s["setups"].split(" + ")])
    fill_w = s["score"]*10
    fill_c = "#a855f7" if is_vcp else ("#00e676" if bull else "#ff3d5a")
    ltp_c  = "#a855f7" if is_vcp else ("#00e676" if bull else "#ff3d5a")

    st.markdown(f"""
    <div class="card {card_c}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <div><span class="sym">{s['symbol']}</span>{dir_tag}{fno_html}</div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:700">{s['score']}/10</div>
      </div>
      <div style="margin-bottom:4px">{setup_tags}</div>
      <div class="score-bar"><div class="score-fill" style="width:{fill_w}%;background:{fill_c}"></div></div>
      <div style="font-size:0.63rem;color:#64748b;margin-bottom:6px">RSI:{s['rsi']}  ADX:{s['adx']}  Vol:{s['vol_ratio']}x  ATR:₹{s['atr']}  |  {s['date']}</div>
      <div class="grid3">
        <div class="cell"><div class="cv" style="color:{ltp_c}">₹{s['price']}</div><div class="cl">Price</div></div>
        <div class="cell"><div class="cv" style="color:#ff3d5a">₹{s['sl']}</div><div class="cl">Stop Loss</div></div>
        <div class="cell"><div class="cv" style="color:#64748b">₹{s['atr']}</div><div class="cl">ATR</div></div>
      </div>
      <div class="grid2">
        <div class="cell"><div class="cv" style="color:#00e676">₹{s['target1']}</div><div class="cl">Target 1 ({s['rr1']}R)</div></div>
        <div class="cell"><div class="cv" style="color:#00e676">₹{s['target2']}</div><div class="cl">Target 2 ({s['rr2']}R)</div></div>
      </div>
      {f'<div style="margin-top:6px;font-size:0.62rem;color:#ff6d00">F&O: {s["fno_reasons"]}</div>' if s["fno_tag"] and s["fno_reasons"]!="—" else ""}
    </div>
    """, unsafe_allow_html=True)
