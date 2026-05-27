"""
Swing Trading Scanner — Streamlit Dashboard
Check every evening after market close
"""

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, time as dtime

st.set_page_config(page_title="Swing Scanner", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")

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
.card.bull{border-color:#00e676}.card.bear{border-color:#ff3d5a}
.card.fno-strong{border-color:#ff6d00}
.sym{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:1px}
.tag{border-radius:20px;padding:2px 10px;font-size:0.65rem;font-weight:700;margin-left:6px}
.tag-bull{background:rgba(0,230,118,0.12);color:#00e676;border:1px solid rgba(0,230,118,0.3)}
.tag-bear{background:rgba(255,61,90,0.12);color:#ff3d5a;border:1px solid rgba(255,61,90,0.3)}
.tag-fno{background:rgba(255,109,0,0.15);color:#ff6d00;border:1px solid rgba(255,109,0,0.3)}
.tag-fno2{background:rgba(255,179,0,0.12);color:#ffb300;border:1px solid rgba(255,179,0,0.3)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:10px 0}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin:10px 0}
.cell{background:#0a0c10;border-radius:8px;padding:7px 4px;text-align:center}
.cv{font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700}
.cl{font-size:0.57rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-top:1px}
.score-bar{height:6px;border-radius:3px;background:#1e2535;margin:6px 0}
.score-fill{height:6px;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#00e676)}
.setup-tag{font-size:0.62rem;background:#1e2535;color:#94a3b8;border-radius:6px;
           padding:2px 8px;margin:2px;display:inline-block}
.stat-row{display:flex;gap:8px;margin-bottom:12px}
.stat-box{flex:1;background:#111520;border-radius:10px;padding:10px;text-align:center;border:1px solid #1e2535}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700}
.stat-lbl{font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px}
</style>
""", unsafe_allow_html=True)


def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      data={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"HTML"}, timeout=5)
    except: pass


def compute_indicators(df):
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["signal"]   = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["signal"]
    tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    up   = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    pdi  = 100*up.rolling(14).mean()/atr.replace(0,np.nan)
    ndi  = 100*down.rolling(14).mean()/atr.replace(0,np.nan)
    dx   = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    df["adx"]     = dx.rolling(14).mean()
    df["atr"]     = atr
    df["vol_avg20"]= v.rolling(20).mean()
    df["vol_ratio"]= v/df["vol_avg20"].replace(0,np.nan)
    df["high52"]  = h.rolling(252).max()
    df["low52"]   = l.rolling(252).min()
    return df


def detect_setup(df, symbol):
    if len(df) < 220: return None
    row  = df.iloc[-1]; prev = df.iloc[-2]
    c    = float(row["Close"])
    rsi  = float(row["rsi"]); macd_h = float(row["macd_hist"])
    adx  = float(row["adx"]); atr    = float(row["atr"])
    ema20= float(row["ema20"]); ema50 = float(row["ema50"]); ema200=float(row["ema200"])
    vol_r= float(row["vol_ratio"]); high52=float(row["high52"]); low52=float(row["low52"])
    sig  = float(row["signal"]); macd  = float(row["macd"])
    prev_rsi   = float(prev["rsi"]); prev_mh = float(prev["macd_hist"])
    setups=[]; score=0

    # Trend
    if c>ema20>ema50>ema200 and adx>25 and 50<rsi<75 and macd>sig:
        setups.append("TREND_BULL"); score+=3
    if c<ema20<ema50<ema200 and adx>25 and 25<rsi<50 and macd<sig:
        setups.append("TREND_BEAR"); score+=3
    # Breakout
    if c>=high52*0.98 and vol_r>1.5 and rsi>55 and adx>20:
        setups.append("BREAKOUT_BULL"); score+=4
    if c<=low52*1.02 and vol_r>1.5 and rsi<45 and adx>20:
        setups.append("BREAKOUT_BEAR"); score+=4
    # Pullback
    in_up = ema20>ema50>ema200; in_dn = ema20<ema50<ema200
    near20= abs(c-ema20)/ema20<0.02; near50=abs(c-ema50)/ema50<0.02
    if in_up and (near20 or near50) and rsi>prev_rsi and rsi>45 and macd_h>prev_mh:
        setups.append("PULLBACK_BULL"); score+=3
    if in_dn and (near20 or near50) and rsi<prev_rsi and rsi<55 and macd_h<prev_mh:
        setups.append("PULLBACK_BEAR"); score+=3
    if not setups: return None

    ind_score=0
    bull = any("BULL" in s for s in setups)
    if bull:
        if 50<rsi<70: ind_score+=2
        elif rsi>=70: ind_score+=1
    else:
        if 30<rsi<50: ind_score+=2
        elif rsi<=30: ind_score+=1
    if macd_h>0 and bull: ind_score+=2
    if macd_h<0 and not bull: ind_score+=2
    if adx>30: ind_score+=2
    elif adx>20: ind_score+=1
    if vol_r>2: ind_score+=2
    elif vol_r>1.5: ind_score+=1
    if (c>ema20>ema50>ema200) or (c<ema20<ema50<ema200): ind_score+=2
    ind_score=min(ind_score,10)

    is_fno=symbol in FNO_SET; fno_score=0; fno_reasons=[]
    if is_fno:
        if adx>30: fno_score+=3; fno_reasons.append(f"ADX {adx:.0f}")
        if vol_r>2: fno_score+=2; fno_reasons.append(f"Vol {vol_r:.1f}x")
        if c>=high52*0.98 or c<=low52*1.02: fno_score+=2; fno_reasons.append("52W extreme")
        if "BREAKOUT" in " ".join(setups): fno_score+=2; fno_reasons.append("Breakout")
        if ind_score>=7: fno_score+=1; fno_reasons.append("High confluence")

    fno_tag=""
    if is_fno and fno_score>=7: fno_tag="🔥 STRONG F&O"
    elif is_fno and fno_score>=4: fno_tag="⚡ F&O POSSIBLE"
    elif is_fno: fno_tag="📋 F&O"

    direction="BULL" if bull else "BEAR"
    sl  = round(c-1.5*atr,2) if bull else round(c+1.5*atr,2)
    t1  = round(c+2*atr,2)   if bull else round(c-2*atr,2)
    t2  = round(c+3.5*atr,2) if bull else round(c-3.5*atr,2)
    risk= abs(c-sl)

    return dict(symbol=symbol,price=round(c,2),setups=" + ".join(setups),
                direction=direction,score=ind_score,fno_tag=fno_tag,
                fno_reasons=", ".join(fno_reasons),
                rsi=round(rsi,1),adx=round(adx,1),vol_ratio=round(vol_r,2),
                ema20=round(ema20,2),ema50=round(ema50,2),atr=round(atr,2),
                sl=sl,target1=t1,target2=t2,
                rr1=round(abs(t1-c)/risk,1) if risk>0 else 0,
                rr2=round(abs(t2-c)/risk,1) if risk>0 else 0,
                date=df.index[-1].strftime("%Y-%m-%d"))


def fetch_and_scan(symbol):
    try:
        df=yf.download(f"{symbol}.NS",period="2y",interval="1d",progress=False,auto_adjust=True)
        if df.empty or len(df)<220: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=compute_indicators(df)
        return detect_setup(df,symbol)
    except: return None


# ── UI ──
st.markdown('<div class="header">SWING SCANNER</div>', unsafe_allow_html=True)
st.markdown('<div class="subhead">Nifty 500 + F&O  •  EOD Data  •  3 Setup Types  •  Multi-Indicator</div>', unsafe_allow_html=True)

if "signals" not in st.session_state: st.session_state.signals=[]
if "scanned" not in st.session_state: st.session_state.scanned=False
if "last_scan" not in st.session_state: st.session_state.last_scan=""
if "alerted"  not in st.session_state: st.session_state.alerted=set()

col1,col2=st.columns([2,1])
with col1:
    fno_only=st.toggle("F&O setups only",value=False)
with col2:
    scan_btn=st.button("🔍 Scan Now",use_container_width=True)

min_score=st.slider("Min indicator score",1,10,6)

if scan_btn:
    st.session_state.signals=[]
    prog=st.progress(0,text="Starting scan...")
    for i,sym in enumerate(ALL_SYMBOLS):
        prog.progress((i+1)/len(ALL_SYMBOLS),text=f"Scanning {sym}...")
        r=fetch_and_scan(sym)
        if r and r["score"]>=min_score:
            st.session_state.signals.append(r)
            key=f"{sym}_{r['direction']}"
            if key not in st.session_state.alerted:
                fno_note=f"\n{r['fno_tag']} — {r['fno_reasons']}" if r['fno_tag'] else ""
                send_telegram(
                    f"📊 <b>SWING SETUP — {sym}</b>\n"
                    f"{'🟢' if r['direction']=='BULL' else '🔴'} {r['setups']}\n"
                    f"💰 Price: ₹{r['price']}  |  Score: {r['score']}/10\n"
                    f"🛑 SL: ₹{r['sl']}  |  T1: ₹{r['target1']}  |  T2: ₹{r['target2']}\n"
                    f"📈 RSI:{r['rsi']}  ADX:{r['adx']}  Vol:{r['vol_ratio']}x{fno_note}"
                )
                st.session_state.alerted.add(key)
    prog.empty()
    st.session_state.scanned=True
    st.session_state.last_scan=datetime.now().strftime("%d %b %Y %H:%M")
    st.rerun()

if st.session_state.last_scan:
    st.caption(f"Last scan: {st.session_state.last_scan}")

sigs=st.session_state.signals
if fno_only: sigs=[s for s in sigs if s["fno_tag"]]
sigs=sorted(sigs,key=lambda x:(-x["score"],"STRONG" not in x["fno_tag"]))

# Stats
bulls=[s for s in sigs if s["direction"]=="BULL"]
bears=[s for s in sigs if s["direction"]=="BEAR"]
fnos =[s for s in sigs if "STRONG" in s.get("fno_tag","")]
st.markdown(f"""
<div class="stat-row">
  <div class="stat-box"><div class="stat-val" style="color:#3b82f6">{len(sigs)}</div><div class="stat-lbl">Total</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#00e676">{len(bulls)}</div><div class="stat-lbl">Bullish</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#ff3d5a">{len(bears)}</div><div class="stat-lbl">Bearish</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#ff6d00">{len(fnos)}</div><div class="stat-lbl">Strong F&O</div></div>
</div>
""",unsafe_allow_html=True)

st.divider()

if not sigs and st.session_state.scanned:
    st.markdown('<div style="text-align:center;padding:40px;color:#64748b">No setups found matching criteria.<br>Try lowering the min score.</div>',unsafe_allow_html=True)
elif not st.session_state.scanned:
    st.markdown('<div style="text-align:center;padding:40px;color:#64748b">📊 Tap <b>Scan Now</b> after market closes (3:30 PM+)<br>to get today\'s swing setups for tomorrow.</div>',unsafe_allow_html=True)

for s in sigs:
    bull   = s["direction"]=="BULL"
    card_c = "bull" if bull else "bear"
    if "STRONG" in s.get("fno_tag",""): card_c="fno-strong"
    dir_tag= f'<span class="tag tag-{"bull" if bull else "bear"}">{"▲ BULLISH" if bull else "▼ BEARISH"}</span>'
    fno_html=""
    if s["fno_tag"]:
        tc="tag-fno" if "STRONG" in s["fno_tag"] else "tag-fno2"
        fno_html=f'<span class="tag {tc}">{s["fno_tag"]}</span>'

    setup_tags="".join([f'<span class="setup-tag">{x}</span>' for x in s["setups"].split(" + ")])
    fill_w    =s["score"]*10
    ltp_col   ="#00e676" if bull else "#ff3d5a"

    st.markdown(f"""
    <div class="card {card_c}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <div><span class="sym">{s['symbol']}</span>{dir_tag}{fno_html}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#64748b">{s['date']}</div>
      </div>
      <div>{setup_tags}</div>
      <div class="score-bar"><div class="score-fill" style="width:{fill_w}%"></div></div>
      <div style="font-size:0.65rem;color:#64748b;margin-bottom:6px">Indicator Score: {s['score']}/10  |  RSI:{s['rsi']}  ADX:{s['adx']}  Vol:{s['vol_ratio']}x</div>
      <div class="grid3">
        <div class="cell"><div class="cv" style="color:{ltp_col}">₹{s['price']}</div><div class="cl">Price</div></div>
        <div class="cell"><div class="cv" style="color:#ff3d5a">₹{s['sl']}</div><div class="cl">Stop Loss</div></div>
        <div class="cell"><div class="cv" style="color:#64748b">₹{s['atr']}</div><div class="cl">ATR</div></div>
      </div>
      <div class="grid2">
        <div class="cell"><div class="cv" style="color:#00e676">₹{s['target1']}</div><div class="cl">Target 1 ({s['rr1']}R)</div></div>
        <div class="cell"><div class="cv" style="color:#00e676">₹{s['target2']}</div><div class="cl">Target 2 ({s['rr2']}R)</div></div>
      </div>
      {f'<div style="margin-top:6px;font-size:0.62rem;color:#ff6d00">🔥 F&O reasons: {s["fno_reasons"]}</div>' if s["fno_tag"] and s["fno_reasons"]!="—" else ""}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  VCP INTEGRATION (appended)
# ─────────────────────────────────────────────────────────────
from vcp import detect_vcp

def fetch_and_scan_all(symbol):
    """Fetch data once, run both multi-confluence + VCP scan."""
    try:
        df = yf.download(f"{symbol}.NS", period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 220:
            return []
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        results = []

        # Multi-confluence setups
        df_ind = compute_indicators(df.copy())
        r = detect_setup(df_ind, symbol)
        if r:
            results.append(r)

        # VCP setup
        vcp = detect_vcp(df.copy(), symbol)
        if vcp and vcp["vcp_score"] >= 5:
            # Normalise keys to match card rendering
            vcp["score"]     = vcp["vcp_score"]
            vcp["setups"]    = f"{vcp['pattern_type']} ({'→'.join(str(d)+'%' for d in vcp['depths'])})"
            vcp["fno_tag"]   = "🔥 STRONG F&O" if (symbol in FNO_SET and vcp["vcp_score"] >= 7) else \
                               ("⚡ F&O POSSIBLE" if symbol in FNO_SET else "")
            vcp["fno_reasons"] = f"Near pivot {vcp['dist_pivot']}% | RS rank {vcp['rs_rank']} | {'Vol drying ✅' if vcp['vol_drying'] else 'Vol normal'}"
            vcp["vol_ratio"] = vcp["vol_ratio"]
            vcp["rsi"]       = "—"
            vcp["adx"]       = "—"
            results.append(vcp)

        return results
    except:
        return []
