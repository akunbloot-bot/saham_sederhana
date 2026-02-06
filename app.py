import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np

# Config constants (from config.py)
REFRESH_INTERVAL_OPTIONS = [0, 30, 60, 300]  # Seconds; 0 disables
SCORING_THRESHOLDS = {
    'ideal_pe': 15,
    'ideal_pbv': 2,
    'min_roe': 0.05,
    'max_debt_equity': 1.5,
    'min_growth': 5,
    'min_margin': 5
}

# Data source functions (from data_source.py)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1y", interval="1d")
        
        # Fetch financial statements
        income = ticker.get_income_stmt()
        balance = ticker.get_balance_sheet()

        data = {}
        data['price'] = info.get('currentPrice')
        data['market_cap'] = info.get('marketCap')
        data['volume'] = info.get('volume')
        data['pe'] = info.get('trailingPE')
        data['pbv'] = info.get('priceToBook')
        data['eps'] = info.get('trailingEps')
        data['roe'] = info.get('returnOnEquity')
        data['debt_equity'] = info.get('debtToEquity')

        # Revenue growth (last year vs previous)
        if not income.empty and 'Total Revenue' in income.index and len(income.columns) >= 2:
            revenues = income.loc['Total Revenue']
            data['revenue_growth'] = ((revenues[income.columns[0]] - revenues[income.columns[1]]) / revenues[income.columns[1]]) * 100 if revenues[income.columns[1]] != 0 else None

        # Net profit margin
        if not income.empty and 'Net Income' in income.index and 'Total Revenue' in income.index:
            net_income = income.loc['Net Income', income.columns[0]]
            revenue = income.loc['Total Revenue', income.columns[0]]
            data['net_margin'] = (net_income / revenue) * 100 if revenue != 0 else None

        return data, history
    except Exception as e:
        st.warning(f"Gagal mengambil data: {e}")
        return {}, pd.DataFrame()

# Indicators functions (from indicators.py)
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None

def calculate_ma(df, window):
    return df['Close'].rolling(window=window).mean().iloc[-1] if len(df) >= window else None

def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def volume_trend(df):
    if len(df) < 50:
        return 'N/A'
    avg_vol_20 = df['Volume'].rolling(20).mean().iloc[-1]
    avg_vol_50 = df['Volume'].rolling(50).mean().iloc[-1]
    if avg_vol_20 > avg_vol_50 * 1.1:
        return 'Increasing'
    elif avg_vol_20 < avg_vol_50 * 0.9:
        return 'Decreasing'
    return 'Stable'

def trend_direction(df):
    ma20 = calculate_ma(df, 20)
    ma50 = calculate_ma(df, 50)
    ma200 = calculate_ma(df, 200)
    close = df['Close'].iloc[-1]
    if ma20 is None or ma50 is None or ma200 is None:
        return 'N/A'
    if close > ma20 and ma20 > ma50 and ma50 > ma200:
        return 'Up'
    elif close < ma20 and ma20 < ma50 and ma50 < ma200:
        return 'Down'
    return 'Sideways'

def get_technical_indicators(history):
    indicators = {}
    indicators['rsi'] = calculate_rsi(history)
    indicators['ma20'] = calculate_ma(history, 20)
    indicators['ma50'] = calculate_ma(history, 50)
    indicators['ma200'] = calculate_ma(history, 200)
    indicators['macd'], indicators['signal'] = calculate_macd(history)
    indicators['volume_trend'] = volume_trend(history)
    indicators['trend'] = trend_direction(history)
    return indicators

# AI analysis functions (from ai_analysis.py)
def get_ai_decision(final_score, fundamental_score, technical_score, risk_score):
    if final_score > 80:
        recommendation = "STRONG BUY"
    elif final_score > 60:
        recommendation = "BUY (Speculative)"
    elif final_score > 40:
        recommendation = "HOLD"
    else:
        recommendation = "AVOID"
    
    confidence = min(100, final_score + (100 - risk_score) / 2)
    risk_level = "Low" if risk_score > 70 else "Medium" if risk_score > 40 else "High"
    horizon = "Long-term" if fundamental_score > technical_score else "Short-term" if technical_score > fundamental_score else "Medium-term"
    
    return recommendation, confidence, risk_level, horizon

def get_ai_explanation(fundamental_score, technical_score, risk_score, data, indicators):
    explanation = []
    # Fundamental
    fund_help = []
    fund_hurt = []
    if 'pe' in data and data['pe']: 
        if data['pe'] < 15: fund_help.append(f"PE Ratio rendah ({data['pe']:.2f})")
        else: fund_hurt.append(f"PE Ratio tinggi ({data['pe']:.2f})")
    if 'roe' in data and data['roe']: 
        if data['roe'] > 0.15: fund_help.append(f"ROE kuat ({data['roe']:.2f})")
        else: fund_hurt.append(f"ROE lemah ({data['roe']:.2f})")
    explanation.append(f"Skor Fundamental ({fundamental_score:.1f}): Dibantu oleh {', '.join(fund_help) if fund_help else 'tidak ada'}. Dirugikan oleh {', '.join(fund_hurt) if fund_hurt else 'tidak ada'}.")
    
    # Technical
    tech_help = []
    tech_hurt = []
    if indicators['rsi'] is not None:
        if indicators['rsi'] < 30: tech_help.append("RSI oversold")
        elif indicators['rsi'] > 70: tech_hurt.append("RSI overbought")
    if indicators['trend'] == 'Up': tech_help.append("Tren naik yang kuat")
    explanation.append(f"Skor Teknikal ({technical_score:.1f}): Dibantu oleh {', '.join(tech_help) if tech_help else 'tidak ada'}. Dirugikan oleh {', '.join(tech_hurt) if tech_hurt else 'tidak ada'}.")
    
    # Risk
    risk_dom = "Volatilitas tinggi" if 'market_cap' in data and data['market_cap'] < 1e9 else "Likuiditas rendah"
    explanation.append(f"Skor Risiko ({risk_score:.1f}): Risiko dominan adalah {risk_dom}. Skor keseluruhan rendah karena ketidaklengkapan data atau volatilitas.")
    
    return "\n".join(explanation)

def get_ai_conclusion(data, indicators, recommendation, horizon):
    strengths = ["Valuasi menarik", "Pertumbuhan stabil"] if 'revenue_growth' in data and data['revenue_growth'] > 5 else ["Profitabilitas solid"]
    weaknesses = ["Utang tinggi"] if 'debt_equity' in data and data['debt_equity'] > 1.5 else ["Margin rendah"]
    risks = ["Volatilitas pasar", "Risiko sektor"]
    profile = "Investor konservatif" if recommendation in ["HOLD", "AVOID"] else "Investor agresif"
    short_outlook = "Potensi rebound jangka pendek jika tren volume meningkat." if indicators['volume_trend'] == 'Increasing' else "Hati-hati dengan tren sideways."
    long_outlook = "Prospek jangka panjang positif dengan ROE yang kuat." if 'roe' in data and data['roe'] > 0.1 else "Perlu monitor pertumbuhan revenue."
    
    conclusion = f"""
    Kekuatan: {', '.join(strengths)}.
    Kelemahan: {', '.join(weaknesses)}.
    Faktor Risiko: {', '.join(risks)}.
    Profil Investor yang Cocok: {profile}.
    Outlook Jangka Pendek: {short_outlook}
    Outlook Jangka Panjang: {long_outlook}
    """
    return conclusion

# Scoring functions (from scoring.py)
def fundamental_score(data):
    scores = []
    if 'pe' in data and data['pe'] is not None:
        scores.append(max(0, min(100, 150 - data['pe'] * 5)))
    if 'pbv' in data and data['pbv'] is not None:
        scores.append(max(0, min(100, 150 - data['pbv'] * 50)))
    if 'eps' in data and data['eps'] is not None:
        scores.append(min(100, max(0, data['eps'] * 10)) if data['eps'] > 0 else 0)
    if 'roe' in data and data['roe'] is not None:
        scores.append(min(100, max(0, (data['roe'] - 0.05) * 500)))
    if 'debt_equity' in data and data['debt_equity'] is not None:
        scores.append(max(0, min(100, 150 - data['debt_equity'] * 50)))
    if 'revenue_growth' in data and data['revenue_growth'] is not None:
        scores.append(min(100, max(0, data['revenue_growth'] * 2)))
    if 'net_margin' in data and data['net_margin'] is not None:
        scores.append(min(100, max(0, data['net_margin'] * 4)))
    return np.mean(scores) if scores else 0

def technical_score(indicators):
    scores = []
    rsi = indicators['rsi'] if indicators['rsi'] is not None else 50
    trend = indicators['trend']
    rsi_score = 50
    if rsi < 30 and trend != 'Down': rsi_score = 100
    elif rsi > 70 and trend != 'Up': rsi_score = 0
    scores.append(rsi_score)
    ma_score = 100 if trend == 'Up' else 50 if trend == 'Sideways' else 0
    scores.append(ma_score)
    macd, signal = indicators['macd'], indicators['signal']
    macd_score = 100 if macd > signal and macd > 0 else 0 if macd < signal and macd < 0 else 50
    scores.append(macd_score)
    vol_score = 100 if indicators['volume_trend'] == 'Increasing' else 50 if indicators['volume_trend'] == 'Stable' else 0
    scores.append(vol_score)
    return np.mean(scores) if scores else 0

def risk_score(data, history):
    scores = []
    if not history.empty:
        returns = history['Close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        scores.append(max(0, 100 - vol * 200))
    if 'volume' in data and data['volume']:
        scores.append(min(100, data['volume'] / 1e6 * 10))
    if 'market_cap' in data and data['market_cap']:
        scores.append(min(100, data['market_cap'] / 1e9 * 10))
    penalty = -50 if 'market_cap' in data and data['market_cap'] < 1e9 and vol > 0.5 else 0
    return max(0, min(100, np.mean(scores) + penalty if scores else 0))

def calculate_scores(data, history, indicators):
    missing_count = sum(1 for v in data.values() if v is None)
    missing_penalty = missing_count * 10
    f = fundamental_score(data) - missing_penalty / 2
    t = technical_score(indicators)
    r = risk_score(data, history) - missing_penalty / 2
    final = 0.4 * max(0, f) + 0.35 * t + 0.25 * max(0, r)
    return max(0, min(100, f)), t, max(0, min(100, r)), min(100, final), missing_penalty

# UI components functions (from ui_components.py)
def display_fundamental_table(data):
    metrics = {
        "Price": data.get('price', 'N/A'),
        "Market Cap": data.get('market_cap', 'N/A'),
        "Volume": data.get('volume', 'N/A'),
        "PE Ratio": data.get('pe', 'N/A'),
        "PBV": data.get('pbv', 'N/A'),
        "EPS": data.get('eps', 'N/A'),
        "ROE": data.get('roe', 'N/A'),
        "Debt to Equity": data.get('debt_equity', 'N/A'),
        "Revenue Growth": data.get('revenue_growth', 'N/A'),
        "Net Profit Margin": data.get('net_margin', 'N/A')
    }
    df = pd.DataFrame(list(metrics.items()), columns=["Metrik", "Nilai"])
    df['Status'] = df['Nilai'].apply(lambda x: '⚠️' if x == 'N/A' else '')
    st.table(df)

def display_kpi_cards(data, indicators):
    cols = st.columns(3)
    with cols[0]:
        st.metric("Harga Saat Ini", f"{data.get('price', 'N/A')}", delta=None)
    with cols[1]:
        trend = indicators['trend']
        arrow = "↑" if trend == 'Up' else "↓" if trend == 'Down' else "↔"
        color_class = "trend-arrow-up" if trend == 'Up' else "trend-arrow-down" if trend == 'Down' else "trend-arrow-side"
        st.markdown(f"<div class='kpi-card'>Arah Tren: <span class='{color_class}'>{arrow} {trend}</span></div>", unsafe_allow_html=True)
    with cols[2]:
        st.metric("RSI", f"{indicators['rsi']:.2f}" if indicators['rsi'] else 'N/A')

def display_score_gauge(final_score, f, t, r):
    cols = st.columns(4)
    with cols[0]:
        st.progress(final_score / 100)
        st.write(f"Skor Final: {final_score:.1f}")
    with cols[1]:
        st.progress(f / 100)
        st.write(f"Fundamental: {f:.1f}")
    with cols[2]:
        st.progress(t / 100)
        st.write(f"Teknikal: {t:.1f}")
    with cols[3]:
        st.progress(r / 100)
        st.write(f"Risiko: {r:.1f}")

def display_recommendation(decision, confidence, risk_level, horizon):
    color = "green" if "BUY" in decision else "yellow" if decision == "HOLD" else "red"
    st.markdown(f"<h3 class='{color}'>Rekomendasi: {decision}</h3>", unsafe_allow_html=True)
    st.write(f"Tingkat Keyakinan: {confidence:.1f}%")
    st.write(f"Tingkat Risiko: {risk_level}")
    st.write(f"Horizon Waktu: {horizon}")

def display_explanation_panel(explanation):
    with st.expander("Penjelasan AI (Transparan)"):
        st.write(explanation)

def display_conclusion(conclusion):
    st.subheader("Kesimpulan Analis")
    st.write(conclusion)

# Main app logic (from app.py)
st.set_page_config(layout="wide", page_title="Analisis Saham IDX")

css = """
<style>
    .section { background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .kpi-card { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stMetric { font-family: 'Arial', sans-serif; }
    .green { color: green; }
    .yellow { color: orange; }
    .red { color: red; }
    .trend-arrow-up { color: green; font-size: 1.5em; }
    .trend-arrow-down { color: red; font-size: 1.5em; }
    .trend-arrow-side { color: gray; font-size: 1.5em; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

ticker_input = st.text_input("Masukkan kode saham (contoh: BUMI, BBRI, TLKM)", value="", key="ticker").upper()
refresh_interval = st.selectbox("Interval auto-refresh (detik, 0 untuk mati)", REFRESH_INTERVAL_OPTIONS, index=0)

if ticker_input:
    symbol = ticker_input + ".JK"
    with st.spinner("Mengambil data saham..."):
        data, history = get_stock_data(symbol)
        if data and not history.empty:
            indicators = get_technical_indicators(history)
            fundamental_score, technical_score, risk_score, final_score, missing_penalty = calculate_scores(data, history, indicators)
            decision, confidence, risk_level, horizon = get_ai_decision(final_score, fundamental_score, technical_score, risk_score)
            explanation = get_ai_explanation(fundamental_score, technical_score, risk_score, data, indicators)
            conclusion = get_ai_conclusion(data, indicators, decision, horizon)

            st.header(f"Analisis Saham {ticker_input} ({symbol})")

            st.subheader("Data Fundamental")
            col1, col2 = st.columns(2)
            with col1:
                display_fundamental_table(data)
            with col2:
                display_kpi_cards(data, indicators)

            st.subheader("Analisis Teknikal")
            st.write(f"RSI (14): {indicators['rsi']:.2f}" if indicators['rsi'] else "RSI (14): N/A")
            st.write(f"MA20: {indicators['ma20']:.2f}, MA50: {indicators['ma50']:.2f}, MA200: {indicators['ma200']:.2f}" if indicators['ma20'] else "MA: N/A")
            st.write(f"MACD: {indicators['macd']:.2f}, Signal: {indicators['signal']:.2f}")
            st.write(f"Tren Volume: {indicators['volume_trend']}")
            st.write(f"Arah Tren: {indicators['trend']}")

            st.subheader("Skor AI")
            display_score_gauge(final_score, fundamental_score, technical_score, risk_score)

            display_recommendation(decision, confidence, risk_level, horizon)

            display_explanation_panel(explanation)

            display_conclusion(conclusion)
        else:
            st.error("Data tidak tersedia untuk saham ini. Coba ticker lain.")

    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()
