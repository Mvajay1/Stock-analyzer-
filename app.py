# ======================================
# FULL-FLEDGED STOCK ANALYZER STREAMLIT APP
# ======================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# -------------------------------
# 1️⃣ Sector thresholds
# -------------------------------
SECTOR_THRESHOLDS = {
    'IT': {'ROE': (15, 35), 'ROCE': (15, 35), 'NPM': (10, 25), 'DebtEquity': (0, 0.5), 'PE': (20, 40), 'PB': (2, 6)},
    'Banking': {'ROE': (12, 18), 'ROCE': (10, 15), 'NPM': (5, 15), 'DebtEquity': (6, 10), 'PE': (10, 15), 'PB': (1.5, 2.5)},
    'Manufacturing': {'ROE': (10, 20), 'ROCE': (12, 18), 'NPM': (5, 15), 'DebtEquity': (0.5, 1.5), 'PE': (12, 18), 'PB': (1.5, 2.5)},
    'FMCG': {'ROE': (20, 35), 'ROCE': (25, 35), 'NPM': (15, 25), 'DebtEquity': (0.5, 1.0), 'PE': (35, 45), 'PB': (5, 7)},
    'Pharma': {'ROE': (15, 25), 'ROCE': (18, 28), 'NPM': (10, 20), 'DebtEquity': (0.1, 0.5), 'PE': (20, 30), 'PB': (3, 5)},
    'Automobile': {'ROE': (8, 15), 'ROCE': (10, 20), 'NPM': (3, 10), 'DebtEquity': (0.5, 1.5), 'PE': (10, 20), 'PB': (1.5, 2.5)},
    'Energy': {'ROE': (5, 12), 'ROCE': (8, 15), 'NPM': (2, 8), 'DebtEquity': (1.0, 2.0), 'PE': (8, 15), 'PB': (1.0, 2.0)},
    'Telecom': {'ROE': (5, 10), 'ROCE': (6, 12), 'NPM': (2, 8), 'DebtEquity': (0.5, 1.0), 'PE': (8, 15), 'PB': (1.0, 2.0)},
    'Utilities': {'ROE': (6, 12), 'ROCE': (8, 15), 'NPM': (3, 10), 'DebtEquity': (1.0, 2.0), 'PE': (10, 18), 'PB': (1.5, 2.5)}
}

# -------------------------------
# 2️⃣ Market cap thresholds
# -------------------------------
MARKET_CAP_LIMITS = {
    'Penny (0–500 Cr)': (0, 500),
    'Small (500–5,000 Cr)': (500, 5000),
    'Mid (5,000–20,000 Cr)': (5000, 20000),
    'Large (20,000+ Cr)': (20000, 1e12)
}

# -------------------------------
# 3️⃣ Normalize metrics
# -------------------------------
def normalize_metric(value, sector, metric):
    thresholds = SECTOR_THRESHOLDS.get(sector, {}).get(metric, (0, 100))
    min_val, max_val = thresholds
    norm = (value - min_val)/(max_val-min_val)
    norm = max(0,min(1,norm))
    if metric=='DebtEquity':
        norm = 1 - norm
    return norm

# -------------------------------
# 4️⃣ Composite metrics
# -------------------------------
def compute_composites(df):
    df['QualityIndex'] = df[['ROE_norm','ROCE_norm','NPM_norm']].mean(axis=1)
    df['FinancialStrength'] = df[['InterestCoverage_norm','FCF_MCAP_norm']].mean(axis=1)
    df['GrowthConsistency'] = 1 - df[['RevenueCAGR_norm','EPSCAGR_norm']].std(axis=1)
    df['ValuationAttractiveness'] = df[['PE_norm','PB_norm']].mean(axis=1)
    return df

# -------------------------------
# 5️⃣ Final score
# -------------------------------
WEIGHTS = {
    'QualityIndex': 0.3,
    'GrowthConsistency':0.2,
    'FinancialStrength':0.2,
    'ValuationAttractiveness':0.2,
    'DividendYield_norm':0.1
}

def compute_final_score(df):
    score = np.zeros(len(df))
    for metric,w in WEIGHTS.items():
        score += df[metric]*w
    df['FinalScore'] = score*100
    return df

# -------------------------------
# 6️⃣ Fetch fundamentals
# -------------------------------
def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'Ticker': ticker,
        'Sector': info.get('sector','IT'),
        'MarketCap': info.get('marketCap',0)/1e7,
        'ROE': info.get('returnOnEquity',0),
        'ROCE': info.get('returnOnAssets',0),
        'NPM': info.get('profitMargins',0),
        'RevenueCAGR': info.get('revenueGrowth',0),
        'EPSCAGR': info.get('earningsQuarterlyGrowth',0),
        'DebtEquity': info.get('debtToEquity',0),
        'InterestCoverage': info.get('interestCoverage',0),
        'FCF_MCAP': info.get('freeCashflow',0)/max(info.get('marketCap',1),1),
        'PE': info.get('trailingPE',0),
        'PB': info.get('priceToBook',0),
        'DividendYield': info.get('dividendYield',0)
    }

# -------------------------------
# 7️⃣ Score stocks / Top 10
# -------------------------------
def score_stocks(tickers, sector, market_cap):
    min_cap,max_cap = MARKET_CAP_LIMITS[market_cap]
    data = [fetch_fundamentals(t) for t in tickers]
    df = pd.DataFrame(data)
    df = df[(df['Sector']==sector) & (df['MarketCap']>=min_cap) & (df['MarketCap']<=max_cap)]
    if df.empty:
        return pd.DataFrame()
    for col in ['ROE','ROCE','NPM','RevenueCAGR','EPSCAGR','DebtEquity','InterestCoverage','FCF_MCAP','PE','PB','DividendYield']:
        df[col+'_norm'] = df.apply(lambda x: normalize_metric(x[col], x['Sector'], col), axis=1)
    df = compute_composites(df)
    df = compute_final_score(df)
    df_sorted = df.sort_values(by='FinalScore',ascending=False)
    return df_sorted

# -------------------------------
# 8️⃣ Price prediction (linear regression)
# -------------------------------
def predict_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y", interval="1mo")
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist['MonthIndex'] = np.arange(len(hist))
    X = hist[['MonthIndex']]
    y = hist['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_quarter = len(hist)+3
    future_2year = len(hist)+24
    pred_quarter = model.predict([[future_quarter]])[0]
    pred_2year = model.predict([[future_2year]])[0]
    return round(pred_quarter,2), round(pred_2year,2)

# -------------------------------
# 9️⃣ Sentiment analysis
# -------------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        if not news_list:
            return "Neutral"
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news_list]
        avg_score = np.mean(scores)
        if avg_score>0.05:
            return "Bullish"
        elif avg_score<-0.05:
            return "Bearish"
        else:
            return "Neutral"
    except:
        return "Neutral"

# -------------------------------
# 10️⃣ Streamlit App Layout
# -------------------------------
st.title("📈 NSE Stock Analyzer & Predictor")

# Example tickers (can be replaced with full NSE tickers)
tickers = ['TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS','ASIANPAINT.NS','RELIANCE.NS','SBIN.NS','LT.NS','HINDUNILVR.NS','KOTAKBANK.NS']

# Sidebar input
st.sidebar.header("Filter Options")
selected_sector = st.sidebar.selectbox("Select Sector", list(SECTOR_THRESHOLDS.keys()))
selected_cap = st.sidebar.selectbox("Select Market Cap", list(MARKET_CAP_LIMITS.keys()))
stock_input = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS)","TCS.NS")
st.sidebar.markdown("Market Cap Ranges:\n- Penny: 0–500 Cr\n- Small: 500–5,000 Cr\n- Mid: 5,000–20,000 Cr\n- Large: 20,000+ Cr")

# Run Analysis
if st.sidebar.button("Run Analysis"):
    st.subheader("🔥 Top 10 Watchlist based on Sector & Market Cap")
    top10 = score_stocks(tickers, selected_sector, selected_cap)
    if top10.empty:
        st.write("No stocks found for selected filters.")
    else:
        st.dataframe(top10.head(10))

    st.subheader(f"⭐ Recommendation Score for {stock_input}")
    score_df = score_stocks([stock_input], selected_sector, selected_cap)
    if score_df.empty:
        st.write("Stock not found or does not match sector/cap filters.")
    else:
        score_val = round(score_df['FinalScore'].values[0],2)
        st.write(score_val)

    st.subheader(f"📈 Predicted Prices for {stock_input}")
    preds = predict_price(stock_input)
    if preds:
        st.write(f"Price after ~1 quarter: ₹{preds[0]}")
        st.write(f"Price after ~2 years: ₹{preds[1]}")
    else:
        st.write("Not enough historical data to predict.")

    st.subheader(f"📰 News Sentiment for {stock_input}")
    sentiment = get_sentiment(stock_input)
    st.write(sentiment)

    # Simple signal based on score and sentiment
    signal = "Hold"
    if score_df.empty:
        signal="N/A"
    else:
        if score_val>70 and sentiment=="Bullish":
            signal="Strong Buy"
        elif score_val>60:
            signal="Buy"
        elif score_val<40 and sentiment=="Bearish":
            signal="Sell"
    st.subheader("💡 Suggested Action")
    st.write(signal)
