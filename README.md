# Stock-analyzer-
Interactive Stock Analyzer & Predictor for NSE: Generate top watchlists, get stock recommendation scores, predict future prices, and analyze news sentiment — all in one Streamlit app for smarter investment decisions. NSE Stock Analyzer & Predictor
A full-fledged interactive Streamlit app that helps users analyze, predict, and make informed investment decisions on Indian stocks.
Features
1.	Top 10 Watchlist
2.	Filters by Sector and Market Cap (Penny / Small / Mid / Large).
3.	Displays the strongest stocks based on a fundamental scoring system.
4.	Stock Recommendation Score
5.	Users can input a specific stock ticker to get a comprehensive score based on:
6.	ROE, ROCE, NPM, Debt/Equity, PE, PB, Revenue & EPS growth, Dividend Yield.
7.	Generates a Buy / Hold / Sell signal based on the score and sentiment.
8.	Stock Price Prediction
9.	Predicts stock price after ~1 quarter and ~2 years using historical data and a linear regression ML model.
10.	News Sentiment Analysis
11.	Fetches recent news headlines for the stock and analyzes market sentiment (Bullish / Neutral / Bearish) using VADER.
12.	User-Friendly Interface
13.	Interactive filters in Streamlit sidebar.
14.	Shows market cap ranges for clarity.
15.	Combines fundamental score, sentiment, and predicted prices into a clear action recommendation.
Tech Stack:
•	Python: yfinance, pandas, numpy, scikit-learn, vaderSentiment
•	Streamlit: Web app interface
•	ML Models: Linear Regression for price prediction
•	Deployment: Heroku / Streamlit Cloud
How It Works:
•	Top 10 Watchlist is generated based on sector and market cap filters.
•	Recommendation Score is calculated from normalized financial metrics.
•	Price Predictions estimate future performance using linear regression on historical prices.
•	Sentiment Analysis analyzes recent news to provide short-term market signals.
•	Combined insights give a buy/hold/sell action recommendation.
Future Enhancements
•	Upgrade price prediction with XGBoost or RandomForest for better accuracy.
•	Include technical indicators (moving averages, RSI, MACD).
•	Add alert system for changes in score, price, or sentiment.
•	Expand to full NSE/Indian stock universe.

