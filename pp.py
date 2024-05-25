import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from textblob import TextBlob
from datetime import datetime
from arch import arch_model

# Function to fetch stock data
@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to calculate VIX index
def calculate_vix(data):
    returns = data['Close'].pct_change().dropna()
    returns_squared = (returns ** 2).rolling(window=20).sum() * (252 / 20)
    vix = 100 * np.sqrt(returns_squared)
    return vix

# Function to calculate technical indicators: SMA, EMA, RSI, Bollinger Bands
def calculate_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI_14'] = RSIIndicator(data['Close'], window=14).rsi()
    # Calculate Bollinger Bands
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['BB_upper'] = indicator_bb.bollinger_hband()
    data['BB_lower'] = indicator_bb.bollinger_lband()
    return data

# Function to plot stock data chart
def plot_stock_data(data):
    st.subheader("Stock Price Visualization")
    st.line_chart(data['Close'], use_container_width=True)

# Function to plot SMA chart
def plot_sma(data):
    st.subheader("Simple Moving Average (SMA)")
    st.line_chart(data[['Close', 'SMA_20']], use_container_width=True)

# Function to plot EMA chart
def plot_ema(data):
    st.subheader("Exponential Moving Average (EMA)")
    st.line_chart(data[['Close', 'EMA_20']], use_container_width=True)

# Function to plot RSI chart
def plot_rsi(data):
    st.subheader("Relative Strength Index (RSI)")
    st.line_chart(data['RSI_14'], use_container_width=True)

# Function to plot Bollinger Bands chart
def plot_bollinger_bands(data):
    st.subheader("Bollinger Bands")
    st.line_chart(data[['Close', 'BB_upper', 'BB_lower']], use_container_width=True)

# Function to calculate historical volatility (rolling standard deviation of returns)
def calculate_historical_volatility(data, window=20):
    returns = data['Close'].pct_change().dropna()
    historical_volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
    return historical_volatility

# Function to forecast future market volatility using GARCH model
def forecast_volatility(data, forecast_horizon=20):
    returns = data['Close'].pct_change().dropna() * 100  # Convert to percentage returns
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecasts = model_fit.forecast(horizon=forecast_horizon)
    forecast_volatility = np.sqrt(forecasts.variance.values[-1, :])  # Get the forecasted variance and take the square root to get the forecasted volatility
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_horizon+1)[1:]  # Create index for future dates
    forecast_volatility_series = pd.Series(forecast_volatility, index=forecast_index)
    return forecast_volatility_series

# Function to generate insights from volatility forecast
def generate_insights(future_volatility):
    insights = []

    # Example insights based on forecasted volatility
    if future_volatility.iloc[-1] > 20:
        insights.append("High forecasted volatility indicates increased market uncertainty.")
    else:
        insights.append("Forecasted volatility is relatively low, suggesting a more stable market environment.")

    return insights

# Function to display volatility forecast
def display_volatility_forecast(future_volatility):
    st.subheader("Volatility Forecasting")
    st.write("Forecasted Volatility:")
    st.line_chart(future_volatility)

    # Generate insights from volatility forecast
    insights = generate_insights(future_volatility)
    for insight in insights:
        st.write(insight)

# Function to perform correlation analysis
def perform_correlation_analysis(data):
    st.subheader("Correlation Analysis:")
    st.write("You can perform correlation analysis between different technical indicators, such as SMA, EMA, RSI, and Bollinger Bands, to identify potential trends or patterns in the data.")
    correlation_matrix = data[['Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 'BB_upper', 'BB_lower']].corr()
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)

# Function to analyze news sentiment
def analyze_news_sentiment(news_text):
    analysis = TextBlob(news_text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment_category = "Positive"
    elif polarity < 0:
        sentiment_category = "Negative"
    else:
        sentiment_category = "Neutral"
    return polarity, sentiment_category

# Function to display news sentiment analysis section
def display_news_sentiment_analysis():
    st.subheader("News Sentiment Analysis")
    news_text = st.text_area("Enter News Text", "")
    if st.button("Analyze Sentiment"):
        if news_text:
            polarity, sentiment_category = analyze_news_sentiment(news_text)
            st.write(f"Sentiment Polarity: {polarity}")
            st.write(f"Sentiment Category: {sentiment_category}")
        else:
            st.warning("Please enter some news text.")

# Function to determine market volatility level
def analyze_market_volatility(vix):
    # Define thresholds for volatility levels
    low_threshold = 15
    high_threshold = 30
    
    # Determine the current level of market volatility
    if vix < low_threshold:
        volatility_level = "Low"
        insight = "The market is currently experiencing low volatility."
    elif vix >= low_threshold and vix < high_threshold:
        volatility_level = "Moderate"
        insight = "The market is currently experiencing moderate volatility."
    else:
        volatility_level = "High"
        insight = "The market is currently experiencing high volatility. Caution is advised."
    
    return volatility_level, insight

# Main function to run the Streamlit app
def main():
    st.title("Financial Market Volatility Analysis ")
    
    # Date range selection
    start_date = st.date_input("Start Date", datetime(2021, 1, 1))
    end_date = st.date_input("End Date", datetime.now())
    
    # Symbol input
    symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    # Fetch stock data
    data = get_stock_data(symbol, start_date, end_date)
    
    # Show stock data
    st.subheader("Stock Data")
    st.write(data)
    
    # Calculate technical indicators: SMA, EMA, RSI, Bollinger Bands
    data = calculate_technical_indicators(data)
    
    # Plot stock data visualization
    plot_stock_data(data)
    
    # Calculate and show VIX index
    vix = calculate_vix(data)
    st.subheader("Volatility (VIX) Index")
    st.line_chart(vix)
    
    # Analyze market volatility
    volatility_level, insight = analyze_market_volatility(vix.iloc[-1])
    st.subheader("Market Volatility Analysis")
    st.write(f"Current Volatility Level: {volatility_level}")
    st.write(f"Insight: {insight}")
    
    # Add dropdown for selecting indicator
    indicator_type = st.selectbox("Select Indicator:", ("Stock Price", "SMA", "EMA", "RSI", "Bollinger Bands"))
    
    # Plot selected indicator
    if indicator_type == "SMA":
        plot_sma(data)
    elif indicator_type == "EMA":
        plot_ema(data)
    elif indicator_type == "RSI":
        plot_rsi(data)
    elif indicator_type == "Bollinger Bands":
        plot_bollinger_bands(data)
    
    # Additional Insights and Assistance
    st.subheader("Additional Insights and Assistance")
    
    # Perform correlation analysis
    perform_correlation_analysis(data)
    
    # Display news sentiment analysis section
    display_news_sentiment_analysis()

    # Forecast future market volatility
    forecast_window = st.slider("Forecast Window (Days)", min_value=5, max_value=50, value=20, step=5)
    future_volatility = forecast_volatility(data, forecast_horizon=forecast_window)

    # Debugging output
    st.write("Future Volatility Debugging Output:")
    st.write(future_volatility)

    display_volatility_forecast(future_volatility)

if __name__ == "__main__":
    main()
