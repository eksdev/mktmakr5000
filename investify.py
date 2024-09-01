#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
import tensorflow as tf
from datetime import timedelta
from prettytable import PrettyTable

# Function to fetch stock data using yfinance
def get_data(ticker, period='3y'):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data['Close']
    except Exception as e:
        st.error(f"Failed to retrieve data for ticker {ticker}: {e}")
        return None

# Web scraping functions
def get_html_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

def scrapestats(stock):
    pd.options.display.float_format = '{:.2f}'.format
    url = f'https://finviz.com/quote.ashx?t={stock}&p=d'
    html = get_html_content(url)
    if html is None:
        print(f"Failed to retrieve HTML for {stock}")
        return None

    soup = BeautifulSoup(html, 'html.parser')
    metrics_table = soup.find('table', class_='js-snapshot-table snapshot-table2 screener_snapshot-table-body')

    if not metrics_table:
        print("Metrics table not found")
        return None

    metrics = []
    for row in metrics_table.find_all('tr'):
        cols = row.find_all('td')
        for i in range(0, len(cols), 2):
            metric_name = cols[i].text.strip()
            metric_value = cols[i + 1].text.strip()

            if metric_name == 'Market Cap':
                if 'B' in metric_value:
                    value = round(float(metric_value.replace('B', '')) * 1_000_000_000, 2)
                elif 'M' in metric_value:
                    value = round(float(metric_value.replace('M', '')) * 1_000_000, 2)
                else:
                    value = round(float(metric_value), 2)
                metrics.append({'Metric': metric_name, 'Value': value})

            elif metric_name in ['P/B', 'P/S', 'Debt/Eq', 'Target Price']:
                try:
                    value = round(float(metric_value), 2)
                except ValueError:
                    value = None
                metrics.append({'Metric': metric_name, 'Value': value})

    return pd.DataFrame(metrics)

def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period="1d")
    return todays_data['Close'].iloc[-1]

def get_related_symbols(stock):
    url = f'https://finance.yahoo.com/quote/{stock}/analysis/'
    html = get_html_content(url)
    if html is None:
        print(f"Failed to retrieve HTML for {stock}")
        return None

    soup = BeautifulSoup(html, 'html.parser')
    sections = soup.find_all('section', {'data-testid': 'card-container'})
    related_tickers = []

    for section in sections:
        ticker_container = section.find('div', class_='ticker-container')
        if ticker_container:
            symbol = ticker_container.find('span', class_='symbol').text.strip()
            related_tickers.append(symbol)
            related_tickers = [ticker for ticker in related_tickers if '.' not in ticker]

    if related_tickers:
        return related_tickers
    else:
        print("Related Tickers Not found")
        return None

def get_metrics(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    html = get_html_content(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')
    metrics_table = soup.find('table', class_='js-snapshot-table snapshot-table2 screener_snapshot-table-body')
    metrics = []
    if metrics_table:
        for row in metrics_table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) % 2 == 0:
                for i in range(0, len(cols), 2):
                    metric_name = cols[i].text.strip()
                    metric_value = cols[i + 1].text.strip()
                    metrics.append({'Metric': metric_name, 'Value': metric_value})

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[metrics_df['Metric'].isin(['Market Cap', 'Forward P/E', 'P/E', 'Insider Own', 'Short Interest', 'Income', 'Sales', 'ROE', 'ROA', "Beta", "Employees", "Sales Y/Y TTM"])]
    metrics_df = metrics_df.reset_index(drop=True)

    def clean_metric(value):
        if value == '-' or value == '':
            return None
        return float(value.strip('%').replace('B', '').replace('M', '').replace(',', ''))

    metrics = {
        'P/E': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'P/E', 'Value'].values[0]),
        'Forward P/E': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'Forward P/E', 'Value'].values[0]),
        'ROE': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'ROE', 'Value'].values[0]),
        'ROA': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'ROA', 'Value'].values[0]),
        'Beta': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'Beta', 'Value'].values[0]),
        'Employees': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'Employees', 'Value'].values[0]),
        'Sales Y/Y TTM': clean_metric(metrics_df.loc[metrics_df['Metric'] == 'Sales Y/Y TTM', 'Value'].values[0])
    }

    return list(metrics.values())

def compare_metrics(stock):
    similar_symbols = get_related_symbols(stock)
    similar_symbols = [stock] + similar_symbols
    compare_sheet = []
    for s in similar_symbols:
        x = get_metrics(s)
        compare_sheet.append([s] + x)

    cs = pd.DataFrame(compare_sheet)
    cs.columns = ['Stock', 'P/E', 'Forward P/E', 'ROE', 'ROA', 'Beta', 'Employees', 'Sales Y/Y TTM']
    cs.set_index('Stock', inplace=True)
    cs = cs.astype(float, errors='ignore')

    return cs

def get_analyst_ratings(ticker):
    news_sources = ['(Motley Fool)', '(Reuters)', '(InvestorPlace)', '(The Wall Street Journal)']
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    html = get_html_content(url)
    if html is None:
        print(f"Failed to retrieve HTML for {ticker}")
        return [], [], "", []

    soup = BeautifulSoup(html, 'html.parser')
    ratings = soup.find_all('tr', class_='styled-row is-hoverable is-bordered is-rounded is-border-top is-hover-borders has-label has-color-text')
    insider_sales = soup.find_all('tr', class_='fv-insider-row')
    news = soup.find_all('tr', class_='cursor-pointer has-label')

    list_ratings = []
    list_insider_trades = []
    list_news = []
    description = ""

    descript = soup.find('td', class_='fullview-profile')
    if descript:
        description = descript.text.strip()

    for n in news:
        news_store = {}
        try:
            publisher = n.find('div', class_='news-link-right flex gap-1 items-center').find_next('span')
            if publisher:
                news_store["publisher"] = publisher.text.strip()
        except Exception as e:
            print(f"Error retrieving publisher: {e}")

        try:
            article = n.find('a', class_='tab-link-news')
            if article:
                news_store["article"] = article.text.strip()
        except Exception as e:
            print(f"Error retrieving article: {e}")

        try:
            link = n.find('a', class_='tab-link-news')
            if link:
                news_store["link"] = link.get('href')
        except Exception as e:
            print(f"Error retrieving link: {e}")

        if news_store.get("publisher") in news_sources:
            list_news.append(news_store)

    for r in ratings:
        rating_store = {}
        try:
            date = r.find('td')
            if date:
                rating_store["date"] = date.text.strip()
        except Exception as e:
            print(f"Error retrieving date: {e}")

        try:
            analyst = r.find('td', class_='text-left')
            if analyst:
                rating_store["analyst"] = analyst.text.strip()
        except Exception as e:
            print(f"Error retrieving analyst: {e}")

        try:
            rating_type = r.find('td', class_='text-left').find_next()
            if rating_type:
                rating_store["rating_type"] = rating_type.text.strip()
        except Exception as e:
            print(f"Error retrieving rating type: {e}")

        list_ratings.append(rating_store)

    for i in insider_sales:
        insider_store = {}
        try:
            insider_type = i.find_all('td')[1]
            if insider_type:
                insider_store["insider_type"] = insider_type.text.strip()
        except Exception as e:
            print(f"Error retrieving insider type: {e}")

        try:
            date = i.find_all('td')[2]
            if date:
                insider_store["date"] = date.text.strip()
        except Exception as e:
            print(f"Error retrieving date of trade: {e}")

        try:
            trade_type = i.find_all('td')[3]
            if trade_type:
                insider_store["trade_type"] = trade_type.text.strip()
        except Exception as e:
            print(f"Error retrieving trade type: {e}")

        try:
            avg_price = i.find_all('td')[4]
            if avg_price:
                insider_store["avg_price"] = avg_price.text.strip()
        except Exception as e:
            print(f"Error retrieving average price: {e}")

        try:
            quantity = i.find_all('td')[5]
            if quantity:
                insider_store["quantity"] = quantity.text.strip()
        except Exception as e:
            print(f"Error retrieving quantity: {e}")

        try:
            insider_store["book_value"] = str(round(float(insider_store["quantity"].replace(',', '')) * float(insider_store["avg_price"].replace(',', ''))))
        except Exception as e:
            insider_store["book_value"] = 'N/A'

        if insider_store.get("insider_type") == "President and CEO":
            list_insider_trades.append(insider_store)

    return list_ratings, list_insider_trades, description, list_news

# Monte Carlo method for stock forecasting
class StockForecaster:
    def __init__(self, symbol, period='3y'):
        self.symbol = symbol
        self.data = get_data(symbol, period)

    def predict_gbm(self, df, days_in_future, iterations):
        prices = tf.convert_to_tensor(df.values, dtype=tf.float32)
        log_returns = tf.math.log(prices[1:] / prices[:-1])
        drift = tf.reduce_mean(log_returns) - 0.5 * tf.math.reduce_variance(log_returns)
        stdev = tf.math.reduce_std(log_returns)
        random_values = tf.random.normal([days_in_future, iterations], dtype=tf.float32)
        daily_returns = tf.exp(drift + stdev * random_values)
        initial_prices = tf.fill([iterations], prices[-1])
        price_paths = tf.TensorArray(dtype=tf.float32, size=days_in_future + 1, clear_after_read=False)
        price_paths = price_paths.write(0, initial_prices)

        for t in range(1, days_in_future + 1):
            last_prices = price_paths.read(t - 1)
            current_prices = last_prices * daily_returns[t - 1]
            price_paths = price_paths.write(t, current_prices)

        price_paths = price_paths.stack()
        return price_paths.numpy()

    def monte_carlo_simulation(self, df, days, iterations):
        price_paths = self.predict_gbm(df, days, iterations)
        last_actual_price = df.iloc[-1]
        returns = price_paths[-1] / last_actual_price - 1
        mean_returns = returns.mean()
        sd_returns = returns.std()
        thresholds = [-40, -20, -10, -5, 0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820]
        table = PrettyTable()
        table.field_names = [f"Change over next {days} days (%)", "Probability (%)"]

        prob_distribution = []
        for threshold in thresholds:
            prob = norm.cdf(threshold / 100, mean_returns, sd_returns) * 100
            if prob < 99:
                table.add_row([threshold, f"{prob:.2f}%"])
                prob_distribution.append((threshold, prob))

        return table, prob_distribution

    def forecast_stock(self, prediction_years=2, iterations=35):
        days = 365 * prediction_years
        predicted_prices = self.predict_gbm(self.data, days, iterations)
        predicted_avg_prices = np.mean(predicted_prices, axis=1)
        future_dates = [self.data.index[-1] + timedelta(days=x) for x in range(1, len(predicted_avg_prices) + 1)]
        full_dates = self.data.index.append(pd.Index(future_dates))
        full_prices = np.concatenate([self.data.values, predicted_avg_prices])

        forecast_data = pd.DataFrame({
            'Historical Close': self.data,
            'Forecast': pd.Series(predicted_avg_prices, index=future_dates)
        }, index=full_dates)

        return forecast_data

# Portfolio optimization functions
def minimize_cv(tickers, period, num_portfolios=30000):
    data = pd.DataFrame({ticker: get_data(ticker, period) for ticker in tickers})
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        cv = abs(portfolio_std_dev / portfolio_return)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = cv

    min_cv_idx = np.argmin(results[2])
    min_cv_allocation = weights_record[min_cv_idx]
    min_cv_return = results[0, min_cv_idx]
    min_cv_std_dev = results[1, min_cv_idx]

    allocation = pd.DataFrame({
        'Ticker': tickers,
        'Allocation (%)': min_cv_allocation * 100
    })

    portfolio_daily_returns = (returns * min_cv_allocation).sum(axis=1)
    equity_curve = (1 + portfolio_daily_returns).cumprod()
    portfolio_skewness = skew(portfolio_daily_returns)
    portfolio_kurtosis = kurtosis(portfolio_daily_returns)
    
    return allocation, min_cv_return, min_cv_std_dev, portfolio_skewness, portfolio_kurtosis, equity_curve

class Dashboard:
    def __init__(self, symbols, period):
        self.symbols = symbols
        self.p = period
        self.i = '1d'
        
    def minimizecv(self):
        allocation, min_cv_return, min_cv_std_dev, portfolio_skewness, portfolio_kurtosis, equity_curve = minimize_cv(self.symbols, self.p)
        coef = round(min_cv_std_dev / min_cv_return, 2)
        portfolio_skewness = skew(equity_curve)
        portfolio_kurtosis = kurtosis(equity_curve)
        return allocation, min_cv_return, min_cv_std_dev, portfolio_skewness, portfolio_kurtosis, equity_curve

    def benchmarkcv(self):
        allocation, min_cv_return, min_cv_std_dev, portfolio_skewness, portfolio_kurtosis, equity_curve = minimize_cv(['SPY'], self.p)
        coef = round(min_cv_std_dev / min_cv_return, 2)
        return coef
    
    def analystratings(self, symbol):
        analystratings, ceotrades, description, newslist = get_analyst_ratings(symbol)
        return analystratings, ceotrades, description, newslist
    
    def comparemetrics(self, symbol):
        cs = compare_metrics(symbol)
        return cs

# Set up the Streamlit app
st.title("Zamson Portfolio Optimizer")
page = st.selectbox("Choose a page:", ["Optimization", "Research", "Projection"])

if page == "Optimization":
    portfolio_input = st.text_input("Please type in a portfolio of equities, separated by commas", "").upper()
    portfolio = list(set(portfolio_input.split(",")))
    period = st.slider("Select the period for analysis (in years)", min_value=1, max_value=10, value=1, step=1)
    period_str = f"{period}y"

    if portfolio and portfolio[0]:
        try:
            dashboard = Dashboard(portfolio, period_str)
            allocation, min_cv_return, min_cv_std_dev, portfolio_skewness, portfolio_kurtosis, equity_curve = dashboard.minimizecv()
            benchmark_cv = dashboard.benchmarkcv()

            combined_data = pd.DataFrame(equity_curve, columns=['Portfolio'])
            for ticker in portfolio:
                stock_data = get_data(ticker, period_str)
                if stock_data is not None:
                    stock_equity_curve = (1 + stock_data.pct_change().dropna()).cumprod()
                    normalized_stock_equity_curve = stock_equity_curve / stock_equity_curve.iloc[0]
                    combined_data[ticker] = normalized_stock_equity_curve

            st.write("### Portfolio and Individual Stock Equity Curves")
            st.line_chart(combined_data)

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Optimal Portfolio Allocation")
                st.dataframe(allocation, height=400)

            with col2:
                st.write("### Portfolio Statistics")
                st.write(f"**Annual Return:** {min_cv_return*100:.2f}%")
                st.write(f"**Annual Standard Deviation:** {min_cv_std_dev*100:.2f}%")
                st.write(f"**Coefficient of Variation:** {min_cv_std_dev/min_cv_return:.4f}")
                st.write(f"**Portfolio Skewness:** {portfolio_skewness:.4f}")
                st.write(f"**Portfolio Kurtosis:** {portfolio_kurtosis:.4f}")
                st.write(f"**Benchmark CV (SPY):** {benchmark_cv:.4f}")
        
        except ValueError as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

elif page == "Research":
    st.write("### Research Page")
    st.write("Here you can explore the equities market deeply with access to Financials, Targets, Ratios, and Projections.")
    
    equity = st.text_input("Enter a symbol for Analysis:", "").upper()
    if equity:
        try:
            dashboard = Dashboard([equity], '1y')
            ratings_df, ceo_trades_df, description, news_list = dashboard.analystratings(equity)
            metrics_df = dashboard.comparemetrics(equity)
            stats_df = scrapestats(equity)
            cp = get_current_price(equity)
            stats_df['Current Price'] = cp 

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Company Description")
                st.write(description)

            with col2:
                st.write("### CEO Trades")
                st.dataframe(pd.DataFrame(ceo_trades_df), height=300)

            with col1:
                st.write("### Key Financial Metrics")
                st.dataframe(metrics_df, height=300)

            with col2:
                st.write("### Financial Ratios and Projections")
                st.dataframe(stats_df, height=300)

            st.write("### Analyst Ratings")
            st.dataframe(ratings_df, height=300)

            if news_list:
                st.write("### Recent News")
                for news in news_list:
                    st.write(f"- {news}")

        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")

elif page == "Projection":
    st.write("### Projection Page")
    st.write("Enter a stock symbol to forecast its future prices using Monte Carlo Simulation and GBM.")

    symbol = st.text_input("Enter a stock symbol:", "").upper()
    if symbol:
        try:
            forecaster = StockForecaster(symbol)
            forecast_data = forecaster.forecast_stock()
            st.write(f"### {symbol} Historical and Forecasted Prices")
            st.line_chart(forecast_data)

            st.write("### Monte Carlo Simulation")
            days = st.slider("Select the number of days for the Monte Carlo simulation", 30, 365, 365)
            iterations = st.slider("Select the number of iterations for the Monte Carlo simulation", 100, 10000, 1000)
            table, prob_distribution = forecaster.monte_carlo_simulation(forecaster.data, days, iterations)
            st.write(table)

        except ValueError as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
