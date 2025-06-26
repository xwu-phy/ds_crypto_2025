# ds_crypto_2025
data science project on cryptocurrency

Cryptocurrency Trading Bot: A Machine Learning Approach
Forecasting Price Movements with Predictive Modeling

Introduction
The goal of this project is to build a trading bot powered by machine learning to predict cryptocurrency price movements. By analyzing historical data, we aim to create an adaptive, data-driven strategy that moves beyond simple rule-based systems.

Methodology
Our approach is rooted in supervised machine learning, where we frame the problem as a binary classification task: will the price go up or down in the next period?

Data Sourcing and Resampling: We use 1-minute OHLCV (Open, High, Low, Close, Volume) data, which is resampled into various timeframes (1-minute, 4-hour, 1-day) to test different trading frequencies.

Missing Data: For any missing data points, we use forward filling (ffill). This method assumes a static market during the missing interval and propagates the last known OHLCV data forward.

Feature Engineering: We generate a rich set of features from the price data, including lagged returns, rolling volatility, moving average ratios, and standard technical indicators like MACD and ATR.

Model & Backtesting: A LightGBM classifier is trained on historical data and its performance is evaluated on a hold-out test set to simulate real-world trading conditions.

Data Overview
Data Type: OHLCV (Open, High, Low, Close, Volume) (Download the dataset from [Google Drive](https://drive.google.com/file/d/1ADpUoKo2IAiTaNEEpOH0nthxmhFyCiw6/view?usp=sharing).)

Time Period: January 1, 2021 - April 1, 2025

Assets: BTC, BNB, ETH, SOL, XRP

Sources: Yahoo Finance & Kaggle

Model Deep Dive: LightGBM,Linear Regression,LSTM, XGBoost,ARIMA.
 
LightGBM (Light Gradient Boosting Machine): We chose LightGBM because it is a fast and efficient gradient boosting framework. It builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. Its performance is ideal for the large and noisy datasets found in financial markets.

Handling Class Imbalance with scale_pos_weight:

Problem: Financial market data is often imbalanced. For example, there are typically fewer periods of significant upward movement compared to periods where the price is flat or moving down.

Solution: The scale_pos_weight parameter adjusts the importance of the minority class (in our case, "price up"). This forces the model to pay more attention to these less frequent but crucial events.

Benefit: This prevents the model from simply defaulting to predicting the majority class, thereby improving its ability to identify valuable trading signals.

Results & Strategy Comparison
The performance of our machine learning model across 1-minute, 4-hour, and 1-day intervals is rigorously compared against baseline strategies, such as "Buy and Hold." We use Profit & Loss (P&L) curves and a suite of other performance metrics to assess the effectiveness of each strategy.

Conclusion and Future Work
Summary: This project successfully demonstrates a complete framework for building, training, and backtesting a machine learning-based trading strategy for the volatile cryptocurrency market.

Future Work:

On-Chain Data: Incorporate data directly from the blockchain to gauge network health and activity. This includes metrics like transaction volume, the number of active addresses, and network fees. A surge in active addresses, for instance, can signal growing adoption and potential upward price pressure.

Market Microstructure Data: Utilize detailed exchange-level data to understand real-time supply and demand. This includes analyzing order book depth to identify support and resistance levels, and tracking perpetual funding rates and open interest to gauge the sentiment and positioning of leveraged traders.
