# ds_crypto_2025
data science project on cryptocurrency

#  Cryptocurrency Price Forecasting and Strategy Backtesting
## Forecasting Price Movements with Predictive Modeling

This project investigates the feasibility of forecasting near-term cryptocurrency price movements using machine learning models trained on historical OHLCV data. The goal is to transform these forecasts into actionable trading strategies that outperform standard market baselines.

## 📋 Project Overview

The core research question is whether we can build ML models that provide a statistical edge in predicting short-term price direction (up or down). The project follows these key steps:
1.  **Problem Framing**: The task is treated as a binary classification problem.
2.  **Model Exploration**: A range of models are tested, from simple linear regressions to complex non-linear ensembles and neural networks.
3.  **Strategy Evaluation**: Model performance is measured by backtesting trading strategies (e.g., Long-Short) and evaluating their cumulative returns and Sharpe ratios against baselines like Buy-and-Hold.

## 📊 Data & Features

### Data Source
* **Assets**: BTC, BNB, ETH, SOL, XRP
* **Time Range**: January 1, 2021 – April 29, 2025
* **Source**: Yahoo Finance, Kaggle
* **Download**: [Google Drive](https://drive.google.com/file/d/1ADpUoKo2IAiTaNEEpOH0nthxmhFyCiw6/view?usp=sharing).
* **Time Intervals**: Data was resampled and evaluated at 1-minute, 10-minute, 4-hour, and 1-day frequencies. The 10-minute interval provided the most robust results.
* **Data Split**:
    * **Training Set**: All data before January 1, 2025.
    * **Test Set**: All data from January 1, 2025, to April 29, 2025.

### Feature Engineering
A rich feature set was engineered to capture different aspects of market dynamics:
* **Price & Return Features**: Relative changes between OHLC values.
* **Momentum & Volatility Features**: Rolling standard deviations, moving averages, and momentum indicators.
* **Trend Features**: Signals based on the convergence/divergence of short-term and long-term moving averages (SMA, EMA).
* **Volume Features**: Metrics to gauge buying and selling pressure.
* **Time/Seasonality Features**: Sinusoidal encoding of the time of day and day of week to model periodic patterns.

## 🤖 Models & Validation

### Models Tested
* ARIMA
* Logistic Regression
* LightGBM
* XGBoost
* LSTM

### Validation
A rigorous, temporally-aware validation process was used to prevent data leakage:
* An **expanding-window cross-validation** was applied to the training set.
* The final models were evaluated on a completely **out-of-sample test set** (all data after Jan 1, 2025).

## 📈 Key Results (10-Minute Interval)
![model_performance_comparison_enhanced](https://github.com/user-attachments/assets/f296b10a-6252-4356-b0bb-31da57da8252)

All machine learning models demonstrated a significant performance uplift over the ARIMA and Buy-and-Hold baselines on the 10-minute interval test set.

| Model                 | Annualized Return | Sharpe Ratio |
| --------------------- | ----------------- | ------------ |
| **Logistic Regression** | **16.91%** | **0.68** |
| XGBoost               | 15.40%            | 0.64         |
| LSTM               | 15.33%             | 0.63         |
| LightGBM              | 12.80%            | 0.41         |
| ARIMA (Baseline)      | -3.81%            | -0.23        |
| Buy & Hold (Baseline) | -8.11%            | -0.27        |

## Conclusion & Next Steps

### Conclusion
1.  **Effective Strategies**: The results confirm that ML models can generate strategies with strong, positive risk-adjusted returns in volatile crypto markets.
2.  **Market Dependence**: Model performance is highest during clear trending phases and tends to degrade in sideways or range-bound markets.

### Future Work
* **Regime-Specific Models**: Retrain and test models specifically for different market regimes (e.g., bull, bear, flat).
* **Alternative Data**: Integrate new data sources, such as on-chain metrics and order book data.
* **Live Deployment**: Explore pathways for real-time model deployment and automated portfolio rebalancing.


Market Microstructure Data: Utilize detailed exchange-level data to understand real-time supply and demand. This includes analyzing order book depth to identify support and resistance levels, and tracking perpetual funding rates and open interest to gauge the sentiment and positioning of leveraged traders.
