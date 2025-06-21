import numpy as np
import pandas as pd
import os
from erdos_src.config import CFG

CONFIG = {
    'initial_capital': 10000.0,
    'commission_rate': 0,       # 0.1% trading fee 0.001
    'slippage_factor': 0,      # 0.05% slippage 0.0005
    'data_frequency': '1H',         # Can be '1H', '4H', '1D', etc.
    'risk_free_rate': 0.0,
    'strategy_params': {
        'long_short': {
            'buy_threshold': 0.5,
            'sell_threshold': 0.5
        },
        'trend_following': {
            'sma_window': 200       # The lookback period for the long-term SMA that defines the trend. A common value is 50, 100, or 200.
        },
        'mean_reversion': {
            'bb_window': 20,        # Lookback window for Bollinger Bands
            'bb_std_dev': 2.0       # Number of standard deviations
        }
    }
}



# --- Strategy Signal Generation ---
# --- 1. Mean Reversion Strategy (with Bollinger Bands/ No need for prediction) ---
def generate_mean_reversion_positions(data):
    """
    Generates positions for a Mean Reversion strategy using Bollinger Bands®.

    This strategy operates independently of a prediction model. It sells when
    the price touches the upper band and buys when it touches the lower band,
    assuming the price will revert to the mean.

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
    Returns:
        pd.Series: A Series of trading positions.
    """
    window = CONFIG['strategy_params']['mean_reversion']['bb_window']
    num_std_dev = CONFIG['strategy_params']['mean_reversion']['bb_std_dev']
    # 1. Calculate the three Bollinger Bands®.
    middle_band = data['close'].rolling(window=window).mean()           # Middle Band: A simple moving average.
    std_dev = data['close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)

    # 2. Initialize positions with zeros.
    positions = pd.Series(index=data.index, data=0, name='position')
    
    # 3. Generate signals based on price crossing the bands.
    # Note: We compare today's price with yesterday's band to avoid lookahead bias.
    # Go Long (+1) if the price crosses BELOW the lower band.
    positions[data['close'] < lower_band] = 1
    # Go Short (-1) if the price crosses ABOVE the upper band.
    positions[data['close'] > upper_band] = -1

    # 4. Generate exit signals: Close the position when the price crosses the middle band.
    # If we are long and price goes above middle, or if we are short and price goes below, exit.
    # To implement this simply, we can use a forward-fill approach.
    # However, a simpler model for backtesting is to hold for a fixed number of periods
    # or until the price crosses the middle band. For simplicity here, we will hold
    # the position until an opposing signal is generated. A more complex exit logic
    # would involve tracking the state (e.g., are we currently in a long trade?).
    # Let's refine the exit logic for clarity:

    # Create a consolidated signal series (1 for buy, -1 for sell, 0 for hold)
    signals = pd.Series(index=data.index, data=0, name='signal')
    signals[data['close'] < lower_band] = 1         # Buy signal
    signals[data['close'] > upper_band] = -1        # Sell signal
    
    # Forward-fill the signals to hold positions until a new signal appears.
    # For example, a +1 signal will be carried forward until a -1 or 0 signal appears.
    positions = signals.replace(to_replace=0, method='ffill').fillna(0)

    return positions

# --- 2. Buy and Hold Strategy (No need for prediction) ---
def generate_buy_and_hold_positions(data):
    """
    Generates positions for a simple Buy and Hold strategy.

    This strategy enters a long position (+1) at the very beginning
    and holds it until the very end. It's used as a benchmark to
    measure the performance of more active strategies.

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.

    Returns:
        pd.Series: A Series of positions (+1 for all periods).
    """
    # Create a new series named 'positions' with the same index as our data.
    # The value is 1 for every single time period.
    positions = pd.Series(index=data.index, data=1, name='position')
    
    return positions

# --- 3. Long-short strategy ---
def generate_long_short_positions(predictions):
    """
    Generates positions directly from the model's prediction signals.

    This strategy acts as a pure test of the model's accuracy.
    A +1 prediction becomes a long position, a 0 prediction becomes a short position.

    Args:
        predictions (pd.Series): A Series of prediction signals (+1, -1, or 0) from forecasting model, or the Series of pred_proba
    Returns:
        pd.Series: A Series of trading positions.
    """
    positions = pd.Series(index=predictions.index, data=0, name='position')
    long_condition = predictions > CONFIG['strategy_params']['long_short']['buy_threshold']
    positions[long_condition] = 1
    short_condition = (1 - predictions) > CONFIG['strategy_params']['long_short']['sell_threshold']
    positions[short_condition] = -1
    return positions

# --- 4. trend_following_overlay ---
def generate_trend_following_positions(data, predictions):
    """
    Generates positions by filtering primary signals with a long-term trend.

    This strategy only allows trades that are aligned with the overall market
    direction, which is determined by a Simple Moving Average (SMA). It's
    designed to prevent taking short-term trades against a strong trend.

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
        predictions (pd.Series): The primary trading signals (+1, -1, 0) from
                                 another function (e.g., generate_positions_with_threshold).
    Returns:
        pd.Series: A Series of filtered trading positions (+1, -1, or 0).
    """
    # --- Step 1: Define the Long-Term Trend ---
    # We calculate the Simple Moving Average (SMA) over a long window. This 
    # represents the average price over a significant period and acts as our
    # indicator of the overall market trend.
    long_window = CONFIG['strategy_params']['trend_following']['sma_window']
    long_term_trend = data['close'].rolling(window=long_window).mean()
    positions = pd.Series(index=data.index, data=0, name='position')

    # --- Step 2: Apply the "Go Long" Rule ---
    # We set the position to +1 (Long) only where BOTH of our conditions are met:
    # a) The primary prediction signal is +1 (our model wants to go long).
    # b) The current closing price is above the long-term trend line,
    #    confirming the market is in a general uptrend.
    long_condition = (predictions == 1) & (data['close'] > long_term_trend)
    positions[long_condition] = 1

    # --- Step 3: Apply the "Go Short" Rule ---
    # We set the position to -1 (Short) only where BOTH of our conditions are met:
    # a) The primary prediction signal is -1 (our model wants to go short).
    # b) The current closing price is below the long-term trend line,
    #    confirming the market is in a general downtrend.
    short_condition = (predictions == -1) & (data['close'] < long_term_trend)
    positions[short_condition] = -1

    return positions



# --- Vectorized Backtesting Engine ---
def run_vectorized_backtest(price_data: pd.DataFrame, positions: pd.Series) -> pd.Series:
    """
    Runs a vectorized backtest on a set of price data and position signals.

    This function simulates the process of trading a strategy over historical
    data to see how it would have performed, including realistic costs.

    Args:
        price_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        positions (pd.Series): Series with your strategy's target positions
                               (1 for long, -1 for short, 0 for flat).

    Returns:
        pd.Series: A Series representing the daily or hourly net returns of the strategy.
    """

    asset_returns = price_data['open'].pct_change().shift(-1)
    lagged_positions = positions.shift(1).fillna(0)
    strategy_gross_returns = asset_returns * lagged_positions

    # --- model Trading Costs ---
    # trading cost
    trades = lagged_positions.diff().fillna(0).abs()
    commission_costs = trades * CONFIG['commission_rate']

    # Model "slippage" - the hidden cost of a fast market.
    # When you place an order, the price might move slightly against you before
    # the trade is actually executed. This is slippage.
    # We model this as a small penalty that is larger when the market is more
    # volatile (i.e., when 'asset_returns' is large).
    # This is an *estimated* cost to make the backtest more conservative and realistic.
    slippage_costs = trades * asset_returns.abs() * CONFIG['slippage_factor']


    total_costs = commission_costs + slippage_costs
    strategy_net_returns = strategy_gross_returns - total_costs
    return strategy_net_returns.fillna(0)


def generate_strategy_positions(strategy_name, data, predictions=None):
    """
    Dispatcher to generate positions based on the strategy name.
    """
    if strategy_name == 'mean_reversion':
        return generate_mean_reversion_positions(data)
    elif strategy_name == 'buy_and_hold':
        return generate_buy_and_hold_positions(data)
    elif strategy_name == 'long_short':
        if predictions is None:
            raise ValueError("Long-short strategy requires predictions.")
        return generate_long_short_positions(predictions)
    elif strategy_name == 'trend_following':
        if predictions is None:
            raise ValueError("Trend-following strategy requires predictions.")
        return generate_trend_following_positions(data, predictions)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def calculate_portfolio_return(strategy_name, weights, all_coin_data, all_coin_predictions=None):
    """
    Calculates the returns of a portfolio of assets, given a trading strategy and asset weights.

    Args:
        strategy_name (str): The name of the strategy to use.
        weights (dict): A dictionary with coin names as keys and their portfolio weights as values.
        all_coin_data (dict): A dictionary of pandas DataFrames, one for each coin, with price data.
        all_coin_predictions (dict, optional): A dictionary of pandas Series, one for each coin,
                                               with prediction signals. Required for some strategies.

    Returns:
        pd.Series: A Series representing the daily net returns of the portfolio.
    """
    all_weighted_returns = {}
    
    for coin, weight in weights.items():
        if coin not in all_coin_data.keys():
            print(f"Warning: Price data for {coin} not found. Skipping.")
            continue

        price_data = all_coin_data[coin]
        predictions = None
        if all_coin_predictions and coin in all_coin_predictions:
            predictions = all_coin_predictions[coin]

        positions = generate_strategy_positions(strategy_name, price_data, predictions)
        net_returns = run_vectorized_backtest(price_data, positions)
        all_weighted_returns[coin] = net_returns * weight

    if not all_weighted_returns:
        return pd.Series(dtype=np.float64)

    portfolio_returns_df = pd.DataFrame(all_weighted_returns)
    portfolio_return = portfolio_returns_df.sum(axis=1, min_count=0).fillna(0)

    return portfolio_return


# --- Module 5: Main Execution and Reporting ---
# def main():
#     """
#     Main function to run the simulation, calculate KPIs, and save results.
#     """
#     # --- Step 1: Setup - Create dummy data for demonstration ---
#     # In a real scenario, you would load your OHLCV data here for each coin.
#     coin_name = 'BTC'
#     output_path = f'./performance_report_{coin_name}.csv'
    
#     print(f"Generating dummy data for {coin_name}...")
#     dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=5000, freq=CONFIG['data_frequency']))
#     price_path = 100 + np.random.randn(5000).cumsum() * 0.5
#     data = pd.DataFrame({
#         'Open': price_path,
#         'High': price_path * 1.01,
#         'Low': price_path * 0.99,
#         'Close': price_path
#     }, index=dates)

#     # Create dummy ML prediction signals (replace with your model's output)
#     dummy_predictions = pd.Series(np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3]), index=data.index)
    
#     # --- Step 2: Run Simulation for Each Strategy ---
#     strategies_to_test = ['buy_and_hold', 'naive_long_short', 'trend_following_overlay', 'mean_reversion_bb']
#     #all_results =

#     print(f"\nRunning backtests for {coin_name}...")
#     for strategy_name in strategies_to_test:
#         print(f"  - Testing strategy: '{strategy_name}'")
        
#         # Generate the positions for the current strategy
#         positions = generate_strategy_positions(strategy_name, data, dummy_predictions)
        
#         # Run the backtest to get net returns
#         net_returns = run_vectorized_backtest(data, positions)
        
#         # Calculate performance KPIs
#         kpis = calculate_performance_metrics(net_returns, CONFIG['data_frequency'])
#         kpis = strategy_name
#         all_results.append(kpis)

#     # --- Step 3: Consolidate and Save Results ---
#     results_df = pd.DataFrame(all_results)
#     results_df = results_df.set_index('Strategy') # Set strategy name as the index

#     # Format percentage columns for better readability
#     for col in:
#         results_df[col] = results_df[col].apply(lambda x: f"{x:.2%}")

#     # Save the results to a CSV file
#     results_df.to_csv(output_path)
#     print(f"\nPerformance report saved to: {os.path.abspath(output_path)}")
#     print("\n--- Performance Report ---")
#     print(results_df.to_string())


if __name__ == '__main__':
    main()