import numpy as np
import pandas as pd
import os

CONFIG = {
    'initial_capital': 10000.0,
    'commission_rate': 0,       # 0.1% trading fee 0.001
    'slippage_factor': 0,      # 0.05% slippage 0.0005
    'risk_free_rate': 0.0,
    'strategy_params': {
        'long_short': {
            'buy_threshold': 0.5,
            'sell_threshold': 0.5
        },
        'trend_following': {
            'sma_window': 50       # The lookback period for the long-term SMA that defines the trend. A common value is 50, 100, or 200.
        }
    }
}


# --- Strategy Signal Generation ---
# --- Buy and Hold Strategy (No need for prediction) ---
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

# --- Long-short strategy ---
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

# --- Long-only strategy ---
def generate_long_only_positions(predictions):
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
    # short_condition = (1 - predictions) > CONFIG['strategy_params']['long_short']['sell_threshold']
    # positions[short_condition] = 0
    return positions

# --- trend_following_overlay ---
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
    if strategy_name == 'buy_and_hold':
        return generate_buy_and_hold_positions(data)
    elif strategy_name == 'long_short':
        if predictions is None:
            raise ValueError("Long-short strategy requires predictions.")
        return generate_long_short_positions(predictions)
    elif strategy_name == 'long_only':
        if predictions is None:
            raise ValueError("Long-only strategy requires predictions.")
        return generate_long_only_positions(predictions)
    elif strategy_name == 'trend_following':
        if predictions is None:
            raise ValueError("Trend-following strategy requires predictions.")
        return generate_trend_following_positions(data, predictions)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def calculate_single_coin_return(strategy_name, result_df):
    price_cols = ['open', 'close']
    price_data = result_df[price_cols]
    predictions = result_df['y_test_pred']
    positions = generate_strategy_positions(strategy_name, price_data, predictions)
    net_returns = run_vectorized_backtest(price_data, positions)
    return net_returns


def calculate_portfolio_return(strategy_name, weights, all_coin_data, all_coin_predictions=None):
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