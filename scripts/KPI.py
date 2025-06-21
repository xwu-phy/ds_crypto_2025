
import pandas as pd
import numpy as np
import os

# --- Module 1: Configuration ---
# Centralized dictionary for all trading and backtesting parameters.
# This makes it easy to adjust settings without changing the core logic.
CONFIG = {
    'risk_free_rate': 0.0
}


# --- Module 2: Performance Metrics Calculation ---
def calculate_performance_metrics(returns_series: pd.Series, freq: str = '1H') -> dict:
    """
    Calculates a dictionary of performance metrics based on a series of returns.
    This version includes detailed comments explaining each KPI.

    Args:
        returns_series (pd.Series): A pandas Series of periodic strategy returns.
        freq (str): The frequency of the data ('1H', '4H', '1D') for annualization.

    Returns:
        dict: A dictionary containing the calculated Key Performance Indicators (KPIs).
    """
    if returns_series.empty:
        return {
            'Annualized Return': 0.0, 'Annualized Volatility': 0.0,
            'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
            'Maximum Drawdown': 0.0, 'Calmar Ratio': 0.0,
            'Kelly Criterion': 0.0
        }

    # --- Annualization Factor ---
    # Definition: The number of trading periods in a single year.
    # Equation: Based on the frequency of the data (e.g., 24*365 for hourly).
    if 'H' in freq.upper():
        periods_per_year = 24 * 365 / int(freq.upper().replace('H', ''))
    elif 'D' in freq.upper():
        periods_per_year = 252 # Trading days in a year
    else:
        periods_per_year = 24 * 365 # Default to hourly if format is unknown

    # --- Annualized Return ---
    # Definition: The geometric average amount of money earned by an investment each year.
    # It represents the constant return rate that would yield the same final value.
    # Equation: (1 + Mean_Periodic_Return) ^ Periods_Per_Year - 1
    annualized_return = (1 + returns_series.mean()) ** periods_per_year - 1

    # --- Annualized Volatility ---
    # Definition: A measure of how much the strategy's returns fluctuate over a year.
    # It is the annualized standard deviation of the periodic returns.
    # Equation: Standard_Deviation_of_Periodic_Returns * sqrt(Periods_Per_Year)
    annualized_volatility = returns_series.std() * np.sqrt(periods_per_year)

    # --- Sharpe Ratio ---
    # Definition: Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk.
    # It represents the average return earned in excess of the risk-free rate per unit of *total* volatility. [1, 2]
    # Equation: (Annualized_Return - Risk_Free_Rate) / Annualized_Volatility [3]
    sharpe_ratio = (annualized_return - CONFIG['risk_free_rate']) / annualized_volatility if annualized_volatility!= 0 else 0

    # --- Sortino Ratio ---
    # Definition: A variation of the Sharpe Ratio that only considers downside volatility (harmful risk).
    # It measures the risk-adjusted return by penalizing only those returns falling below a target. [4, 5]
    # Equation: (Annualized_Return - Risk_Free_Rate) / Downside_Deviation [6]
    downside_returns = returns_series[returns_series < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if not downside_returns.empty else 0
    sortino_ratio = (annualized_return - CONFIG['risk_free_rate']) / downside_deviation if downside_deviation!= 0 else 0

    # --- Maximum Drawdown (MDD) ---
    # Definition: The largest single drop from a peak to a trough in a portfolio's value,
    # before a new peak is achieved. It quantifies the worst-case loss an investor would have endured. [7]
    # Equation: (Trough_Value - Peak_Value) / Peak_Value [7]
    equity_curve = (1 + returns_series).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # --- Calmar Ratio ---
    # Definition: Measures the performance of an investment on a risk-adjusted basis relative to its maximum drawdown.
    # It helps answer: "How much return am I getting for the maximum pain endured?" [8]
    # Equation: Annualized_Return / abs(Maximum_Drawdown) [8]
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown!= 0 else 0

    # --- Kelly Criterion ---
    # Definition: A mathematical formula for position sizing that determines the optimal fraction of capital
    # to risk on a single trade to maximize long-term growth. It is not a performance metric itself,
    # but a useful calculation based on historical performance.
    # Equation: K% = W - ((1 - W) / R)
    
    # First, filter out non-trading periods for accurate win/loss calculation
    traded_returns = returns_series[returns_series!= 0]
    
    if not traded_returns.empty:
        # Calculate W (Win Probability)
        # Definition: The historical probability of a trade being profitable.
        # Equation: Number_of_Winning_Trades / Total_Number_of_Trades
        winning_trades = traded_returns[traded_returns > 0]
        W = len(winning_trades) / len(traded_returns)

        # Calculate R (Win/Loss Ratio)
        # Definition: The ratio of the average gain from winning trades to the average loss from losing trades.
        # Equation: Average_Win / abs(Average_Loss)
        losing_trades = traded_returns[traded_returns < 0]
        average_win = winning_trades.mean() if not winning_trades.empty else 0
        average_loss = losing_trades.mean() if not losing_trades.empty else 0
        R = average_win / abs(average_loss) if average_loss!= 0 else float('inf')

        # Calculate Kelly Percentage
        if R > 0 and R!= float('inf'):
            kelly_percentage = W - ((1 - W) / R)
        else:
            kelly_percentage = 0.0 # If R is 0 or infinite, formula suggests no bet
    else:
        kelly_percentage = 0.0 # No trades, so Kelly is 0

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Kelly Criterion': kelly_percentage
    }
