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
    """Calculates performance metrics based on a series of returns."""
    if returns_series.empty:
        return {
            'Annualized Return': 0.0, 'Annualized Volatility': 0.0,
            'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
            'Maximum Drawdown': 0.0, 'Calmar Ratio': 0.0,
            'Kelly Criterion': 0.0
        }

    # Annualization factor
    if 'H' in freq.upper():
        periods_per_year = 24 * 365 / int(freq.upper().replace('H', ''))
    elif 'D' in freq.upper():
        periods_per_year = 252  # Trading days in a year
    else:
        periods_per_year = 24 * 365  # Default to hourly

    # Annualized Return
    cumulative_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(returns_series)) - 1

    # Annualized Volatility
    annualized_volatility = returns_series.std(ddof=0) * np.sqrt(periods_per_year)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - CONFIG['risk_free_rate']) / annualized_volatility if annualized_volatility != 0 else 0

    # Sortino Ratio
    downside_returns = returns_series[returns_series < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if not downside_returns.empty else 0
    sortino_ratio = (annualized_return - CONFIG['risk_free_rate']) / downside_deviation if downside_deviation != 0 else 0

    # Maximum Drawdown
    equity_curve = (1 + returns_series).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Kelly Criterion
    traded_returns = returns_series[returns_series != 0]
    
    if not traded_returns.empty:
        winning_trades = traded_returns[traded_returns > 0]
        W = len(winning_trades) / len(traded_returns)

        losing_trades = traded_returns[traded_returns < 0]
        average_win = winning_trades.mean() if not winning_trades.empty else 0
        average_loss = losing_trades.mean() if not losing_trades.empty else 0
        R = average_win / abs(average_loss) if average_loss != 0 else float('inf')

        if R > 0 and R != float('inf'):
            kelly_percentage = W - ((1 - W) / R)
        else:
            kelly_percentage = 0.0
    else:
        kelly_percentage = 0.0

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Kelly Criterion': kelly_percentage
    }
