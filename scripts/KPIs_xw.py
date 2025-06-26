
## Define KPIs 
#### (geometric compounding)

import pandas as pd
import numpy as np

# --- Module 1: Configuration ---
CONFIG = {
    'risk_free_rate': 0.0
}

# --- Module 2: Performance Metrics Calculation ---
def calculate_performance_metrics(returns_series: pd.Series, freq: str = '1H') -> dict:
    """
    Calculates a dictionary of performance metrics based on a series of returns,
    using true geometric compounding for cumulative & annualized returns.

    Args:
        returns_series (pd.Series): periodic (e.g. hourly/daily) simple returns R_t.
        freq (str): data frequency for annualization ('1H','4H','1D', etc).

    Returns:
        dict: KPI metrics including Cumulative Return, CAGR, volatility, ratios, etc.
    """
    # handle empty series
    if returns_series.empty:
        return {
            'Cumulative Return':      0.0,
            'Annualized Return':      0.0,
            'Annualized Volatility':  0.0,
            'Sharpe Ratio':           0.0,
            'Sortino Ratio':          0.0,
            'Maximum Drawdown':       0.0,
            'Calmar Ratio':           0.0,
            'Kelly Criterion':        0.0
        }

    # --- determine periods per year ---
    f = freq.upper()
    if 'H' in f:
        periods_per_year = 24 * 365 / int(f.replace('H',''))
    elif 'D' in f:
        periods_per_year = 252
    else:
        periods_per_year = 24 * 365

    # number of observations
    n_periods = returns_series.shape[0]

    # --- Cumulative Return (geometric) ---
    cumulative_return = (1 + returns_series).prod() - 1

    # --- Annualized Return (CAGR) ---
    # (1 + total_return)^(periods_per_year / n_periods) - 1
    total_return = 1 + cumulative_return
    annualized_return = total_return ** (periods_per_year / n_periods) - 1

    # --- Annualized Volatility ---
    annualized_volatility = returns_series.std(ddof=1) * np.sqrt(periods_per_year)

    # --- Sharpe Ratio (using CAGR) ---
    excess_return = annualized_return - CONFIG['risk_free_rate']
    sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0.0

    # --- Sortino Ratio ---
    downside = returns_series[returns_series < 0]
    downside_dev = (
        downside.std(ddof=1) * np.sqrt(periods_per_year)
        if not downside.empty else 0.0
    )
    sortino_ratio = excess_return / downside_dev if downside_dev != 0 else 0.0

    # --- Equity curve & Max Drawdown ---
    equity_curve = (1 + returns_series).cumprod()
    running_max   = equity_curve.cummax()
    drawdown      = (equity_curve - running_max) / running_max
    max_drawdown  = drawdown.min()

    # --- Calmar Ratio ---
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown != 0 else 0.0
    )

    # --- Kelly Criterion ---
    traded = returns_series[returns_series != 0]
    if not traded.empty:
        wins   = traded[traded > 0]
        losses = traded[traded < 0]
        W = len(wins) / len(traded)
        avg_win  = wins.mean() if len(wins)  else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        R = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
        kelly = W - (1 - W) / R if (R > 0 and R != np.inf) else 0.0
    else:
        kelly = 0.0

    return {
        'Cumulative Return':      cumulative_return,
        'Annualized Return':      annualized_return,
        'Annualized Volatility':  annualized_volatility,
        'Sharpe Ratio':           sharpe_ratio,
        'Sortino Ratio':          sortino_ratio,
        'Maximum Drawdown':       max_drawdown,
        'Calmar Ratio':           calmar_ratio,
        'Kelly Criterion':        kelly
    }