import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _calculate_momentum_indicators(df: pd.DataFrame, price_col='close', ma_windows=[10, 30, 60]) -> pd.DataFrame:
    """Calculates classic momentum indicators like SMA, EMA, MACD, RSI."""
    logger.info("Calculating momentum indicators...")
    df_out = pd.DataFrame(index=df.index) # Create a new df to hold features
    
    # Simple and Exponential Moving Averages
    for window in ma_windows:
        df_out[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
        df_out[f'ema_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()

    # MACD
    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    df_out['macd'] = ema_12 - ema_26
    df_out['macd_signal'] = df_out['macd'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_out['rsi'] = 100 - (100 / (1 + rs))

    # roc
    df_out['roc_10'] = df[price_col].pct_change(periods=10) * 100
    return df_out

def _calculate_volatility_indicators(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """Calculates volatility indicators like Bollinger Bands and ATR."""
    logger.info("Calculating volatility indicators...")
    df_out = pd.DataFrame(index=df.index)
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=window).mean()
    std_dev_20 = df['close'].rolling(window=window).std()
    df_out['bb_upper'] = sma_20 + (std_dev_20 * 2)
    df_out['bb_lower'] = sma_20 - (std_dev_20 * 2)
    df_out['bb_width'] = (df_out['bb_upper'] - df_out['bb_lower']) / sma_20 # Normalized width

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_out['atr_14'] = true_range.rolling(window=14).mean()
    
    return df_out

def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features based on volume and trade data."""
    logger.info("Calculating volume & trade flow features...")
    df_out = pd.DataFrame(index=df.index)
    
    # Volume-Weighted Average Price per bar
    df_out['vwap_bar'] = df['quote_asset_volume'] / df['volume']
    # Average trade size
    df_out['avg_trade_size'] = df['volume'] / df['number_of_trades']
    # Taker buy/sell ratio
    taker_sell_volume = df['volume'] - df['taker_buy_base_asset_volume']
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-9 
    df_out['taker_buy_sell_ratio'] = df['taker_buy_base_asset_volume'] / (taker_sell_volume + epsilon)

    return df_out

def _calculate_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Calculates cyclical time-based features."""
    logger.info("Calculating time features...")
    df_out = pd.DataFrame(index=df.index)
    
    timestamp = df[ts_col]
    df_out['hour_sin'] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    df_out['hour_cos'] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    df_out['dayofweek_sin'] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    df_out['dayofweek_cos'] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    
    return df_out

# -- Main Feature Engineering Pipeline --
def run_feature_engineering(df: pd.DataFrame, coin_id_col: str, ts_col: str) -> pd.DataFrame:
    """
    Main function to run all feature engineering steps for each coin.
    """
    logger.info("Starting feature engineering process...")
    
    all_features = []
    for coin in df[coin_id_col].unique():
        logger.info(f"Processing features for coin: {coin}...")
        coin_df = df[df[coin_id_col] == coin].copy()
        
        # --- Call modular feature functions ---
        momentum_features = _calculate_momentum_indicators(coin_df)
        volatility_features = _calculate_volatility_indicators(coin_df)
        volume_features = _calculate_volume_features(coin_df)
        time_features = _calculate_time_features(coin_df, ts_col)
        
        # Combine original data with new features
        coin_with_features = pd.concat([coin_df, momentum_features, volatility_features, volume_features, time_features], axis=1)
        all_features.append(coin_with_features)
        
    featured_df = pd.concat(all_features)
    
    logger.info(f"Shape before dropping NaNs from feature engineering: {featured_df.shape}")
    featured_df = featured_df.dropna()
    logger.info(f"Shape after dropping NaNs: {featured_df.shape}")
    
    return featured_df
