import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks


def _calculate_momentum_indicators(df: pd.DataFrame, price_col='close', ma_windows=[5, 10, 20, 30, 50, 60, 100]) -> pd.DataFrame:
    """Calculates momentum indicators like SMA, EMA, MACD, RSI."""
    df_out = pd.DataFrame(index=df.index)
    
    # Lagged returns
    df_out['return_1'] = df[price_col].pct_change(periods=1)
    df_out['return_3'] = df[price_col].pct_change(periods=3)
    
    # Rolling statistics
    df_out['ma_5'] = df[price_col].rolling(window=5).mean()
    df_out['std_5'] = df[price_col].rolling(window=5).std()
    
    returns_1 = df[price_col].pct_change(periods=1)
    
    # Volatility
    for window in [5, 20]:
        df_out[f'vol_{window}'] = returns_1.rolling(window=window).std()
    
    # Volume moving averages
    for window in [5, 20, 50]:
        df_out[f'volu_ma_{window}'] = df['volume'].rolling(window=window).mean()
    
    # SMA/EMA features
    df_out['sma_12'] = df[price_col].rolling(window=12).mean()
    df_out['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df_out['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    
    for window in [20, 50]:
        df_out[f'ema_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
        df_out[f'price_vs_ema_{window}'] = (df[price_col] - df_out[f'ema_{window}']) / df_out[f'ema_{window}']
    
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
    df_out['rsi_momentum'] = df_out['rsi'].diff()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df_out['stoch_k'] = 100 * (df[price_col] - low_min) / (high_max - low_min)
    
    return df_out


def _calculate_volatility_indicators(df: pd.DataFrame, window=20) -> pd.DataFrame:
    """Calculates volatility indicators like Bollinger Bands and ATR."""
    df_out = pd.DataFrame(index=df.index)
    
    # Bollinger Bands
    df_out['bb_mid'] = df['close'].rolling(window=window).mean()
    df_out['bb_std'] = df['close'].rolling(window=window).std()
    sma_20 = df['close'].rolling(window=window).mean()
    std_dev_20 = df['close'].rolling(window=window).std()
    df_out['bb_upper'] = sma_20 + (std_dev_20 * 2)
    df_out['bb_lower'] = sma_20 - (std_dev_20 * 2)
    df_out['bb_position'] = (df['close'] - df_out['bb_lower']) / (df_out['bb_upper'] - df_out['bb_lower'])
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_out['atr_14'] = true_range.rolling(window=14).mean()
    df_out['atr_ratio'] = true_range / df_out['atr_14']
    
    # Volatility ratios
    for period in [5, 10, 20]:
        df_out[f'volatility_{period}'] = df['close'].rolling(window=period).std() / df['close'].rolling(window=period).mean()
    for period in [10, 20]:  
        df_out[f'volatility_change_{period}'] = df_out[f'volatility_{period}'].pct_change()
    
    # Parkinson Volatility
    df_out['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                                     ((np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean()))
    
    # Garman-Klass Volatility
    df_out['garman_klass_vol'] = np.sqrt(((0.5 * (np.log(df['high'] / df['low']) ** 2)) - 
                                         ((2 * np.log(2) - 1) * (np.log(df['close'] / df['close'].shift(1)) ** 2))).rolling(window=20).mean())
    
    return df_out


def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features based on volume and trade data."""
    df_out = pd.DataFrame(index=df.index)
    
    # VWAP
    df_out['vwap_5'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Volume ratios and pressure
    epsilon = 1e-9
    df_out['volume_ratio_buy_pressure'] = df['taker_buy_quote_asset_volume'] / (df['quote_asset_volume'] + epsilon)
    df_out['vwap_bar'] = df['quote_asset_volume'] / df['volume']
    df_out['avg_trade_size'] = df['volume'] / df['number_of_trades']
    
    taker_sell_volume = df['volume'] - df['taker_buy_base_asset_volume']
    df_out['taker_buy_sell_ratio'] = df['taker_buy_base_asset_volume'] / (taker_sell_volume + epsilon)
    
    # Volume momentum and trends
    for period in [5, 10, 20]:
        df_out[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df_out[f'volume_ratio_{period}'] = df['volume'] / df_out[f'volume_sma_{period}']
    for period in [5, 10]:    
        df_out[f'volume_momentum_{period}'] = df['volume'].pct_change(periods=period)
    
    # Technical volume indicators
    df_out['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df_out['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).fillna(0).cumsum()
    
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df_out['adl'] = (clv * df['volume']).fillna(0).cumsum()
    df_out['cmf'] = (clv * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df_out['volume_roc'] = df['volume'].pct_change(periods=10) * 100
    
    df_out['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df_out['price_vs_vwap'] = (df['close'] - df_out['vwap']) / df_out['vwap']
        
    return df_out


def _calculate_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates price action patterns and candlestick features."""
    df_out = pd.DataFrame(index=df.index)
    
    df_out['hl_range'] = df['high'] - df['low']
    
    # Candlestick patterns
    df_out['body_size'] = np.abs(df['close'] - df['open'])
    df_out['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df_out['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df_out['body_ratio'] = df_out['body_size'] / (df['high'] - df['low'])
    df_out['shadow_ratio'] = (df_out['upper_shadow'] + df_out['lower_shadow']) / (df['high'] - df['low'])
    
    # Pattern recognition
    df_out['is_doji'] = np.where(df_out['body_ratio'] < 0.1, 1, 0)
    df_out['is_hammer'] = np.where((df_out['lower_shadow'] > 2 * df_out['body_size']) & 
                                  (df_out['upper_shadow'] < df_out['body_size']), 1, 0)
    df_out['is_shooting_star'] = np.where((df_out['upper_shadow'] > 2 * df_out['body_size']) & 
                                         (df_out['lower_shadow'] < df_out['body_size']), 1, 0)
    
    # Price gaps
    df_out['gap_up'] = np.where(df['open'] > df['high'].shift(1), 1, 0)
    df_out['gap_down'] = np.where(df['open'] < df['low'].shift(1), 1, 0)
    
    # Support and resistance levels
    for period in [20, 50]:
        df_out[f'resistance_{period}'] = df['high'].rolling(window=period).max()
        df_out[f'support_{period}'] = df['low'].rolling(window=period).min()
        df_out[f'price_vs_resistance_{period}'] = (df['close'] - df_out[f'resistance_{period}']) / df_out[f'resistance_{period}']
        df_out[f'price_vs_support_{period}'] = (df['close'] - df_out[f'support_{period}']) / df_out[f'support_{period}']
    
    # Price momentum
    df_out['price_momentum_1'] = df['close'].pct_change(periods=1)
    df_out['price_momentum_3'] = df['close'].pct_change(periods=3)
    df_out['price_momentum_5'] = df['close'].pct_change(periods=5)
    df_out['price_acceleration'] = df_out['price_momentum_1'].diff()
    
    # High-Low spread
    df_out['hl_spread'] = (df['high'] - df['low']) / df['close']
    df_out['hl_spread_ma'] = df_out['hl_spread'].rolling(window=20).mean()
    
    # Price efficiency ratio
    for period in [10, 20]:
        path_length = df['close'].diff().abs().rolling(window=period).sum()
        direct_distance = (df['close'] - df['close'].shift(period)).abs()
        df_out[f'efficiency_ratio_{period}'] = direct_distance / path_length
    
    return df_out


def _calculate_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates market microstructure features."""
    df_out = pd.DataFrame(index=df.index)
    
    # Trade size analysis
    df_out['trade_size_mean'] = df['volume'].rolling(window=20).mean() / df['number_of_trades'].rolling(window=20).mean()
    df_out['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Order flow imbalance
    df_out['order_imbalance'] = (df['taker_buy_base_asset_volume'] - (df['volume'] - df['taker_buy_base_asset_volume'])) / df['volume']
    df_out['order_imbalance_ma'] = df_out['order_imbalance'].rolling(window=20).mean()
    
    # Market efficiency features
    for period in [10, 20]:
        def hurst_exponent(series, window):
            hurst_values = []
            for i in range(window-1, len(series)):
                window_data = series.iloc[i-window+1:i+1]
                if len(window_data) < 2:
                    hurst_values.append(np.nan)
                    continue
                
                mean_val = window_data.mean()
                deviations = window_data - mean_val
                cumulative = deviations.cumsum()
                R = cumulative.max() - cumulative.min()
                S = window_data.std()
                
                if S == 0:
                    hurst_values.append(np.nan)
                else:
                    hurst_values.append(np.log(R/S) / np.log(window))
            
            return pd.Series(hurst_values, index=series.index[window-1:])
        
        df_out[f'hurst_{period}'] = hurst_exponent(df['close'], period)
    
    return df_out


def _calculate_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Calculates cyclical time-based features."""
    df_out = pd.DataFrame(index=df.index)
    
    timestamp = df[ts_col]
    df_out['hour_sin'] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    df_out['hour_cos'] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    df_out['dayofweek_sin'] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    df_out['dayofweek_cos'] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    
    return df_out


def run_feature_engineering(df: pd.DataFrame, coin_id_col: str, ts_col: str) -> pd.DataFrame:
    """Main function to run all feature engineering steps for each coin."""
    all_features = []
    for coin in df[coin_id_col].unique():
        coin_df = df[df[coin_id_col] == coin].copy()
        
        momentum_features = _calculate_momentum_indicators(coin_df)
        volatility_features = _calculate_volatility_indicators(coin_df)
        volume_features = _calculate_volume_features(coin_df)
        price_action_features = _calculate_price_action_features(coin_df)
        market_microstructure_features = _calculate_market_microstructure_features(coin_df)
        time_features = _calculate_time_features(coin_df, ts_col)
        
        coin_with_features = pd.concat([
            coin_df, momentum_features, volatility_features, volume_features,
            price_action_features, market_microstructure_features,
            time_features
        ], axis=1)
        all_features.append(coin_with_features)
        
    featured_df = pd.concat(all_features)
    featured_df = featured_df.dropna()
    
    return featured_df
