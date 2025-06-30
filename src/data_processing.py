import pandas as pd
import numpy as np
import os
from feature_engineering import run_feature_engineering
from utils import reduce_mem_usage
from config import get_model_config


def load_data(path, coin_id_col, ts_col, chosen_coin=None):
    '''Load data from a parquet file and preprocess it.'''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])

    df.sort_values(by=[coin_id_col, ts_col], inplace=True)
    if chosen_coin is not None:
        df = df[df[coin_id_col] == chosen_coin]
    return df


def resample_to_granularity(df, config):
    """Resample 1-minute OHLCV data to specified granularity."""
    df_resampled = df.copy()
    
    timestamp_col = config['data']['columns']['timestamp']
    coin_id_col = config['data']['columns']['coin_id']
    freq = config['data']['granularity']
    
    df_resampled[timestamp_col] = pd.to_datetime(df_resampled[timestamp_col])
    df_resampled = df_resampled.set_index(timestamp_col)
    
    agg_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'close_time': 'last', 'quote_asset_volume': 'sum',
        'number_of_trades': 'sum', 'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum', 'missing_flag': 'max',
        coin_id_col: 'first', 'year': 'first'
    }
    
    resampled_groups = []
    for coin_id, group in df_resampled.groupby(coin_id_col, observed=True):
        group_clean = group.drop(columns=[coin_id_col])
        resampled_group = group_clean.resample(freq, label='left', closed='left').agg({
            col: rule for col, rule in agg_rules.items() 
            if col != coin_id_col
        })
        resampled_group[coin_id_col] = coin_id
        resampled_group = resampled_group.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        resampled_groups.append(resampled_group)
    
    df_final = pd.concat(resampled_groups, ignore_index=False)
    df_final = df_final.reset_index()
    df_final['year'] = df_final[timestamp_col].dt.year
    
    df_final[coin_id_col] = df_final[coin_id_col].astype('category')
    df_final['year'] = df_final['year'].astype('category')
    df_final = df_final.sort_values([coin_id_col, timestamp_col]).reset_index(drop=True)
    
    return df_final


def split_data(df: pd.DataFrame, config, round_frequency: str = 'month'):
    """Splits data into training and testing sets."""
    ts_col = config['data']['columns']['timestamp']
    train_ratio = config['splitting']['train_ratio']
    
    min_ts = df[ts_col].min()
    max_ts = df[ts_col].max()
    total_duration = max_ts - min_ts
    exact_cutoff_ts = min_ts + (total_duration * train_ratio)

    if round_frequency == 'month':
        rounded_cutoff_ts = exact_cutoff_ts.to_period('M').to_timestamp()
    elif round_frequency == 'day':
        rounded_cutoff_ts = exact_cutoff_ts.normalize()
    else:
        rounded_cutoff_ts = exact_cutoff_ts

    train_df = df[df[ts_col] < rounded_cutoff_ts].copy()
    test_df = df[df[ts_col] >= rounded_cutoff_ts].copy()
    return train_df, test_df


def create_target(df: pd.DataFrame, config) -> pd.DataFrame:
    """Calculates the target variable for the given dataframe."""
    coin_id_col = config['data']['columns']['coin_id']
    target_col = config['data']['columns']['target']
    horizon = config['data']['prediction_horizon_steps']
    
    df_out = df.copy()
    df_out['future_price'] = df_out.groupby(coin_id_col, observed = True)['close'].shift(-horizon)
    df_out['future_log_return'] = np.log(df_out['future_price']) - np.log(df_out['close'])
    df_out[target_col] = (df_out['future_log_return'] > 0).astype(int)
    df_out = df_out.dropna(subset=['future_price', target_col])
    return df_out


def get_features_and_target(coin_id, freq, split=True, environment="development"):
    '''Load data, split into train/test, and create features/target.'''
    config = get_model_config("base", coin_id, freq, environment)
    
    df = load_data(
        path=config['data']['processed_data_file'], 
        coin_id_col=config['data']['columns']['coin_id'], 
        ts_col=config['data']['columns']['timestamp'], 
        chosen_coin=coin_id
    )
    
    df = resample_to_granularity(df=df, config=config)
    df = create_target(df=df, config=config)
    df = reduce_mem_usage(df, 'processed_data')
    df = run_feature_engineering(
        df=df, 
        coin_id_col=config['data']['columns']['coin_id'], 
        ts_col=config['data']['columns']['timestamp']
    ) 
    
    if split:
        train_df, test_df = split_data(df=df, config=config)
        return train_df, test_df
    else:
        return df