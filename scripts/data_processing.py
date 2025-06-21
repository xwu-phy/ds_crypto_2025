# erdos_src/data_processing.py
import pandas as pd
import numpy as np
import os
from config import CFG
from feature_engineering_xh import run_feature_engineering


def load_data(path, coin_id_col, ts_col, chosen_coin = None):
    '''
    Load data from a parquet file and preprocess it.
    Args:
        path (str): Path to the parquet file.
        coin_id_col (str): Name of the column containing coin IDs.
        ts_col (str): Name of the timestamp column.
        chosen_coin (str, optional): Specific coin ID to filter data. Defaults to None.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with coin data.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])

    df.sort_values(by=[coin_id_col, ts_col], inplace=True)
    if chosen_coin is not None:
        df = df[df[coin_id_col] == chosen_coin]
    return df


def resample_to_granularity(df):
    """
    Resample 1-minute OHLCV data to specified granularity in hours.
    Parameters:
    df (pd.DataFrame): Input dataframe with 1-minute OHLCV data
    Returns:
    pd.DataFrame: Resampled dataframe at CFG.GRANULARITY hour intervals
    """
    df_resampled = df.copy()
    
    # Ensure timestamp is datetime and set as index for resampling
    df_resampled[CFG.TIMESTAMP_COLUMN] = pd.to_datetime(df_resampled[CFG.TIMESTAMP_COLUMN])
    df_resampled = df_resampled.set_index(CFG.TIMESTAMP_COLUMN)
    
    # Define aggregation rules for OHLCV data
    agg_rules = {
        'open': 'first',                                    # First value in the period
        'high': 'max',                                      # Maximum value in the period  
        'low': 'min',                                       # Minimum value in the period
        'close': 'last',                                    # Last value in the period
        'volume': 'sum',                                    # Sum of volume
        'close_time': 'last',                              # Last close time in the period
        'quote_asset_volume': 'sum',                       # Sum of quote asset volume
        'number_of_trades': 'sum',                         # Sum of number of trades
        'taker_buy_base_asset_volume': 'sum',              # Sum of taker buy base volume
        'taker_buy_quote_asset_volume': 'sum',             # Sum of taker buy quote volume
        'missing_flag': 'max',                             # Max (if any missing, flag the period)
        CFG.COIN_ID_COLUMN: 'first',                       # Keep coin_id (should be same for all)
        'year': 'first'                                    # Keep year (should be same for most)
    }
    
    # Group by coin_id and resample each group separately
    freq = CFG.GRANULARITY
    resampled_groups = []
    for coin_id, group in df_resampled.groupby(CFG.COIN_ID_COLUMN, observed=True):
        # Remove the coin_id from the group since it's constant within each group
        group_clean = group.drop(columns=[CFG.COIN_ID_COLUMN])
        
        # Resample this coin's data
        resampled_group = group_clean.resample(freq, label='left', closed='left').agg({
            col: rule for col, rule in agg_rules.items() 
            if col != CFG.COIN_ID_COLUMN  # Skip coin_id in aggregation
        })
        
        # Add back the coin_id as a constant column
        resampled_group[CFG.COIN_ID_COLUMN] = coin_id
        
        # Remove any rows where we don't have complete OHLCV data
        resampled_group = resampled_group.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        resampled_groups.append(resampled_group)
    
    # Combine all resampled groups
    df_final = pd.concat(resampled_groups, ignore_index=False)
    df_final = df_final.reset_index()
    df_final['year'] = df_final[CFG.TIMESTAMP_COLUMN].dt.year
    
    # Ensure data types match original
    df_final[CFG.COIN_ID_COLUMN] = df_final[CFG.COIN_ID_COLUMN].astype('category')
    df_final['year'] = df_final['year'].astype('category')
    
    # Sort by timestamp and coin_id for consistency
    df_final = df_final.sort_values([CFG.COIN_ID_COLUMN, CFG.TIMESTAMP_COLUMN]).reset_index(drop=True)
    
    return df_final


def split_data(df: pd.DataFrame, ts_col: str, train_ratio: float, round_frequency: str = 'month'):
    """
    Splits the data into training and testing sets, rounding the split point
    to the beginning of the specified frequency ('month' or 'day').
    Args:
        df (pd.DataFrame): The DataFrame containing the data to be split.
        ts_col (str): The name of the timestamp column in the DataFrame.
        train_ratio (float): The ratio of the data to be used for training (0 < train_ratio < 1).
        round_frequency (str): The frequency to round the split point ('month' or 'day').
    Returns:
        tuple: A tuple containing the training DataFrame and the testing DataFrame.
    """
    # --- Calculate the exact cutoff timestamp ---
    min_ts = df[ts_col].min()
    max_ts = df[ts_col].max()
    total_duration = max_ts - min_ts
    exact_cutoff_ts = min_ts + (total_duration * train_ratio)

    # --- Round the cutoff timestamp down to the start of the specified frequency ---
    if round_frequency == 'month':
        # 'M' stands for Month End, so we get the period and convert back to the start of that period
        rounded_cutoff_ts = exact_cutoff_ts.to_period('M').to_timestamp()
    elif round_frequency == 'day':
        # 'D' stands for Day, this normalizes the time to midnight
        rounded_cutoff_ts = exact_cutoff_ts.normalize()
    else:
        # If no rounding is specified, use the exact point
        rounded_cutoff_ts = exact_cutoff_ts

    train_df = df[df[ts_col] < rounded_cutoff_ts].copy()
    test_df = df[df[ts_col] >= rounded_cutoff_ts].copy()

    return train_df, test_df


def create_target(df: pd.DataFrame, coin_id_col:str, target_col:str, horizon:int) -> pd.DataFrame:
    """
    Calculates the target variable for the given dataframe.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        coin_id_col (str): The name of the column containing coin IDs.
        target_col (str): The name of the target column to be created.
        horizon (int): The number of minutes into the future to predict.
    Returns:
        pd.DataFrame: The DataFrame with the target variable added.
    """

    df_out = df.copy()
    df_out['future_price'] = df_out.groupby(coin_id_col, observed = True)['close'].shift(-horizon)
    df_out['future_log_return'] = np.log(df_out['future_price']) - np.log(df_out['close'])
    df_out[target_col] = (df_out['future_log_return'] > 0).astype(int)
    df_out = df_out.dropna(subset=['future_price', target_col])
    return df_out

def _reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
            continue
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            continue
        col_type = dataframe[col].dtype
        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe
    
# -- Main function to get features and target
def get_features_and_target(path, split=True, chosen_coin=None):
    '''
    Load data, split into train/test, and create features/target.
    Args:
        path (str): Path to the preprocessed data file.
        chosen_coin (str, optional): Specific coin ID to filter data. Defaults to None.
    Returns:
        tuple: A tuple containing the training DataFrame and the testing DataFrame.
        or df
    '''
    df = load_data(path = CFG.FF_DATA_PATH, coin_id_col = CFG.COIN_ID_COLUMN, ts_col = CFG.TIMESTAMP_COLUMN, chosen_coin = None)
    df = resample_to_granularity(df = df)
    df = create_target(df = df, coin_id_col = CFG.COIN_ID_COLUMN, 
                          target_col = CFG.TARGET_COLUMN, 
                          horizon = CFG.PREDICTION_HORIZON_STEPS)

    df = _reduce_mem_usage(dataframe=df, dataset='df')
    df = run_feature_engineering(df = df, coin_id_col = CFG.COIN_ID_COLUMN, ts_col = CFG.TIMESTAMP_COLUMN) 
    

    train_df, test_df = split_data(df = df, ts_col = CFG.TIMESTAMP_COLUMN, train_ratio = CFG.TRAIN_RATIO, round_frequency=CFG.SPLIT_ROUND_FREQUENCY)

    if chosen_coin is not None:
        df = df[df[CFG.COIN_ID_COLUMN] == chosen_coin]
        train_df = train_df[train_df[CFG.COIN_ID_COLUMN] == chosen_coin]
        test_df = test_df[test_df[CFG.COIN_ID_COLUMN] == chosen_coin]
    return (train_df, test_df) if split else df
