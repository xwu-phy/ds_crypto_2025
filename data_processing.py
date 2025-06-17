# erdos_src/data_processing.py
import pandas as pd
import numpy as np
import os

class CFG:
    # --- Data & Feature Parameters ---
    COIN_ID_COLUMNUMN = 'coin_id'
    TIMESTAMP_COLUMN = 'timestamp'
    TARGET_COLUMN = 'target_direction'
    PREDICTION_HORIZON_MINS = 10                                            # We are predicting 10 minutes into the future
    CHOSEN_COIN = 'BTCUSDT'
    # FEATURES = [
    #     "open", "high", "low", "close", "volume", 
    #     "quote_asset_volume", "number_of_trades",
    #     "taker_buy_quote_asset_volume",
    #     "sma_30", "ema_30", "macd", "rsi", "roc_10", 
    #     "bb_upper", "bb_lower",
    #     "atr_14",
    #     'taker_buy_sell_ratio'
    # ]
    
    # --- Splitting & CV Parameters ---
    TRAIN_RATIO = 0.8                                                       # Use 80% for dev, 20% for final test
    SPLIT_ROUND_FREQUENCY = 'month'                                         # "month", "day", "":  Rounded frequency for split cut
    CV_SPLITS = 5                                                           # Number of folds for TimeSeriesSplit


def load_data(path, chosen_coin = None):
    '''
    Load data from a parquet file and preprocess it.
    Args:
        path (str): Path to the parquet file.
        chosen_coin (str, optional): Specific coin ID to filter data. Defaults to None.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with coin data.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df[CFG.TIMESTAMP_COLUMN]):
        print("Converting timestamp column to datetime format.")
        df[CFG.TIMESTAMP_COLUMN] = pd.to_datetime(df[CFG.TIMESTAMP_COLUMN])

    print(f"Ensure the dataframe is sorted by ({CFG.COIN_ID_COLUMN}, {CFG.TIMESTAMP_COLUMN}).")
    df.sort_values(by=[CFG.COIN_ID_COLUMN, CFG.TIMESTAMP_COLUMN], inplace=True)
    if chosen_coin is not None:
        print(f"Isolating data for coin: {chosen_coin}")
        df = df[df[CFG.COIN_ID_COLUMN] == chosen_coin]
    else:
        print("No specific coin_id provided, loading all data.")
    print(f"Data loaded with shape: {df.shape}")
    return df

def split_data(df: pd.DataFrame):
    """
    Splits the data into training and testing sets, rounding the split point
    to the beginning of the specified frequency ('month' or 'day').
    Args:
        df (pd.DataFrame): The DataFrame containing the data to be split.
    Returns:
        tuple: A tuple containing the training DataFrame and the testing DataFrame.
    """
    print(f"Splitting data with a {CFG.TRAIN_RATIO:.0%} train ratio, rounding to the start of the {CFG.SPLIT_ROUND_FREQUENCY}...")

    # --- Calculate the exact cutoff timestamp ---
    min_ts = df[CFG.TIMESTAMP_COLUMN].min()
    max_ts = df[CFG.TIMESTAMP_COLUMN].max()
    total_duration = max_ts - min_ts
    exact_cutoff_ts = min_ts + (total_duration * CFG.TRAIN_RATIO)

    print(f"Exact calculated split point: {exact_cutoff_ts}")

    # --- Round the cutoff timestamp down to the start of the specified frequency ---
    if CFG.SPLIT_ROUND_FREQUENCY == 'month':
        # 'M' stands for Month End, so we get the period and convert back to the start of that period
        rounded_cutoff_ts = exact_cutoff_ts.to_period('M').to_timestamp()
    elif CFG.SPLIT_ROUND_FREQUENCY == 'day':
        # 'D' stands for Day, this normalizes the time to midnight
        rounded_cutoff_ts = exact_cutoff_ts.normalize()
    else:
        # If no rounding is specified, use the exact point
        rounded_cutoff_ts = exact_cutoff_ts
    print(f"Split point rounded to: {rounded_cutoff_ts}")

    train_df = df[df[CFG.TIMESTAMP_COLUMN] < rounded_cutoff_ts].copy()
    test_df = df[df[CFG.TIMESTAMP_COLUMN] >= rounded_cutoff_ts].copy()

    print(f"Training data from {train_df[CFG.TIMESTAMP_COLUMN].min()} to {train_df[CFG.TIMESTAMP_COLUMN].max()}")
    print(f"Test data from {test_df[CFG.TIMESTAMP_COLUMN].min()} to {test_df[CFG.TIMESTAMP_COLUMN].max()}")
    return train_df, test_df
