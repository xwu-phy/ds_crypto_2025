# erdos_src/config.py
class CFG:
    # --- Data Paths ---
    RAW_DATA_PATH = "/project/littlewood/Xiaoyuan/DS/erdos-proj/data/OHLCV.parquet"
    FF_DATA_PATH = "/project/littlewood/Xiaoyuan/DS/erdos-proj/data/OHLCV_ffill.parquet"
    TUNING_RESULTS_PATH = '/project/littlewood/Xiaoyuan/DS/erdos-proj/models/'
    #FINAL_MODEL_PATH = './final_xgb_model.json'

    # --- Data & Feature Parameters ---
    COIN_ID_COLUMN = 'coin_id'
    TIMESTAMP_COLUMN = 'timestamp'
    GRANULARITY = '1h'                                                         # 4-hours for 1 time step
    TARGET_COLUMN = 'target_direction'
    PREDICTION_HORIZON_STEPS = 1                                           # We are predicting 1 step into the future
    CHOSEN_COIN = 'BTCUSDT'
    FEATURES = [
        "open", "high", "low", "close", "volume", 
        "quote_asset_volume", "number_of_trades",
        "taker_buy_quote_asset_volume",
        "sma_30", "ema_30", "macd", "rsi", "roc_10", 
        "bb_upper", "bb_lower",
        "atr_14",
        'taker_buy_sell_ratio'
    ]
    ROLLING_FIXED_WINDOW_SIZE = 0.5                                                 # x% of data used for rolling validation with fixed window size: ratio * data size
    ROLLING_INITIAL_WINDOW_SIZE = 0.3                                       # X% of data used in initial training in rolling validation with expanding window

    # --- Splitting & CV Parameters ---
    TRAIN_RATIO = 0.9                                                       # Use 80% for dev, 20% for final test
    SPLIT_ROUND_FREQUENCY = 'none'                                         # "month", "day", "":  Rounded frequency for split cut
    CV_SPLITS = 5                                                           # Number of folds for TimeSeriesSplit
    
    # --- Hyperparameter Tuning Parameters ---
    TUNING_ITERATIONS = 500                                                 # Number of random combinations to try
