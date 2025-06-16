# Scripts

This folder contains utility scripts. Below is a description of the main scripts and their functionality.

---

## `CryptoDataCleaner.py`

A modular class for remove duplicate and fill missing rows in raw OHLCV data:

1. **Removing duplicate rows based on (coin_id, timestamp)**
2. **Filling missing timestamps** for each coin at fixed 1-minute intervals
3. **Forward-filling missing OHLCV values** assume zero trade (volume) and static price (all prices = close price from last known data point), as described in details below.
4. **Validating data integrity** (e.g. checking for NaNs, negative values, or logical OHLC issues)

Data generated using CryptoDataCleaner.py is saved in [output.zip](https://drive.google.com/file/d/17llEK20pb0Q4pmDU2Gg1H4ruWJFFyAHg/view?usp=sharing).

- **Forward Fill Steps**

  When filling in missing timestamps, the script first creates a complete `(coin_id, timestamp)` grid using `pd.date_range()` and `pd.MultiIndex.from_product()`. This ensures that every expected minute for every coin is present in the data.
  
  For forward-filling missing OHLCV values:
  
  - **Price columns (`open`, `high`, `low`, `close`)** are forward-filled using the last known value for each coin (`groupby('coin_id').ffill()`).
  - **Volume-related columns** (`volume`, `quote_asset_volume`, etc.) are assumed to be 0 during missing periods (i.e. no trading activity occurred).
  - A `missing_flag` column is also added to indicate which rows were originally missing (`1`) vs. originally present (`0`).
  - The `close_time` is recalculated as `timestamp + 59.999s` to align with the expected 1-minute candle interval.

- **Example Usage**

  ```python
  from CryptoDataCleaner import CryptoDataCleaner
  import pandas as pd
  import json, os
  
  DATA_PATH = '...'                               # Path to raw data
  OUTPUT_DIR = '...'                              # Directory to save cleaned data
  df = pd.read_parquet(DATA_PATH, engine="pyarrow")
  cleaner = CryptoDataCleaner(df)
  cleaner.remove_duplicates()                     # remove duplicates
  cleaner.fill_missing_timestamps(freq='1min')    # fill missing timestamps
  cleaner.forward_fill_data()                     # forward fill missing data
  
  cleaned_df = cleaner.get_cleaned_data()
  output_file_name = OUTPUT_DIR + "OHLCV_ffill.parquet"
  cleaned_df.to_parquet(
      output_file_name,
      partition_cols=["coin_id", "year"],
      engine="pyarrow",
      index=False  # timestamp is now a column
  )
  ```
  ---
  
