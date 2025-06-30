import pandas as pd
import os
import json

OUTPUT_DIR = "data/"
DATA_PATH = "data/OHLCV.parquet"


class CryptoDataCleaner:
    """
    A comprehensive data cleaning module for cryptocurrency OHLCV data.
    
    This class provides methods to:
    1. Remove duplicate rows
    2. Fill missing timestamps with appropriate OHLCV values
    3. Validate data integrity
    """
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset=None, keep='first'):
        if subset is None:
            subset = ['timestamp', 'coin_id']
            
        initial_rows = len(self.df)
        duplicates = self.df.duplicated(subset=subset, keep=False)
        num_duplicate_rows = duplicates.sum()
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        final_rows = len(self.df)
        removed_count = initial_rows - final_rows
        
        print(f"Duplicate removal complete:")
        print(f"  - Total duplicate rows found: {num_duplicate_rows}")
        print(f"  - Rows removed: {removed_count}")
        print(f"  - Rows remaining: {final_rows}")
        
        return self.df, removed_count
    
    def fill_missing_timestamps(self, start_time=None, end_time=None, freq='1min'):
        if start_time is None:
            start_time = self.df['timestamp'].min()
        if end_time is None:
            end_time = self.df['timestamp'].max()
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        coin_ids = self.df['coin_id'].unique()

        # generate full grid
        full_time = pd.date_range(start=start_time, end=end_time, freq=freq)
        full_grid = pd.MultiIndex.from_product([coin_ids, full_time], names=['coin_id', 'timestamp']).to_frame(index=False)

        # merge with original data
        subset = ['timestamp', 'coin_id']
        df_full = pd.merge(full_grid, self.df, on=subset, how='left')

        # sort by coin and timestamp for proper forward filling; fill year column
        df_full = df_full.sort_values(by=['coin_id', 'timestamp'])
        df_full['year'] = df_full['timestamp'].dt.year

        print(f'Filled missing timestamps from {start_time} to {end_time} with frequency {freq}.')
        print(f"  Original rows: {len(self.df)} vs. expected full rows: {len(full_grid)}")
        print(f"  Final rows: {len(df_full)}")
        print(f"  Missing rows filled with NaN: {df_full['open'].isna().sum()}; equal diff: {len(df_full) - len(self.df)}")

        self.df = df_full
        return df_full


    def forward_fill_data(self):
        self.df['missing_flag'] = self.df['close'].isna().astype(int)
        self.df['close'] = self.df.groupby('coin_id', observed=True)['close'].ffill()
        self.df['open'].fillna(self.df['close'], inplace=True)
        self.df['high'].fillna(self.df['close'], inplace=True)
        self.df['low'].fillna(self.df['close'], inplace=True)
        
        volume_columns = ['volume', 'quote_asset_volume',
                          'number_of_trades',
                          'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume']
        for col in volume_columns:
            self.df[col].fillna(0, inplace=True)
        
        self.df['close_time'] = self.df['timestamp'] + pd.Timedelta(seconds=59.999)
        
        # Ensure proper data types
        self.df = self._ensure_proper_dtypes(self.df)
        return self.df
    
    def _ensure_proper_dtypes(self, df):
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            'quote_asset_volume', 'taker_buy_base_asset_volume', 
            'taker_buy_quote_asset_volume'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'number_of_trades' in df.columns:
            df['number_of_trades'] = df['number_of_trades'].fillna(0).astype(int)
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)    
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])
        return df
    
    def validate_data_integrity(self):
        validation_results = {}
        
        # Check for remaining NaN values
        nan_counts = self.df.isnull().sum()
        validation_results['nan_counts'] = nan_counts[nan_counts > 0].to_dict()
        
        # Check for negative prices or volumes
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        negative_counts = {}
        for col in numeric_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    negative_counts[col] = negative_count
        validation_results['negative_values'] = negative_counts
        
        # Check OHLC relationships (High >= Low, etc.)
        ohlc_issues = {}
        if all(col in self.df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Low
            high_low_issues = (self.df['high'] < self.df['low']).sum()
            if high_low_issues > 0:
                ohlc_issues['high_less_than_low'] = high_low_issues
                
            # High should be >= Open and Close
            high_open_issues = (self.df['high'] < self.df['open']).sum()
            high_close_issues = (self.df['high'] < self.df['close']).sum()
            if high_open_issues > 0:
                ohlc_issues['high_less_than_open'] = high_open_issues
            if high_close_issues > 0:
                ohlc_issues['high_less_than_close'] = high_close_issues
                
            # Low should be <= Open and Close  
            low_open_issues = (self.df['low'] > self.df['open']).sum()
            low_close_issues = (self.df['low'] > self.df['close']).sum()
            if low_open_issues > 0:
                ohlc_issues['low_greater_than_open'] = low_open_issues
            if low_close_issues > 0:
                ohlc_issues['low_greater_than_close'] = low_close_issues
                
        validation_results['ohlc_issues'] = ohlc_issues
        
        # Check for duplicate timestamps per coin
        duplicate_timestamps = self.df.groupby('coin_id')['timestamp'].nunique()
        expected_timestamps = self.df.groupby('coin_id').size()
        timestamp_issues = {}
        for coin in duplicate_timestamps.index:
            if duplicate_timestamps[coin] != expected_timestamps[coin]:
                timestamp_issues[coin] = {
                    'unique_timestamps': duplicate_timestamps[coin],
                    'total_rows': expected_timestamps[coin]
                }
        validation_results['timestamp_issues'] = timestamp_issues
                
        if validation_results['nan_counts']:
            print("Remaining NaN values found:")
            for col, count in validation_results['nan_counts'].items():
                print(f"   - {col}: {count}")
        else:
            print("No NaN values found")
            
        if validation_results['negative_values']:
            print("Negative values found:")
            for col, count in validation_results['negative_values'].items():
                print(f"   - {col}: {count}")
        else:
            print("No negative values in price/volume columns")
            
        if validation_results['ohlc_issues']:
            print("OHLC relationship issues found:")
            for issue, count in validation_results['ohlc_issues'].items():
                print(f"   - {issue}: {count}")
        else:
            print("OHLC relationships are valid")
            
        if validation_results['timestamp_issues']:
            print("Timestamp issues found:")
            for coin, issues in validation_results['timestamp_issues'].items():
                print(f"   - {coin}: {issues}")
        else:
            print("No timestamp duplication issues")
            
        return validation_results
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary_stats(self):
        current_shape = self.df.shape
        
        summary = {
            'original_shape': self.original_shape,
            'final_shape': current_shape,
            'rows_added': current_shape[0] - self.original_shape[0],
            'coins': self.df['coin_id'].nunique(),
            'date_range': {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max()
            },
            'total_trading_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        }
        
        return summary



def clean_crypto_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Output directory {DATA_PATH} does not exist.")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")

    print("Starting crypto data cleaning process...")
    cleaner = CryptoDataCleaner(df)
    print("\n1. Removing duplicates...")
    cleaner.remove_duplicates()
    print("\n2. Filling missing timestamps...")
    cleaner.fill_missing_timestamps(freq='1min')
    print("\n3. Forward filling missing data...")
    cleaner.forward_fill_data()
    cleaned_df = cleaner.get_cleaned_data()
    
    # save results
    summary_stats_path = os.path.join(OUTPUT_DIR, "ffill_summary_stats.json")
    with open(summary_stats_path, 'w') as f:
        json.dump(cleaner.get_summary_stats(), f, indent=4, default=str)
    validation_path = os.path.join(OUTPUT_DIR, "ffill_validation_report.json")
    with open(validation_path, 'w') as f:
        json.dump(cleaner.validate_data_integrity(), f, indent=4, default=str)
    
    output_file_name = OUTPUT_DIR + "OHLCV_ffill.parquet"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True) 
    cleaned_df.to_parquet(
        output_file_name,
        partition_cols=["coin_id", "year"],
        engine="pyarrow", 
        index=False 
    )
    print(f"Cleaned data saved to {output_file_name} | Summary stats in {summary_stats_path} | Validation report in {validation_path}")
    return

if __name__ == "__main__":
    clean_crypto_data()
    print("Data cleaning completed successfully.")
