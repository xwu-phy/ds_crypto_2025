import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pytz
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta

def load_ohlcv(root_path: str, coin: str = None) -> pd.DataFrame:
    """
    Load all parquet files under root_path (organized by coin_id and year),
    add a 'coin_id' column, parse timestamps (UTC), sort, and optionally filter.
    """
    coin_pattern = f"coin_id={coin}" if coin else "coin_id=*"
    pattern      = os.path.join(root_path, coin_pattern, "year=*", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files: raise FileNotFoundError(f"No files found with pattern: {pattern}")
    
    parts = []
    for fp in files:
        coin_seg     = next(seg for seg in fp.split(os.sep) if seg.startswith("coin_id="))
        this_coin    = coin_seg.split("=", 1)[1]
        df_piece     = pd.read_parquet(fp)
        df_piece['coin_id'] = this_coin
        parts.append(df_piece)
        
    df = pd.concat(parts, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    df.sort_values(['coin_id','timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if coin:
        df = df[df['coin_id'] == coin].reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from {len(files)} files ({coin or 'all coins'}).")
    return df

def create_advanced_features(df):
    """
    Creates a rich set of features including advanced technical indicators.
    """
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()

    # --- Basic Momentum & Volatility ---
    df['lag_ret_1'] = df['return'].shift(1)
    df['lag_ret_2'] = df['return'].shift(2)
    df['lag_ret_3'] = df['return'].shift(3)
    df['vol_3']     = df['return'].rolling(window=3).std()
    df['vol_24']    = df['return'].rolling(window=24).std()

    # --- Trend Features ---
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ma_ratio'] = df['sma_7'] / df['sma_30']
    
    # --- Advanced Technical Indicators ---
    df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    df.ta.atr(high='high', low='low', close='close', length=14, append=True)

    # --- Time-Based Features ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Shift all price-derived features to ensure no leakage
    feature_cols = [
        'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'vol_3', 'vol_24', 'ma_ratio',
        'MACDh_12_26_9', 'ATRr_14'
    ]
    for col in feature_cols:
        df[col] = df[col].shift(1)
            
    return df

# Dummy data for demonstration
if not os.path.exists("OHLCV_ffill.parquet/coin_id=BTCUSDT/year=2024"):
    os.makedirs("OHLCV_ffill.parquet/coin_id=BTCUSDT/year=2024")
    os.makedirs("OHLCV_ffill.parquet/coin_id=BTCUSDT/year=2025")
    dummy_df_2024 = pd.DataFrame({'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', end='2024-12-31 23:59', freq='min')), 'close': 60000 + np.random.randn(527040).cumsum(), 'high': 60000 + np.random.randn(527040).cumsum() + 100, 'low': 60000 + np.random.randn(527040).cumsum() - 100, 'open': 1, 'volume': 1})
    dummy_df_2025 = pd.DataFrame({'timestamp': pd.to_datetime(pd.date_range(start='2025-01-01', end='2025-03-31 23:59', freq='min')), 'close': 80000 + np.random.randn(129600).cumsum(), 'high': 80000 + np.random.randn(129600).cumsum() + 100, 'low': 80000 + np.random.randn(129600).cumsum() - 100, 'open': 1, 'volume': 1})
    dummy_df_2024.to_parquet("OHLCV_ffill.parquet/coin_id=BTCUSDT/year=2024/data.parquet")
    dummy_df_2025.to_parquet("OHLCV_ffill.parquet/coin_id=BTCUSDT/year=2025/data.parquet")

# 1. Load Data
df_min = load_ohlcv("OHLCV_ffill.parquet", coin="BTCUSDT")
df_min.set_index('timestamp', inplace=True)

# --- KEY CHANGE HERE ---
# 2. Resample to 10-minute bars
df_10m = (
    df_min[['open','high','low','close','volume']]
    .resample('10T') # Changed from '30T' to '10T'
    .agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'})
    .dropna()
)

# 3. Create Features (on the new 10-minute DataFrame)
df_featured = create_advanced_features(df_10m.copy())
df_featured['target'] = (df_featured['return'].shift(-1) > 0).astype(int)
df_featured.dropna(inplace=True)

# 4. Train / Validation / Test split (dates remain the same)
utc = pytz.UTC
val_start  = utc.localize(datetime(2024,12,1,0,0))
val_end    = utc.localize(datetime(2024,12,31,23,59))
test_start = utc.localize(datetime(2025,1,1,0,0))
test_end   = utc.localize(datetime(2025,3,30,23,59))

train = df_featured[df_featured.index < val_start]
val   = df_featured[(df_featured.index >= val_start) & (df_featured.index <= val_end)]
test  = df_featured[(df_featured.index >= test_start) & (df_featured.index <= test_end)]

if train.empty or test.empty: raise ValueError("Train or test split is empty.")

features = [
    'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'vol_3', 'vol_24', 'ma_ratio',
    'hour', 'day_of_week', 'MACDh_12_26_9', 'ATRr_14'
]
X_train, y_train = train[features], train['target']
X_test,  y_test  =  test[features],  test['target']

print("Shapes — Train:", X_train.shape, "Test:", X_test.shape)

# 5. Hyperparameter Tuning
param_grid = {'learning_rate': [0.01, 0.05], 'num_leaves': [20, 31, 40], 'n_estimators': [100, 200]}
neg_count = y_train.value_counts().get(0, 0)
pos_count = y_train.value_counts().get(1, 0)
scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1
print(f"\nUsing scale_pos_weight: {scale_pos_weight_value:.2f}")

base_clf = lgb.LGBMClassifier(objective='binary', random_state=42, scale_pos_weight=scale_pos_weight_value)
grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)

print("Starting hyperparameter tuning for 10-minute target...")
grid_search.fit(X_train, y_train)

print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)
best_clf = grid_search.best_estimator_

# 6. Evaluate on Test Set
y_pred = best_clf.predict(X_test)
probs  = best_clf.predict_proba(X_test)[:,1]

print("\n--- Classification Report (Tuned 10min Model) ---")
print(classification_report(y_test, y_pred))

# 7. Visualization Suite
plt.figure(figsize=(8, 6))
lgb.plot_importance(best_clf, importance_type='gain', max_num_features=10, title='Feature Importance (Gain)')
plt.tight_layout(), 
plt.savefig("feature.pdf")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down','Up'], yticklabels=['Down','Up'], cbar=False)
plt.xlabel('Predicted'), plt.ylabel('Actual'), plt.title('Confusion Matrix')
plt.tight_layout(), 
plt.savefig("confusion.pdf")

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc     = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
plt.title('ROC Curve'), plt.legend(loc='lower right')
plt.tight_layout(), 
plt.savefig("roc.pdf")

pnl_df = pd.DataFrame(index=X_test.index)
pnl_df['bh_return'] = df_featured['return'].loc[X_test.index]
pnl_df['signal'] = y_pred * 2 - 1
pnl_df['strat_return'] = pnl_df['signal'] * pnl_df['bh_return'].shift(-1)
pnl_df['cum_bh'] = pnl_df['bh_return'].cumsum()
pnl_df['cum_strat'] = pnl_df['strat_return'].cumsum()

plt.figure(figsize=(10,5))
plt.plot(pnl_df['cum_strat'], label='Strategy P&L')
plt.plot(pnl_df['cum_bh'], label='Buy & Hold (Standard)')
plt.xlabel('Date'), plt.ylabel('Cumulative Return')
plt.title('Cumulative P&L: Strategy vs Buy & Hold (10min Target)')
plt.legend(), plt.tight_layout(), 
plt.savefig("return.pdf")
