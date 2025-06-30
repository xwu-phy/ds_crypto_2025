# Configuration Management

This directory contains YAML configuration files for the cryptocurrency price prediction project.

## Configuration Files

- **`base_config.yaml`**: Common settings (data paths, splitting, features)
- **`xgboost_config.yaml`**: XGBoost hyperparameter optimization settings
- **`lstm_config.yaml`**: LSTM model configuration and optimization settings

## Configuration System

The system loads and merges configurations in this order:
1. Base configuration (common settings)
2. Model-specific configuration 
3. Environment-specific settings (development vs production)

## Usage Examples

### Loading Configuration

```python
from src.config import get_model_config, get_base_config

# Load base configuration
base_config = get_base_config()

# Load model-specific config
config = get_model_config("xgboost", "BTCUSDT", "1H", environment="development")

# Load for different environment
lstm_config = get_model_config("lstm", "ETHUSDT", "4H", environment="production")
```

### Accessing Configuration Values

```python
# Hyperparameter optimization settings
iterations = config['hyperparameter_optimization']['iterations']
search_space = config['hyperparameter_optimization']['search_space']

# File paths (automatically resolved for coin/granularity)
model_path = config['artifacts']['model_file']
features_path = config['artifacts']['selected_features_file']

# Data splitting and feature settings
train_ratio = config['splitting']['train_ratio']
excluded_features = config['features']['exclude_features']
```

### Environment Settings

```python
# Development environment (faster iteration)
dev_config = get_model_config("xgboost", "BTCUSDT", "1H", environment="development")
dev_iterations = dev_config['hyperparameter_optimization']['iterations']  # 50

# Production environment (comprehensive search)
prod_config = get_model_config("xgboost", "BTCUSDT", "1H", environment="production")
prod_iterations = prod_config['hyperparameter_optimization']['iterations']  # 1000
```

### Loading Saved Parameters

```python
# Load configuration with saved hyperparameters from trained models
config = get_model_config("xgboost", "BTCUSDT", "1H", load_saved_params=True)

if 'saved_hyperparameters' in config:
    best_params = config['saved_hyperparameters']
    print(f"Best n_estimators: {best_params['n_estimators']}")
```