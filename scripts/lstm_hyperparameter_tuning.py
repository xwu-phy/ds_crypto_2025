"""
Hyperparameter tuning for LSTM cryptocurrency prediction model.

This module provides systematic approaches to find optimal hyperparameters:
1. Random Search
2. Bayesian Optimization (Optuna)
3. Grid Search (for smaller parameter spaces)
4. Time Series Cross-Validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf

# Optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Project imports
from erdos_src.model_lstm import LSTMCryptoPredictor
from erdos_src.config import CFG
from erdos_src.data_processing import get_features_and_target

logger = logging.getLogger(__name__)

class TimeSeriesSplit:
    """Simple time series cross-validation splitter."""
    
    def __init__(self, n_splits: int = 3, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, df: pd.DataFrame):
        """Generate time series splits."""
        n_samples = len(df)
        test_size_samples = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Calculate split points
            split_point = int(n_samples * (i + 1) / (self.n_splits + 1))
            train_end = split_point
            test_start = train_end
            test_end = min(test_start + test_size_samples, n_samples)
            
            # Skip if not enough samples
            if train_end < 100 or test_end - test_start < 50:
                continue
                
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, test_end))
            
            yield train_idx, test_idx

class LSTMHyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for LSTM models.
    """
    
    def __init__(self, 
                 chosen_coin: str = CFG.CHOSEN_COIN,
                 n_splits: int = 3,
                 test_size_ratio: float = 0.2,
                 optimization_metric: str = 'f1_score',
                 results_save_path: Optional[str] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            chosen_coin: Cryptocurrency to tune for
            n_splits: Number of time series cross-validation splits
            test_size_ratio: Ratio of each split to use as test
            optimization_metric: Metric to optimize ('accuracy', 'f1_score', 'roc_auc')
            results_save_path: Path to save tuning results
        """
        self.chosen_coin = chosen_coin
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.optimization_metric = optimization_metric
        self.results_save_path = results_save_path or f"{CFG.TUNING_RESULTS_PATH}lstm_tuning_results_{chosen_coin}.json"
        
        # Load data once for all experiments
        logger.info(f"Loading data for {chosen_coin}...")
        self.train_df, self.test_df = get_features_and_target(path=CFG.FF_DATA_PATH, chosen_coin=chosen_coin)
        
        # Store results
        self.tuning_results = []
        self.best_score = -np.inf
        self.best_params = None
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary defining parameter ranges for optimization
        """
        return {
            'sequence_length': [30, 60, 120, 180],
            'lstm_units': [
                (50,), (100,), (150,),  # Single layer
                (100, 50), (150, 75), (200, 100),  # Two layers
                (150, 100, 50), (200, 150, 75)  # Three layers
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
            'batch_size': [16, 32, 64, 128],
            'epochs': [50, 100, 150],
            'early_stopping_patience': [10, 15, 20],
            'scaler_type': ['standard', 'robust']
        }
    
    def time_series_cross_validate(self, params: Dict[str, Any]) -> float:
        """
        Perform time series cross-validation for given parameters.
        
        Args:
            params: Hyperparameters to evaluate
            
        Returns:
            Average cross-validation score
        """
        try:
            # Create time series splits
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size_ratio)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(self.train_df)):
                logger.info(f"Evaluating fold {fold + 1}/{self.n_splits}")
                
                # Split data
                fold_train = self.train_df.iloc[train_idx]
                fold_val = self.train_df.iloc[val_idx]
                
                # Initialize predictor with current parameters
                predictor = LSTMCryptoPredictor(**params)
                
                # Prepare data
                X_train, y_train, X_val, y_val = predictor.prepare_data(
                    fold_train, fold_val, self.chosen_coin
                )
                
                # Skip if not enough data for sequences
                if len(X_train) < 100 or len(X_val) < 50:
                    logger.warning(f"Insufficient data for fold {fold + 1}, skipping...")
                    continue
                
                # Build and train model
                input_shape = (X_train.shape[1], X_train.shape[2])
                predictor.build_model(input_shape)
                
                # Train with reduced verbosity
                predictor.model.fit(
                    X_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=(X_val, y_val),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', 
                            patience=params['early_stopping_patience'],
                            restore_best_weights=True,
                            verbose=0
                        )
                    ],
                    verbose=0
                )
                
                # Evaluate
                y_pred_proba = predictor.model.predict(X_val, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).ravel()
                
                # Calculate score based on optimization metric
                if self.optimization_metric == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif self.optimization_metric == 'f1_score':
                    score = f1_score(y_val, y_pred)
                elif self.optimization_metric == 'roc_auc':
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    raise ValueError(f"Unknown metric: {self.optimization_metric}")
                
                scores.append(score)
                logger.info(f"Fold {fold + 1} {self.optimization_metric}: {score:.4f}")
            
            if not scores:
                return -np.inf
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"CV {self.optimization_metric}: {avg_score:.4f} ± {std_score:.4f}")
            return avg_score
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return -np.inf
    
    def random_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter optimization.
        
        Args:
            n_trials: Number of random trials to perform
            
        Returns:
            Best parameters found
        """
        logger.info(f"Starting random search with {n_trials} trials...")
        
        param_space = self.get_parameter_space()
        
        for trial in range(n_trials):
            logger.info(f"\n=== Random Search Trial {trial + 1}/{n_trials} ===")
            
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = np.random.choice(param_values)
            
            logger.info(f"Testing parameters: {params}")
            
            # Evaluate parameters
            start_time = time.time()
            score = self.time_series_cross_validate(params)
            elapsed_time = time.time() - start_time
            
            # Store results
            result = {
                'trial': trial + 1,
                'params': params,
                'score': score,
                'metric': self.optimization_metric,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
            self.tuning_results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"🎉 New best score: {score:.4f}")
            
            # Save intermediate results
            self._save_results()
        
        logger.info(f"Random search completed. Best {self.optimization_metric}: {self.best_score:.4f}")
        return self.best_params
    
    def bayesian_optimization(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
        
        logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            # Define parameter suggestions
            params = {
                'sequence_length': trial.suggest_categorical('sequence_length', [30, 60, 120, 180]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 5e-3),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'epochs': trial.suggest_categorical('epochs', [50, 100, 150]),
                'early_stopping_patience': trial.suggest_categorical('early_stopping_patience', [10, 15, 20]),
                'scaler_type': trial.suggest_categorical('scaler_type', ['standard', 'robust'])
            }
            
            # Suggest LSTM architecture
            n_layers = trial.suggest_int('n_layers', 1, 3)
            lstm_units = []
            for i in range(n_layers):
                units = trial.suggest_categorical(f'lstm_units_layer_{i}', [50, 100, 150, 200])
                lstm_units.append(units)
            params['lstm_units'] = tuple(lstm_units)
            
            logger.info(f"Trial {trial.number}: {params}")
            
            # Evaluate parameters
            score = self.time_series_cross_validate(params)
            
            # Store results for later analysis
            result = {
                'trial': trial.number,
                'params': params,
                'score': score,
                'metric': self.optimization_metric,
                'timestamp': datetime.now().isoformat()
            }
            self.tuning_results.append(result)
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Extract best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Bayesian optimization completed.")
        logger.info(f"Best {self.optimization_metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save detailed results
        self._save_results()
        
        return self.best_params
    
    def grid_search(self, param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform grid search (use for smaller parameter spaces).
        
        Args:
            param_grid: Custom parameter grid (uses subset of full space if None)
            
        Returns:
            Best parameters found
        """
        if param_grid is None:
            # Use a smaller subset for grid search
            param_grid = {
                'sequence_length': [60, 120],
                'lstm_units': [(100, 50), (150, 75)],
                'dropout_rate': [0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64],
                'epochs': [100],
                'early_stopping_patience': [15],
                'scaler_type': ['standard']
            }
        
        logger.info("Starting grid search...")
        logger.info(f"Parameter grid: {param_grid}")
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        logger.info(f"Total combinations to test: {total_combinations}")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"\n=== Grid Search {i + 1}/{total_combinations} ===")
            logger.info(f"Testing parameters: {params}")
            
            # Evaluate parameters
            start_time = time.time()
            score = self.time_series_cross_validate(params)
            elapsed_time = time.time() - start_time
            
            # Store results
            result = {
                'trial': i + 1,
                'params': params,
                'score': score,
                'metric': self.optimization_metric,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
            self.tuning_results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"🎉 New best score: {score:.4f}")
        
        logger.info(f"Grid search completed. Best {self.optimization_metric}: {self.best_score:.4f}")
        self._save_results()
        
        return self.best_params
    
    def _save_results(self):
        """Save tuning results to file."""
        try:
            results_data = {
                'coin': self.chosen_coin,
                'optimization_metric': self.optimization_metric,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'all_trials': self.tuning_results,
                'tuning_completed': datetime.now().isoformat()
            }
            
            with open(self.results_save_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.results_save_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze and visualize tuning results.
        
        Returns:
            DataFrame with analysis of parameter importance
        """
        if not self.tuning_results:
            logger.warning("No tuning results available for analysis")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df_results = pd.DataFrame([
            {**r['params'], 'score': r['score'], 'trial': r['trial']} 
            for r in self.tuning_results if r['score'] > -np.inf
        ])
        
        if df_results.empty:
            logger.warning("No valid results for analysis")
            return df_results
        
        logger.info(f"\n=== Hyperparameter Analysis ===")
        logger.info(f"Total valid trials: {len(df_results)}")
        logger.info(f"Best score: {df_results['score'].max():.4f}")
        logger.info(f"Mean score: {df_results['score'].mean():.4f}")
        logger.info(f"Std score: {df_results['score'].std():.4f}")
        
        # Analyze parameter importance
        numeric_params = ['sequence_length', 'dropout_rate', 'learning_rate', 'batch_size', 'epochs']
        
        logger.info(f"\n=== Parameter Correlations with {self.optimization_metric} ===")
        for param in numeric_params:
            if param in df_results.columns:
                corr = df_results[[param, 'score']].corr().iloc[0, 1]
                logger.info(f"{param}: {corr:.3f}")
        
        # Top performing parameter combinations
        top_results = df_results.nlargest(5, 'score')
        logger.info(f"\n=== Top 5 Parameter Combinations ===")
        for idx, row in top_results.iterrows():
            params_str = ', '.join([f"{k}: {v}" for k, v in row.items() 
                                  if k not in ['score', 'trial']])
            logger.info(f"Score: {row['score']:.4f} | {params_str}")
        
        return df_results


def quick_hyperparameter_search(chosen_coin: str = CFG.CHOSEN_COIN,
                               method: str = 'random',
                               n_trials: int = 20) -> Dict[str, Any]:
    """
    Quick hyperparameter search with reasonable defaults.
    
    Args:
        chosen_coin: Cryptocurrency to optimize for
        method: Search method ('random', 'bayesian', 'grid')
        n_trials: Number of trials
        
    Returns:
        Best parameters found
    """
    tuner = LSTMHyperparameterTuner(chosen_coin=chosen_coin)
    
    if method == 'random':
        return tuner.random_search(n_trials=n_trials)
    elif method == 'bayesian' and OPTUNA_AVAILABLE:
        return tuner.bayesian_optimization(n_trials=n_trials)
    elif method == 'grid':
        return tuner.grid_search()
    else:
        logger.warning(f"Method '{method}' not available, falling back to random search")
        return tuner.random_search(n_trials=n_trials)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    logger.info("Starting hyperparameter tuning example...")
    
    # Quick random search
    best_params = quick_hyperparameter_search(
        chosen_coin=CFG.CHOSEN_COIN,
        method='random',
        n_trials=10
    )
    
    print(f"\nBest parameters found: {best_params}") 