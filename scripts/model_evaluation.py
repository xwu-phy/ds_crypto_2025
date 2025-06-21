# erdos_src/model_evaluation.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add parallel processing imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
import multiprocessing as mp

from erdos_src.config import CFG
from erdos_src.data_processing import get_features_and_target, split_data

logger = logging.getLogger(__name__)

class TimeSeriesEvaluator:
    """
    Comprehensive time series model evaluation with multiple validation strategies.
    """
    
    def __init__(self, model, n_jobs: int = 1):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model with fit() and predict() methods
            n_jobs: Number of parallel jobs (1 for sequential, -1 for all cores)
        """
        self.model = model
        self.features = CFG.FEATURES
        self.target_col = CFG.TARGET_COLUMN
        self.results = {}
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        if not JOBLIB_AVAILABLE:
            self.n_jobs = 1
            logger.warning("Parallel processing disabled - joblib not available")
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate binary classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = np.nan
                metrics['log_loss'] = np.nan
  
        return metrics

    def _train_and_predict_window(self, model_params: Dict, X_train: np.ndarray, 
                                 y_train: np.ndarray, X_test: np.ndarray, 
                                 y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Train model and predict for a single window (for parallel processing)"""
        model_class = type(self.model)
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = getattr(model, 'predict_proba', lambda x: None)(X_test)
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
        return y_pred, y_test, y_pred_proba

    def fixed_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          cv_splits: int = CFG.CV_SPLITS) -> Dict[str, Any]:
        """
        Method 1: Fixed train-test split with cross-validation on training data.
        Args:
            train_df: Training data
            test_df: Test data
            cv_splits: Number of CV splits for training data
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Running fixed-origin train-test split evaluation...")
        
        # Prepare data
        X_train = train_df[self.features].values
        y_train = train_df[self.target_col].values
        X_test = test_df[self.features].values
        y_test = test_df[self.target_col].values
        
        # Cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            self.model.fit(X_fold_train, y_fold_train)
            # Predict
            y_pred = self.model.predict(X_fold_val)
            y_pred_proba = getattr(self.model, 'predict_proba', lambda x: None)(X_fold_val)
            if y_pred_proba is not None:
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(y_fold_val, y_pred, y_pred_proba)
            cv_scores.append(fold_metrics)
            logger.info(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy']:.4f}")
        
        # Train final model on full training data and evaluate on test set
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = getattr(self.model, 'predict_proba', lambda x: None)(X_test)
        if y_test_pred_proba is not None:
            y_test_pred_proba = y_test_pred_proba[:, 1] if y_test_pred_proba.shape[1] > 1 else y_test_pred_proba.flatten()
        
        test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        
        # Aggregate CV results
        cv_avg = {}
        for metric in cv_scores[0].keys():
            cv_avg[metric] = np.mean([fold[metric] for fold in cv_scores])
            cv_avg[f'{metric}_std'] = np.std([fold[metric] for fold in cv_scores])
        
        results = {
            'method': 'fixed_split',
            'cv_scores': cv_scores,
            'cv_average': cv_avg,
            'test_metrics': test_metrics,
            'test_predictions': y_test_pred,
            'test_probabilities': y_test_pred_proba
        }
        
        self.results['fixed_split'] = results

        logger.info(f"Fixed split result:" )
        logger.info(f"Train average CV score: {results['cv_average']};")
        logger.info(f"Test metrics: {results['test_metrics']};")
        return results
    
    def rolling_origin_fixed_window(self, df: pd.DataFrame, 
                                   train_size: int, test_size: int = CFG.PREDICTION_HORIZON_STEPS) -> Dict[str, Any]:
        """
        Method 2.1: Rolling origin with fixed window size.
        Args:
            df: Full dataset
            train_size: Size of training window
            test_size: Size of test window  
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Running rolling origin fixed window evaluation (train_size={train_size}, test_size={test_size})...")
        
        X = df[self.features].values
        y = df[self.target_col].values
        
        all_test_preds = []
        all_test_true = []
        all_test_probs = []
        
        start_idx = train_size
        end_idx = len(df) - test_size + 1
        
        for i in range(start_idx, end_idx):
            # Define windows
            train_start = i - train_size
            train_end = i
            test_start = i
            test_end = i + test_size
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train and predict
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = getattr(self.model, 'predict_proba', lambda x: None)(X_test)
            if y_pred_proba is not None:
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
            
            # Store predictions
            all_test_preds.extend(y_pred)
            all_test_true.extend(y_test)
            if y_pred_proba is not None:
                all_test_probs.extend(y_pred_proba)
        
        # Calculate metrics on all predictions at once
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        all_test_probs = np.array(all_test_probs) if all_test_probs else None
        overall_metrics = self.calculate_metrics(all_test_true, all_test_preds, all_test_probs)
        
        results = {
            'method': 'rolling_fixed_window',
            'train_span': train_size,
            'test_size': test_size,
            'n_folds': len(all_test_preds) // test_size,
            'overall_metrics': overall_metrics,
            'all_test_predictions': all_test_preds,
            'all_test_true': all_test_true,
            'all_test_probabilities': all_test_probs
        }
        
        self.results['rolling_fixed_window'] = results
        
        logger.info(f"Rolling fixed window result: {overall_metrics}")
        return results
    
    def rolling_origin_fixed_window_parallel(self, df: pd.DataFrame, 
                                           train_size: int, test_size: int = CFG.PREDICTION_HORIZON_STEPS) -> Dict[str, Any]:
        """
        Parallel version of rolling origin with fixed window size.
        Args:
            df: Full dataset
            train_size: Size of training window
            test_size: Size of test window  
        Returns:
            Dictionary with evaluation results
        """
        if not JOBLIB_AVAILABLE or self.n_jobs <= 1:
            logger.warning("Parallel processing not available, falling back to sequential")
            return self.rolling_origin_fixed_window(df, train_size, test_size)
        
        logger.info(f"Running parallel rolling origin fixed window evaluation (train_size={train_size}, test_size={test_size}, n_jobs={self.n_jobs})...")
        
        X = df[self.features].values
        y = df[self.target_col].values
        
        # Prepare all windows
        windows = []
        start_idx = train_size
        end_idx = len(df) - test_size + 1
        
        for i in range(start_idx, end_idx):
            train_start = i - train_size
            train_end = i
            test_start = i
            test_end = i + test_size
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            windows.append((X_train, y_train, X_test, y_test))
        
        # Parallel processing
        model_params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
        batch_size = 100
        
        all_test_preds = []
        all_test_true = []
        all_test_probs = []
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._train_and_predict_window)(
                    model_params, X_train, y_train, X_test, y_test
                ) for X_train, y_train, X_test, y_test in batch_windows
            )
            
            for y_pred, y_test, y_pred_proba in results:
                all_test_preds.extend(y_pred)
                all_test_true.extend(y_test)
                if y_pred_proba is not None:
                    all_test_probs.extend(y_pred_proba)
        
        # Calculate metrics using existing method
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        all_test_probs = np.array(all_test_probs) if all_test_probs else None
        overall_metrics = self.calculate_metrics(all_test_true, all_test_preds, all_test_probs)
        
        results = {
            'method': 'rolling_fixed_window_parallel',
            'train_span': train_size,
            'test_size': test_size,
            'n_folds': len(all_test_preds) // test_size,
            'n_jobs': self.n_jobs,
            'overall_metrics': overall_metrics,
            'all_test_predictions': all_test_preds,
            'all_test_true': all_test_true,
            'all_test_probabilities': all_test_probs
        }
        
        self.results['rolling_fixed_window_parallel'] = results
        logger.info(f"Parallel rolling fixed window result: {overall_metrics}")
        return results
    
    def rolling_origin_expanding_window(self, df: pd.DataFrame, 
                                       initial_train_size: int, test_size: int = CFG.PREDICTION_HORIZON_STEPS) -> Dict[str, Any]:
        """
        Method 2.2: Rolling origin with expanding window.
        
        Args:
            df: Full dataset
            initial_train_size: Initial size of training window
            test_size: Size of test window
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Running rolling origin expanding window evaluation (initial_train_size={initial_train_size}, test_size={test_size})...")
        
        X = df[self.features].values
        y = df[self.target_col].values
        
        all_test_preds = []
        all_test_true = []
        all_test_probs = []
        
        start_idx = initial_train_size
        end_idx = len(df) - test_size + 1
        
        for i in range(start_idx, end_idx):
            # Define windows (expanding train window)
            train_start = 0
            train_end = i
            test_start = i
            test_end = i + test_size
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train and predict
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = getattr(self.model, 'predict_proba', lambda x: None)(X_test)
            if y_pred_proba is not None:
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
            
            # Store predictions
            all_test_preds.extend(y_pred)
            all_test_true.extend(y_test)
            if y_pred_proba is not None:
                all_test_probs.extend(y_pred_proba)
        
        # Calculate metrics on all predictions at once
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        all_test_probs = np.array(all_test_probs) if all_test_probs else None
        
        overall_metrics = self.calculate_metrics(all_test_true, all_test_preds, all_test_probs)
        
        results = {
            'method': 'rolling_expanding_window',
            'initial_train_size': initial_train_size,
            'test_size': test_size,
            'n_folds': len(all_test_preds) // test_size,
            'overall_metrics': overall_metrics,
            'all_test_predictions': all_test_preds,
            'all_test_true': all_test_true,
            'all_test_probabilities': all_test_probs
        }
        
        self.results['rolling_expanding_window'] = results
        
        logger.info(f"Rolling expanding window result: {overall_metrics}")
        return results
    
    def rolling_origin_expanding_window_parallel(self, df: pd.DataFrame, 
                                               initial_train_size: int, test_size: int = CFG.PREDICTION_HORIZON_STEPS) -> Dict[str, Any]:
        """
        Parallel version of rolling origin with expanding window.
        
        Args:
            df: Full dataset
            initial_train_size: Initial size of training window
            test_size: Size of test window
            
        Returns:
            Dictionary with evaluation results
        """
        if not JOBLIB_AVAILABLE or self.n_jobs <= 1:
            logger.warning("Parallel processing not available, falling back to sequential")
            return self.rolling_origin_expanding_window(df, initial_train_size, test_size)
        
        logger.info(f"Running parallel rolling origin expanding window evaluation (initial_train_size={initial_train_size}, test_size={test_size}, n_jobs={self.n_jobs})...")
        
        X = df[self.features].values
        y = df[self.target_col].values
        
        # Prepare all windows
        windows = []
        start_idx = initial_train_size
        end_idx = len(df) - test_size + 1
        
        for i in range(start_idx, end_idx):
            train_start = 0
            train_end = i
            test_start = i
            test_end = i + test_size
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            windows.append((X_train, y_train, X_test, y_test))
        
        # Parallel processing
        model_params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
        batch_size = 50  # Smaller batch size for expanding windows
        
        all_test_preds = []
        all_test_true = []
        all_test_probs = []
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._train_and_predict_window)(
                    model_params, X_train, y_train, X_test, y_test
                ) for X_train, y_train, X_test, y_test in batch_windows
            )
            
            for y_pred, y_test, y_pred_proba in results:
                all_test_preds.extend(y_pred)
                all_test_true.extend(y_test)
                if y_pred_proba is not None:
                    all_test_probs.extend(y_pred_proba)
        
        # Calculate metrics using existing method
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        all_test_probs = np.array(all_test_probs) if all_test_probs else None
        
        overall_metrics = self.calculate_metrics(all_test_true, all_test_preds, all_test_probs)
        
        results = {
            'method': 'rolling_expanding_window_parallel',
            'initial_train_size': initial_train_size,
            'test_size': test_size,
            'n_folds': len(all_test_preds) // test_size,
            'n_jobs': self.n_jobs,
            'overall_metrics': overall_metrics,
            'all_test_predictions': all_test_preds,
            'all_test_true': all_test_true,
            'all_test_probabilities': all_test_probs
        }
        
        self.results['rolling_expanding_window_parallel'] = results
        logger.info(f"Parallel rolling expanding window result: {overall_metrics}")
        return results
    
    def evaluate_all_methods(self, df: pd.DataFrame, 
                           methods: Optional[List[str]] = None,
                           cv_splits: int = CFG.CV_SPLITS) -> Dict[str, Any]:
        """
        Run selected evaluation methods and return comprehensive results.
        
        Args:
            df: Full dataset
            methods: List of methods to run. Options: ['fixed', 'rolling_fixed_window', 'rolling_expanding_window']
                    If n_jobs > 1 and joblib available, automatically uses parallel versions
            cv_splits: Number of CV splits
        Returns:
            Dictionary with selected evaluation results
        """
        if methods is None:
            methods = ['fixed', 'rolling_fixed_window', 'rolling_expanding_window']
        
        logger.info(f"Running model evaluation with methods: {methods}")
        logger.info(f"Parallel processing: {'enabled' if JOBLIB_AVAILABLE and self.n_jobs > 1 else 'disabled'} (n_jobs={self.n_jobs})")
        
        results = {}

        # Method 1: Fixed-origin split
        if 'fixed' in methods:
            train_df, test_df = split_data(df = df, ts_col = CFG.TIMESTAMP_COLUMN, train_ratio = CFG.TRAIN_RATIO, round_frequency=CFG.SPLIT_ROUND_FREQUENCY)
            fixed_origin_results = self.fixed_split(train_df, test_df, cv_splits)
            results['fixed'] = fixed_origin_results
        
        # Method 2.1: Rolling origin fixed window (auto-select parallel if available)
        if 'rolling_fixed_window' in methods:
            fixed_window_size = int(len(df) * CFG.ROLLING_FIXED_WINDOW_SIZE)  
            
            if JOBLIB_AVAILABLE and self.n_jobs > 1:
                logger.info("Using parallel version of rolling_fixed_window")
                rolling_fixed_results = self.rolling_origin_fixed_window_parallel(
                    df = df, train_size = fixed_window_size, test_size = CFG.PREDICTION_HORIZON_STEPS
                )
                # Store with original name for consistency
                results['rolling_fixed_window'] = rolling_fixed_results
            else:
                logger.info("Using sequential version of rolling_fixed_window")
                rolling_fixed_results = self.rolling_origin_fixed_window(
                    df = df, train_size = fixed_window_size, test_size = CFG.PREDICTION_HORIZON_STEPS
                )
                results['rolling_fixed_window'] = rolling_fixed_results
        
        # Method 2.2: Rolling origin expanding window (auto-select parallel if available)
        if 'rolling_expanding_window' in methods:
            initial_train_size = int(len(df) * CFG.ROLLING_INITIAL_WINDOW_SIZE)  
            
            if JOBLIB_AVAILABLE and self.n_jobs > 1:
                logger.info("Using parallel version of rolling_expanding_window")
                rolling_expanding_results = self.rolling_origin_expanding_window_parallel(
                    df = df, initial_train_size = initial_train_size, test_size = CFG.PREDICTION_HORIZON_STEPS
                )
                # Store with original name for consistency
                results['rolling_expanding_window'] = rolling_expanding_results
            else:
                logger.info("Using sequential version of rolling_expanding_window")
                rolling_expanding_results = self.rolling_origin_expanding_window(
                    df = df, initial_train_size = initial_train_size, test_size = CFG.PREDICTION_HORIZON_STEPS
                )
                results['rolling_expanding_window'] = rolling_expanding_results

        # Create summary
        summary = {}
        
        if 'fixed' in results:
            summary['fixed'] = {
                'test_accuracy': results['fixed']['test_metrics']['accuracy'],
                'cv_accuracy_mean': results['fixed']['cv_average']['accuracy'],
                'cv_accuracy_std': results['fixed']['cv_average']['accuracy_std']
            }
        
        if 'rolling_fixed_window' in results:
            summary['rolling_fixed_window'] = {
                'accuracy_mean': results['rolling_fixed_window']['overall_metrics']['accuracy'],
                'accuracy_std': 0.0,  # No std since we calculate overall metrics
                'n_folds': results['rolling_fixed_window']['n_folds'],
                'n_jobs': results['rolling_fixed_window'].get('n_jobs', 1)  # Include n_jobs if available
            }
        
        if 'rolling_expanding_window' in results:
            summary['rolling_expanding_window'] = {
                'accuracy_mean': results['rolling_expanding_window']['overall_metrics']['accuracy'],
                'accuracy_std': 0.0,  # No std since we calculate overall metrics
                'n_folds': results['rolling_expanding_window']['n_folds'],
                'n_jobs': results['rolling_expanding_window'].get('n_jobs', 1)  # Include n_jobs if available
            }
        
        self.results = results
        self.results['summary'] = summary
        
        # Log summary
        logger.info("Evaluation completed. Summary:")
        for method, metrics in summary.items():
            if 'test_accuracy' in metrics:
                logger.info(f"{method} test accuracy: {metrics['test_accuracy']:.4f}")
            else:
                n_jobs_info = f" (n_jobs={metrics['n_jobs']})" if metrics.get('n_jobs', 1) > 1 else ""
                logger.info(f"{method} accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}{n_jobs_info}")
        
        return self.results
    
    def get_results_summary(self) -> pd.DataFrame:
        """Return results summary as a DataFrame."""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluation first.")
        
        summary_data = []
        
        for method_name, results in self.results.items():
            if method_name == 'summary':
                continue
                
            if method_name == 'fixed':
                summary_data.append({
                    'Method': 'Fixed Split',
                    'Accuracy': results['test_metrics']['accuracy'],
                    'Precision': results['test_metrics']['precision'],
                    'Recall': results['test_metrics']['recall'],
                    'F1': results['test_metrics']['f1'],
                    'ROC_AUC': results['test_metrics'].get('roc_auc', np.nan),
                    'CV_Accuracy_Mean': results['cv_average']['accuracy'],
                    'CV_Accuracy_Std': results['cv_average']['accuracy_std']
                })
            else:
                summary_data.append({
                    'Method': results['method'].replace('_', ' ').title(),
                    'Accuracy': results['overall_metrics']['accuracy'],
                    'Precision': results['overall_metrics']['precision'],
                    'Recall': results['overall_metrics']['recall'],
                    'F1': results['overall_metrics']['f1'],
                    'ROC_AUC': results['overall_metrics'].get('roc_auc', np.nan),
                    'CV_Accuracy_Mean': results['overall_metrics']['accuracy'],
                    'CV_Accuracy_Std': results['overall_metrics']['accuracy_std']
                })
        
        return pd.DataFrame(summary_data)



def evaluate_model(model, methods: Optional[List[str]] = None, n_jobs: int = 1) -> Dict[str, Any]:
    """
    Evaluate model using specified methods with optional parallel processing.
    
    Args:
        model: Trained model with fit() and predict() methods
        methods: List of evaluation methods to use
        n_jobs: Number of parallel jobs (1 for sequential, -1 for all cores)
    """
    # Initialize, load data
    evaluator = TimeSeriesEvaluator(model, n_jobs=n_jobs)
    df = get_features_and_target(CFG.FF_DATA_PATH, split = False, chosen_coin=CFG.CHOSEN_COIN)
    # Run evaluation
    results = evaluator.evaluate_all_methods(df, methods=methods)
    return results


if __name__ == "__main__":
    # Example usage with automatic parallel/sequential selection
    from sklearn.ensemble import RandomForestClassifier
    
    # Create model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Example 1: Automatic parallel processing (recommended for HPC)
    print("Running with automatic parallel processing...")
    results_parallel = evaluate_model(
        model, 
        methods=['rolling_fixed_window', 'rolling_expanding_window'],
        n_jobs=-1  # Use all available cores - will automatically use parallel versions
    )
    
    # Example 2: Sequential processing (original behavior)
    print("\nRunning with sequential processing...")
    results_sequential = evaluate_model(
        model,
        methods=['rolling_fixed_window', 'rolling_expanding_window'],
        n_jobs=1  # Sequential processing - will use original versions
    )
    
    # Example 3: Mixed methods
    print("\nRunning mixed methods...")
    results_mixed = evaluate_model(
        model,
        methods=['fixed', 'rolling_fixed_window'],  # fixed is always sequential, rolling_fixed_window will be parallel if n_jobs > 1
        n_jobs=8  # Use 8 cores
    )
    
    print("Evaluation completed!")
    print("Note: The system automatically chooses parallel versions when n_jobs > 1 and joblib is available.")
