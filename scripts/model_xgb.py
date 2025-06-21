import os
import pandas as pd
import logging
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
#from sklearn.metrics import make_scorer, precision_score
from erdos_src.data_processing import get_features_and_target
from erdos_src.config import CFG
from erdos_src.model_evaluation import evaluate_model


logger = logging.getLogger(__name__)

def train_xgb_model(chosen_coin=None, **xgb_params):
    train_df, _ = get_features_and_target(CFG.FF_DATA_PATH, chosen_coin=chosen_coin)
    X_train = train_df[CFG.FEATURES]
    y_train = train_df[CFG.TARGET_COLUMN]
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        **xgb_params
    )
    model.fit(X_train, y_train)
    return model


def tune_xgb_model(MODEL_CATEGORY="XGBoost", param_dist=None, n_iter=20, cv_splits=5):
    logger.info("Setting up RandomizedSearchCV...")
    train_df, _ = get_features_and_target(CFG.FF_DATA_PATH, split=True, chosen_coin=CFG.CHOSEN_COIN)
    X_train = train_df[CFG.FEATURES]
    y_train = train_df[CFG.TARGET_COLUMN]
    model = XGBClassifier(
        objective='binary:logistic',
        enable_categorical=True,  
        eval_metric='logloss', 
        random_state=42,
        n_jobs=-1
        )
    
    if param_dist is None:
        param_dist = {
            'max_depth': randint(3, 8),
            'min_child_weight': randint(1, 16),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(50, 2000),
            'subsample': uniform(0.7, 0.3),                                     # Floats from 0.7 up to 1.0
            'colsample_bytree': uniform(0.7, 0.3),
            'gamma': uniform(0, 1),                                             # Floats from 0 up to 1
            'scale_pos_weight': uniform(0.8, 0.35),                            # Floats from 0.8 up to 1.15
            'reg_alpha': uniform(0, 2),                                         # L1 regularization
            'reg_lambda': uniform(0, 5)                                         # L2 regularization
        }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=CFG.TUNING_ITERATIONS,
        scoring=['precision', 'recall', 'f1', 'roc_auc', 'accuracy'],
        refit = 'precision',
        cv=tscv, 
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)

 
    # Create directory structure
    os.makedirs(f"{CFG.TUNING_RESULTS_PATH}{MODEL_CATEGORY}/{CFG.GRANULARITY}/{CFG.CHOSEN_COIN}", exist_ok=True)
    tuning_result_path = f"{CFG.TUNING_RESULTS_PATH}{MODEL_CATEGORY}/{CFG.GRANULARITY}/{CFG.CHOSEN_COIN}/tuning_results.pkl"
    model_path = f"{CFG.TUNING_RESULTS_PATH}{MODEL_CATEGORY}/{CFG.GRANULARITY}/{CFG.CHOSEN_COIN}/final_xgb_model.json"

    # Save tuning results and model
    with open(tuning_result_path, 'wb') as f:
        pd.to_pickle(search, f)
    best_model = search.best_estimator_
    best_model.save_model(model_path)
    
    logging.info(f"Tuning complete. Best precision score: {search.best_score_:.4f}")
    logging.info(f"Best parameters found: {search.best_params_}")
    logging.info(f"Model and result saved in {model_path} and {tuning_result_path}")
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_xgb_model(model, methods=None):
    results = evaluate_model(model, methods=methods, n_jobs=-1)
    return results
