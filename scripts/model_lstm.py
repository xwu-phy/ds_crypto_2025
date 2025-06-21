import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Project imports
from erdos_src.config import CFG
from erdos_src.data_processing import get_features_and_target

logger = logging.getLogger(__name__)

class LSTMCryptoPredictor:
    """
    LSTM model for predicting cryptocurrency return direction.
    
    This class implements a complete pipeline for:
    - Data preprocessing and sequence creation
    - LSTM model building and training
    - Model evaluation and prediction
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: Tuple[int, ...] = (100, 50),
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 15,
                 scaler_type: str = 'standard'):
        """
        Initialize the LSTM predictor.
        
        Args:
            sequence_length: Number of time steps to look back for prediction
            lstm_units: Tuple of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            validation_split: Fraction of training data to use for validation
            early_stopping_patience: Patience for early stopping
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.scaler_type = scaler_type
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.history = None
        
    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get the list of feature columns to use for training.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        # Define columns to exclude from features
        exclude_cols = [
            CFG.COIN_ID_COLUMN, 
            CFG.TIMESTAMP_COLUMN, 
            CFG.TARGET_COLUMN,
            'future_price', 
            'future_log_return',
            'year'  # categorical column that shouldn't be used as numeric feature
        ]
        
        # Get all numeric columns except those to exclude
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        logger.info(f"Selected {len(feature_cols)} feature columns: {feature_cols}")
        return feature_cols
    
    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        """
        Prepare and fit the scaler on training data.
        
        Args:
            train_df: Training dataframe
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit scaler on training data
        train_features = train_df[self.feature_columns].values
        self.scaler.fit(train_features)
        logger.info(f"Fitted {self.scaler_type} scaler on training data")
    
    def _create_sequences(self, data: np.ndarray, targets: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled feature data
            targets: Target values
            sequence_length: Length of sequences to create
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    chosen_coin: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
            chosen_coin: Specific coin to filter for (optional)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Preparing data for LSTM training...")
        
        # Filter by coin if specified
        if chosen_coin:
            train_df = train_df[train_df[CFG.COIN_ID_COLUMN] == chosen_coin].copy()
            test_df = test_df[test_df[CFG.COIN_ID_COLUMN] == chosen_coin].copy()
            logger.info(f"Filtered data for coin: {chosen_coin}")
        
        # Get feature columns
        self.feature_columns = self._get_feature_columns(train_df)
        
        # Prepare scaler
        self._prepare_scaler(train_df)
        
        # Scale the data
        train_scaled = self.scaler.transform(train_df[self.feature_columns].values)
        test_scaled = self.scaler.transform(test_df[self.feature_columns].values)
        
        # Get targets
        train_targets = train_df[CFG.TARGET_COLUMN].values
        test_targets = test_df[CFG.TARGET_COLUMN].values
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_scaled, train_targets, self.sequence_length)
        X_test, y_test = self._create_sequences(test_scaled, test_targets, self.sequence_length)
        
        logger.info(f"Training sequences shape: {X_train.shape}, targets: {y_train.shape}")
        logger.info(f"Testing sequences shape: {X_test.shape}, targets: {y_test.shape}")
        logger.info(f"Target distribution in training: {np.bincount(y_train.astype(int))}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
        """
        logger.info("Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.lstm_units[0], 
                      return_sequences=len(self.lstm_units) > 1,
                      input_shape=input_shape,
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units=units, 
                          return_sequences=return_sequences,
                          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(25, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        self.model = model
        logger.info("LSTM model built successfully")
        logger.info(f"Model summary:\n{model.summary()}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             model_save_path: Optional[str] = None) -> None:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            model_save_path: Path to save the best model (optional)
        """
        logger.info("Starting LSTM training...")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, 
                             min_lr=1e-7, verbose=1)
        ]
        
        if model_save_path:
            callbacks.append(
                ModelCheckpoint(filepath=model_save_path, monitor='val_loss', 
                              save_best_only=True, verbose=1)
            )
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = self.validation_split
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        logger.info("LSTM training completed")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating LSTM model...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).ravel()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Print detailed classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int).ravel()
        
        return y_pred, y_pred_proba.ravel()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


def train_lstm_model(chosen_coin: str = CFG.CHOSEN_COIN, 
                    model_save_path: Optional[str] = None,
                    **kwargs) -> Tuple[LSTMCryptoPredictor, Dict[str, float]]:
    """
    Main function to train and evaluate an LSTM model.
    
    Args:
        chosen_coin: Coin to train the model on
        model_save_path: Path to save the trained model
        **kwargs: Additional arguments for LSTMCryptoPredictor
        
    Returns:
        Tuple of (trained_predictor, evaluation_metrics)
    """
    logger.info(f"Starting LSTM training pipeline for {chosen_coin}")
    
    # Load and prepare data
    train_df, test_df = get_features_and_target(path=CFG.FF_DATA_PATH, split=True, chosen_coin=chosen_coin)
    
    # Initialize predictor
    predictor = LSTMCryptoPredictor(**kwargs)
    
    # Prepare data
    X_train, y_train, X_test, y_test = predictor.prepare_data(train_df, test_df, chosen_coin)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    predictor.build_model(input_shape)
    
    # Train model
    predictor.train(X_train, y_train, model_save_path=model_save_path)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    
    logger.info(f"LSTM training pipeline completed for {chosen_coin}")
    return predictor, metrics


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    predictor, metrics = train_lstm_model(
        chosen_coin=CFG.CHOSEN_COIN,
        sequence_length=60,
        lstm_units=(100, 50),
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        model_save_path=f"{CFG.TUNING_RESULTS_PATH}lstm_model_{CFG.CHOSEN_COIN}.h5"
    )
    
    print(f"\nFinal evaluation metrics: {metrics}")

