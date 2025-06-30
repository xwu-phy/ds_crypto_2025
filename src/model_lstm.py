import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC, Precision, Recall
import os
from datetime import datetime


# Default configuration 
MODEL_CONFIG = {
    # LSTM layer parameters
    'lstm_units_l1': 128,      
    'lstm_units_l2': 64,       
    'dropout_rate_l1': 0.2,    
    'dropout_rate_l2': 0.2,    
    
    # Dense layer parameters
    'dense_units': 32,         
    
    # Training parameters
    'learning_rate': 0.001,    
    'batch_size': 32,         
    'epochs': 100,            
    
    # Callback parameters
    'patience': 10,            
    'min_delta': 0.001,        
    'factor': 0.5,            
    'min_lr': 1e-7,           
    'monitor': 'val_loss',    
    'verbose': 1,

    # LSTM model parameters
    'fit_verbose': 2,          
    'model_path': 'models/LSTM/model/',
    'pred_verbose': 0        
}

class LSTMModel:
    """LSTM model for binary classification of cryptocurrency price direction."""

    def __init__(self, input_shape, lstm_units_l1 = None, lstm_units_l2 = None,
                 dropout_rate_l1 = None, dropout_rate_l2 = None,
                 dense_units = None, learning_rate = None,
                 model_path = None):

        self.input_shape = input_shape
        self.lstm_units_l1 = lstm_units_l1 or MODEL_CONFIG['lstm_units_l1']
        self.lstm_units_l2 = lstm_units_l2 or MODEL_CONFIG['lstm_units_l2']
        self.dropout_rate_l1 = dropout_rate_l1 or MODEL_CONFIG['dropout_rate_l1']
        self.dropout_rate_l2 = dropout_rate_l2 or MODEL_CONFIG['dropout_rate_l2']
        self.dense_units = dense_units or MODEL_CONFIG['dense_units']
        self.learning_rate = learning_rate or MODEL_CONFIG['learning_rate']
        
        self.model = None
        self.model_path = model_path or "./models/"
        os.makedirs(self.model_path, exist_ok=True)
    
    def create_model(self) -> Sequential:
        model = Sequential([
            Input(shape=self.input_shape),
            
            LSTM(units=self.lstm_units_l1, return_sequences=True),
            Dropout(self.dropout_rate_l1),
            
            LSTM(units=self.lstm_units_l2, return_sequences=False),
            Dropout(self.dropout_rate_l2),
            
            Dense(units=self.dense_units, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[AUC(name='roc_auc'), 'accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, patience = None, min_delta = None,
                     factor = None, min_lr = None, monitor = None,
                     save_best_model_name = None, verbose = None):

        patience = patience or MODEL_CONFIG['patience']
        min_delta = min_delta or MODEL_CONFIG['min_delta']
        factor = factor or MODEL_CONFIG['factor']
        min_lr = min_lr or MODEL_CONFIG['min_lr']
        monitor = monitor or MODEL_CONFIG['monitor']
        verbose = verbose or MODEL_CONFIG['verbose']
        
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience//2,
                min_lr=min_lr,
                verbose=verbose
            )
        ]
        
        if save_best_model_name is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_with_timestamp = f"{save_best_model_name}_{timestamp}"
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.model_path, name_with_timestamp),
                    monitor=monitor,
                    save_best_only=True,
                    verbose=verbose
                )
            )
        
        return callbacks
    
    def train(self, X_train, y_train, X_val = None, y_val = None,
              batch_size = None, epochs = None, validation_split = 0.2, 
              callbacks = None, verbose = None):

        if self.model is None:
            self.create_model()
        
        batch_size = batch_size or MODEL_CONFIG['batch_size']
        epochs = epochs or MODEL_CONFIG['epochs']
        verbose = verbose or MODEL_CONFIG['fit_verbose']

        if callbacks is None:
            callbacks = self.get_callbacks()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            validation_split=validation_split if X_val is None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray, verbose = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been created yet. create_model() first.")
        verbose = verbose or MODEL_CONFIG['pred_verbose']
        
        return self.model.predict(X, verbose=verbose)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose=None) -> dict:
        if self.model is None:
            raise ValueError("Model has not been created yet. create_model() first.")
        verbose = verbose or MODEL_CONFIG['fit_verbose']
        
        y_pred_proba = self.predict(X, verbose=verbose)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model has not been created yet. create_model() first.")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        self.model = tf.keras.models.load_model(filepath)