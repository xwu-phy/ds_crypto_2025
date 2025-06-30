# erdos_src/LSTM/data_loader.py
import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from config import get_base_config
from data_processing import get_features_and_target


class LSTMDataLoader:
    """LSTM data loader for scaling and sequence creation."""
    def __init__(self, scaler_type = 'standard', feature_columns = None,
                 sequence_length = None, chosen_coin = None, 
                 granularity = None, use_overlap = False):

        config = get_base_config()
        self.chosen_coin = chosen_coin or config['data']['chosen_coin']
        self.granularity = granularity or config['data']['granularity'] 
        self.sequence_length = sequence_length or 20
        self.target_column = config['data']['columns']['target']
        self.use_overlap = use_overlap
        
        # Initialize scaler
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type} | Valid types: standard, minmax, robust.")
        
        # Set feature columns
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            raise ValueError(f"Feature selection file not specified")

    
    def load_data(self):
        """Load data, scale features, and create sequences."""
        train_df, test_df = get_features_and_target(coin_id=self.chosen_coin, freq=self.granularity, split=True)
        
        X_train = train_df[self.feature_columns].values
        y_train = train_df[self.target_column].values
        X_test = test_df[self.feature_columns].values
        y_test = test_df[self.target_column].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create sequences
        if self.use_overlap:
            X_train_seq, y_train_seq, X_test_seq, y_test_seq = self._create_sequences_with_overlap(
                X_train_scaled, y_train, X_test_scaled, y_test, self.sequence_length)
        else:
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train, self.sequence_length)
            X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test, self.sequence_length)
        
        print(f"Sequences created: X_train {X_train_seq.shape}, X_test {X_test_seq.shape}")
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq
    
    def _create_sequences(self, data, targets, sequence_length):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(targets[i])
        return np.array(X), np.array(y)
    
    def _create_sequences_with_overlap(self, X_train, y_train, X_test, y_test, 
                                     sequence_length: int):
        """Create sequences with overlap to eliminate gaps in test predictions."""
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        
        # Extend test data with last part of training data
        extended_X = np.concatenate([X_train[-sequence_length:], X_test], axis=0)
        extended_y = np.concatenate([y_train[-sequence_length:], y_test], axis=0)
        
        X_test_extended, _ = self._create_sequences(extended_X, extended_y, sequence_length)
        X_test_seq = X_test_extended[:len(y_test)]
        y_test_seq = y_test
        
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq
    
    def get_scaler(self):
        return self.scaler
    