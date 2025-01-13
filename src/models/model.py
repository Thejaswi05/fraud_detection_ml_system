import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import logging
from typing import Tuple, Union, List

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = ['Amount', 'Time']
        self.target_column = 'Class'
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the credit card fraud dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            
            # Select features (using only Amount and Time for simplicity)
            X = df[self.feature_columns].copy()
            y = df[self.target_column]
            
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train(self, data_path: str, random_state: int = 42) -> None:
        """Train the fraud detection model"""
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data(data_path)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=random_state,
                stratify=y
            )
            
            # Initialize and fit the scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train the model
            self.model = LogisticRegression(
                random_state=random_state,
                class_weight='balanced',
                max_iter=1000
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate and log performance metrics
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Model trained successfully. Train score: {train_score:.4f}, Test score: {test_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """Save the trained model and scaler"""
        try:
            # Create directories if they don't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and scaler
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def load_model(
    model_path: str = "models/fraud_detection_model.pkl",
    scaler_path: str = "models/scaler.pkl"
) -> Tuple[LogisticRegression, StandardScaler]:
    """Load the trained model and scaler"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def create_features(amount: float, time: float) -> np.ndarray:
    """Create feature array from transaction details"""
    return np.array([[amount, time]])

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and train model
    model = FraudDetectionModel()
    
    # Train model
    model.train(
        data_path="data/creditcard.csv",
        random_state=42
    )
    
    # Save model and scaler
    model.save_model(
        model_path="models/fraud_detection_model.pkl",
        scaler_path="models/scaler.pkl"
    )
