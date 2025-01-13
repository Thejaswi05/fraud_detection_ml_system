import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)

def load_model(
    model_path: str = "models/fraud_detection_model.pkl",
    scaler_path: str = "models/scaler.pkl"
) -> Tuple[LogisticRegression, StandardScaler]:
    """Load the trained model and scaler"""
    try:
        # Check if model files exist
        if not Path(model_path).exists() or not Path(scaler_path).exists():
            # If models don't exist, train new ones
            logger.info("Model files not found. Training new model...")
            model, scaler = train_model()
            save_model(model, scaler, model_path, scaler_path)
            return model, scaler
            
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

def train_model(
    data_path: str = "src/data/creditcard.csv",
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a new model if one doesn't exist"""
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare features
        X = df[['Amount', 'Time']].values
        y = df['Class'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(
            random_state=random_state,
            class_weight='balanced',
            max_iter=1000
        )
        model.fit(X_scaled, y)
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    model_path: str = "models/fraud_detection_model.pkl",
    scaler_path: str = "models/scaler.pkl"
) -> None:
    """Save the trained model and scaler"""
    try:
        # Create directories if they don't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
    