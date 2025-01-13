from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import logging
from typing import Optional, Dict

from src.monitoring.model_monitoring import ModelMonitor
from src.utils.logger import setup_logger
from src.models.model import load_model, create_features

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Setup logging
logger = setup_logger(__name__)


# Load model and scaler
model, scaler = load_model()

class Transaction(BaseModel):
    amount: float
    time: int

class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    feature_importance: Optional[Dict[str, float]] = None
    prediction_id: str

class PredictionStore:
    def __init__(self):
        self.predictions = []
    
    def add_prediction(self, transaction: dict, prediction: dict):
        prediction_record = {
            "timestamp": datetime.now(),
            "amount": transaction["amount"],
            "time": transaction["time"],
            "is_fraud": prediction["is_fraud"],
            "probability": prediction["probability"]
        }
        self.predictions.append(prediction_record)
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Get recent predictions as DataFrame for monitoring"""
        if not self.predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.predictions)
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        return df[df['timestamp'] >= cutoff_date]

# Initialize prediction store and monitor
prediction_store = PredictionStore()

# Load reference data for monitoring
def load_reference_data():
    try:
        # Replace this with your actual reference data loading logic
        reference_data = pd.read_csv("data/reference_data.csv")
        return reference_data
    except Exception as e:
        logger.error(f"Error loading reference data: {str(e)}")
        return pd.DataFrame()

# Initialize monitor with model and feature columns
monitor = ModelMonitor(
    reference_data=load_reference_data(),
    model=model,
    feature_columns=['Amount', 'Time']
)

def make_prediction(transaction: Transaction) -> dict:
    """Make fraud prediction for a transaction"""
    try:
        # Create feature array
        features = create_features(transaction.amount, transaction.time)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_names = ['amount', 'time']
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            "is_fraud": bool(prediction),
            "probability": float(probability),
            "feature_importance": feature_importance,
            "prediction_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

def send_alert(message: str):
    """Send alert for drift detection"""
    # Implement your alert mechanism here (email, Slack, etc.)
    logger.warning(f"Alert: {message}")

def background_monitoring(current_data: pd.DataFrame):
    """Background task for model monitoring"""
    try:
        if current_data.empty:
            logger.warning("No data available for monitoring")
            return
            
        # Generate comprehensive monitoring data
        dashboard_data = monitor.generate_monitoring_dashboard(current_data)
        
        # Generate and save reports
        data_drift_report = monitor.generate_data_drift_report(current_data)
        predictions = model.predict(current_data[['Amount', 'Time']])
        target_drift_report = monitor.generate_target_drift_report(current_data, predictions)
        
        # Save reports
        monitor.save_report(data_drift_report, "data_drift")
        monitor.save_report(target_drift_report, "target_drift")
        
        # Check for drift and send alerts if necessary
        if monitor.check_drift_threshold(data_drift_report):
            send_alert("Significant data drift detected!")
            
        # Store monitoring results
        store_monitoring_results(dashboard_data)
            
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction, background_tasks: BackgroundTasks):
    try:
        # Make prediction
        prediction = make_prediction(transaction)
        
        # Generate explanation for this prediction
        instance = pd.DataFrame([{
            'Amount': transaction.amount,
            'Time': transaction.time
        }])
        
        explanation = monitor.analyze_prediction(instance)
        prediction['explanation'] = explanation
        
        # Store prediction and trigger monitoring
        prediction_store.add_prediction(transaction.dict(), prediction)
        current_data = prediction_store.get_recent_predictions()
        background_tasks.add_task(background_monitoring, current_data)
        
        return prediction
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}
