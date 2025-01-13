from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict
from pathlib import Path

from src.models.model import load_model, create_features
from src.monitoring.model_monitoring import ModelMonitor
from src.utils.logger import setup_logger
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from datetime import datetime, timedelta
import os

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = setup_logger(__name__)

# Add root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Fraud Detection API</title>
        </head>
        <body>
            <h1>Fraud Detection API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/redoc">/redoc</a> - Alternative API documentation</li>
                <li><a href="/health">/health</a> - Health check endpoint</li>
                <li>POST /predict - Make fraud predictions</li>
            </ul>
        </body>
    </html>
    """

def load_reference_data(sample_size: int = 10000):
    """
    Load reference data from credit card dataset
    Args:
        sample_size: Number of records to use as reference data
    """
    try:
        # Path to the credit card dataset
        data_path = Path("src/data/creditcard.csv")
        
        if data_path.exists():
            # Load the full dataset
            df = pd.read_csv(data_path)
            
            # Stratified sampling to maintain fraud ratio
            fraud_df = df[df['Class'] == 1]
            non_fraud_df = df[df['Class'] == 0]
            
            # Calculate sample sizes maintaining original ratio
            fraud_ratio = len(fraud_df) / len(df)
            fraud_sample_size = int(sample_size * fraud_ratio)
            non_fraud_sample_size = sample_size - fraud_sample_size
            
            # Sample from both classes
            sampled_fraud = fraud_df.sample(n=min(fraud_sample_size, len(fraud_df)))
            sampled_non_fraud = non_fraud_df.sample(n=min(non_fraud_sample_size, len(non_fraud_df)))
            
            # Combine samples
            reference_data = pd.concat([sampled_fraud, sampled_non_fraud])
            
            # Shuffle the data
            reference_data = reference_data.sample(frac=1).reset_index(drop=True)
            
            logger.info(f"Reference data loaded successfully. Shape: {reference_data.shape}")
            logger.info(f"Fraud ratio in reference data: {len(reference_data[reference_data['Class'] == 1]) / len(reference_data):.4f}")
            
            return reference_data[['Amount', 'Time', 'Class']]
            
        else:
            logger.warning(f"Credit card dataset not found at {data_path}. Using synthetic data instead.")
            # Fallback to synthetic data if file doesn't exist
            reference_data = pd.DataFrame({
                'Amount': np.random.uniform(0, 1000, sample_size),
                'Time': np.random.uniform(0, 86400, sample_size),
                'Class': np.random.choice([0, 1], sample_size, p=[0.999, 0.001])
            })
            return reference_data
            
    except Exception as e:
        logger.error(f"Error loading reference data: {str(e)}")
        raise

# Load model and scaler
try:
    model, scaler = load_model()
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Initialize monitor with reference data
try:
    reference_data = load_reference_data()
    monitor = ModelMonitor(
        reference_data=reference_data,
        model=model,
        feature_columns=['Amount', 'Time']
    )
    logger.info("Model monitoring initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model monitoring: {str(e)}")
    raise

class Transaction(BaseModel):
    amount: float
    time: int

class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    feature_importance: Optional[Dict[str, float]] = None
    prediction_id: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction, background_tasks: BackgroundTasks):
    """
    Make fraud prediction for a transaction and trigger monitoring
    """
    try:
        # Create feature array
        features = create_features(transaction.amount, transaction.time)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = bool(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        # Generate prediction ID
        prediction_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(['amount', 'time'], model.feature_importances_))
        
        # Prepare response
        response = {
            "is_fraud": prediction,
            "probability": probability,
            "feature_importance": feature_importance,
            "prediction_id": prediction_id
        }
        
        # Trigger monitoring in background
        current_data = pd.DataFrame([{
            'Amount': transaction.amount,
            'Time': transaction.time,
            'Class': int(prediction)
        }])
        background_tasks.add_task(monitor.analyze_prediction, current_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files directory for reports
reports_dir = Path("monitoring/reports")
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

@app.get("/monitoring/reports",
    summary="List all monitoring reports",
    description="Returns a list of all available monitoring reports"
)
async def list_reports():
    """List all available monitoring reports"""
    try:
        reports = []
        for file in reports_dir.glob("*"):
            if file.suffix in ['.html', '.json']:
                reports.append({
                    "filename": file.name,
                    "type": file.suffix[1:],
                    "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "size": file.stat().st_size,
                    "url": f"/reports/{file.name}"
                })
        return {"reports": sorted(reports, key=lambda x: x['created'], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/latest",
    summary="Get latest monitoring results",
    description="Returns the latest monitoring report data"
)
async def get_latest_monitoring():
    """Get the latest monitoring results"""
    try:
        # Find the latest JSON report
        json_files = list(reports_dir.glob("*drift*.json"))
        if not json_files:
            return {"message": "No monitoring reports available"}
            
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            report_data = json.load(f)
            
        return {
            "timestamp": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
            "report": report_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/generate",
    summary="Generate new monitoring report",
    description="Generates a new monitoring report on demand"
)
async def generate_monitoring_report(
    days: int = Query(default=7, description="Number of days of data to include")
):
    """Generate a new monitoring report on demand"""
    try:
        # Get recent predictions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_data = pd.DataFrame([{
            'Amount': transaction.amount,
            'Time': transaction.time,
            'Class': int(prediction)
        } for transaction, prediction in prediction_store.get_recent_predictions(days)])
        
        if current_data.empty:
            return {"message": "No data available for monitoring"}
            
        # Generate reports
        data_drift_report = monitor.generate_data_drift_report(current_data)
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data_drift_{timestamp}"
        monitor.save_report(data_drift_report, report_path)
        
        return {
            "message": "Report generated successfully",
            "report_url": f"/reports/{report_path}.html",
            "json_url": f"/reports/{report_path}.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "reference_data_shape": reference_data.shape if reference_data is not None else None
    }