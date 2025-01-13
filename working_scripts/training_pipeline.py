import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the credit card fraud dataset."""
        try:
            logger.info("Loading dataset...")
            df = pd.read_csv(filepath)
            
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def train_model(self, X_train, y_train):
        """Train the fraud detection model."""
        try:
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance."""
        try:
            logger.info("Evaluating model...")
            y_pred = self.model.predict(X_test)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Print confusion matrix
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Calculate and print feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                'importance': self.model.feature_importances_
            })
            print("\nTop 10 Most Important Features:")
            print(feature_importance.sort_values('importance', ascending=False).head(10))
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def save_model(self, filepath='fraud_model.pkl'):
        """Save the trained model and scaler."""
        try:
            logger.info(f"Saving model to {filepath}...")
            model_artifacts = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_artifacts, filepath)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    # Initialize model
    fraud_detector = FraudDetectionModel()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = fraud_detector.load_and_preprocess_data(
        'creditcard.csv'  # Update this path to your dataset location
    )
    
    # Train model
    fraud_detector.train_model(X_train, y_train)
    
    # Evaluate model
    fraud_detector.evaluate_model(X_test, y_test)
    
    # Save model
    fraud_detector.save_model()

if __name__ == "__main__":
    main()