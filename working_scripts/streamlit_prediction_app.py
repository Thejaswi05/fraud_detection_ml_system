import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging

# If you're curious of all the loggers
print(st.logger._loggers)  

streamlit_root_logger = logging.getLogger(st.__name__)

def load_model():
    """Load the model artifacts from pickle file."""
    # Load the dictionary containing model and scaler
    artifacts = joblib.load('fraud_model.pkl')
    
    # Extract model and scaler from artifacts
    model = artifacts['model']
    scaler = artifacts['scaler']
    
    return model, scaler

def create_features(amount, time):
    """Create a feature array with zeros for V1-V28 and actual values for Amount and Time."""
    # BEFORE:
    # feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']  # This was wrong order
    
    # AFTER:
    # Use exact order from training data
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
                    
    features = pd.DataFrame(np.zeros((1, 30)), columns=feature_names)
    
    features['Time'] = time
    features['Amount'] = amount
    
    return features

def create_prediction_app():
    st.title('Credit Card Fraud Detection')
    
    # Load model and scaler
    try:
        model, scaler = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Input fields for transaction details
    amount = st.number_input('Transaction Amount ($)', min_value=0.0, format='%f')
    time = st.number_input('Time (seconds from midnight)', min_value=0)
    
    # Initialize session state for the checkbox if it doesn't exist
    if 'show_importance' not in st.session_state:
        st.session_state.show_importance = False
    
    # Move checkbox outside the Predict button logic
    st.session_state.show_importance = st.checkbox('Show feature importance details', 
                                                 value=st.session_state.show_importance)
    
    if st.button('Predict'):
        try:
            # Create feature array
            features = create_features(amount, time)
            
            # Scale the features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Show results
            if prediction == 1:
                st.error(f'⚠️ Fraudulent Transaction Detected!')
                st.write(f'Probability of fraud: {probability:.2%}')
            else:
                st.success(f'✅ Transaction appears legitimate')
                st.write(f'Probability of fraud: {probability:.2%}')
            
            # Show feature importance if checkbox is checked
            if st.session_state.show_importance:
                # Debug: Print if model is loaded
                st.write("Checking model...")
                st.write(f"Model type: {type(model)}")
                
                # Debug: Check if feature_importances_ exists
                if hasattr(model, 'feature_importances_'):
                    st.write("Feature importances are available")
                    importances = model.feature_importances_
                    st.write(f"Number of features: {len(importances)}")
                    
                    # Create feature list
                    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
                    
                    # Debug: Print raw data
                    st.write("Raw importances:", importances)
                    
                    # Create DataFrame
                    feature_imp = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    })
                    st.write("DataFrame created")
                    
                    # Display raw DataFrame first
                    st.write("Raw DataFrame:")
                    st.write(feature_imp)
                    
                    # Then try sorting
                    feature_imp = feature_imp.sort_values('Importance', ascending=False)
                    st.write("Sorted top 10 features:")
                    st.write(feature_imp.head(10))
                else:
                    st.write("No feature importances found in the model")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Debug information:")
            st.write(f"Features shape: {features.shape}")
            st.write(f"Model type: {type(model)}")

if __name__ == '__main__':
    create_prediction_app()