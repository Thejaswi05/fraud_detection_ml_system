import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta

def main():
    st.set_page_config(page_title="Fraud Detection Monitoring", layout="wide")
    st.title("Fraud Detection Monitoring Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Reports", "Data Drift", "Model Performance"])
    
    if page == "Overview":
        show_overview()
    elif page == "Reports":
        show_reports()
    elif page == "Data Drift":
        show_data_drift()
    elif page == "Model Performance":
        show_model_performance()

def show_overview():
    st.header("System Overview")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Check API health
    try:
        health = requests.get("http://localhost:8000/health").json()
        
        with col1:
            st.metric("API Status", health['status'])
        with col2:
            if health.get('reference_data_shape'):
                st.metric("Reference Data Size", f"{health['reference_data_shape'][0]:,} records")
        with col3:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        # Add system metrics
        st.subheader("System Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['API Uptime', 'Model Version', 'Last Prediction'],
            'Value': ['100%', '1.0.0', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        st.table(metrics_df)
        
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

def show_reports():
    st.header("Monitoring Reports")
    
    # Add report generation button
    if st.button("Generate New Report"):
        try:
            response = requests.get("http://localhost:8000/monitoring/generate").json()
            st.success(response['message'])
            st.markdown(f"[View New Report]({response['report_url']})")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    
    # List available reports
    try:
        reports = requests.get("http://localhost:8000/monitoring/reports").json()
        
        st.subheader("Available Reports")
        
        # Create a dataframe for better display
        if reports.get('reports'):
            reports_df = pd.DataFrame(reports['reports'])
            reports_df['created'] = pd.to_datetime(reports_df['created'])
            reports_df['size'] = reports_df['size'].apply(lambda x: f"{x/1024:.2f} KB")
            
            # Display as a table
            st.dataframe(
                reports_df[['filename', 'type', 'created', 'size']],
                hide_index=True
            )
            
            # Add download links
            st.subheader("Download Reports")
            for _, report in reports_df.iterrows():
                st.markdown(f"[{report['filename']}]({report['url']})")
        else:
            st.info("No reports available yet")
            
    except Exception as e:
        st.error(f"Error loading reports: {str(e)}")

def show_data_drift():
    st.header("Data Drift Analysis")
    
    # Get latest monitoring results
    try:
        latest = requests.get("http://localhost:8000/monitoring/latest").json()
        
        if 'report' in latest:
            # Display drift metrics
            drift_data = latest['report'].get('data_drift', {})
            
            if drift_data:
                # Display metrics as a table
                st.subheader("Drift Metrics")
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.4f}"} 
                    for k, v in drift_data.items()
                ])
                st.table(metrics_df)
            else:
                st.info("No drift data available in the latest report")
        else:
            st.warning("No monitoring report available")
            
    except Exception as e:
        st.error(f"Error loading drift analysis: {str(e)}")

def show_model_performance():
    st.header("Model Performance Metrics")
    
    try:
        # Create sample performance metrics
        performance_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.998, 0.95, 0.92, 0.93]
        }
        
        # Display metrics
        st.subheader("Performance Metrics")
        st.table(pd.DataFrame(performance_data))
        
    except Exception as e:
        st.error(f"Error displaying performance metrics: {str(e)}")

if __name__ == "__main__":
    main()