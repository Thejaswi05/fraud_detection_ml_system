import streamlit as st
import pandas as pd
from src.monitoring.model_monitoring import ModelMonitor
from src.utils.config import load_config
from pathlib import Path
import json

def load_monitoring_data():
    """Load monitoring data from storage"""
    config = load_config()
    reference_data = pd.read_csv(config.reference_data_path)
    return reference_data

def render_monitoring_dashboard():
    st.title("ML Model Monitoring Dashboard")
    
    # Load data
    reference_data = load_monitoring_data()
    monitor = ModelMonitor(reference_data)
    
    # Date range selector
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    if st.button("Generate Reports"):
        # Load current data for selected date range
        current_data = load_current_data(start_date, end_date)
        
        # Generate reports
        data_drift_report = monitor.generate_data_drift_report(current_data)
        target_drift_report = monitor.generate_target_drift_report(
            current_data, 
            target_column='fraud'
        )
        
        # Display metrics
        st.header("Data Drift Metrics")
        display_drift_metrics(data_drift_report)
        
        st.header("Target Drift Metrics")
        display_drift_metrics(target_drift_report)
        
        # Save reports
        monitor.save_report(data_drift_report, "data_drift")
        monitor.save_report(target_drift_report, "target_drift")

def display_drift_metrics(report):
    """Display drift metrics in Streamlit"""
    metrics = report.as_dict()
    
    for metric_name, metric_value in metrics.items():
        st.subheader(metric_name)
        st.write(metric_value)