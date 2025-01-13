import schedule
import time
from src.monitoring.model_monitoring import ModelMonitor
from src.utils.config import load_config
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)

def run_monitoring_job():
    """Regular monitoring job"""
    try:
        config = load_config()
        
        # Load reference and current data
        reference_data = pd.read_csv(config.monitoring.reference_data_path)
        current_data = load_current_data()  # Implement this based on your data storage
        
        monitor = ModelMonitor(reference_data)
        
        # Generate reports
        data_drift_report = monitor.generate_data_drift_report(current_data)
        target_drift_report = monitor.generate_target_drift_report(
            current_data, 
            target_column='fraud'
        )
        
        # Save reports
        monitor.save_report(data_drift_report, "data_drift")
        monitor.save_report(target_drift_report, "target_drift")
        
        # Check for alerts
        if data_drift_report.has_significant_drift():
            send_alert("Data drift detected!")
            
    except Exception as e:
        logger.error(f"Monitoring job failed: {str(e)}")

if __name__ == "__main__":
    # Schedule monitoring job
    schedule.every().day.at("00:00").do(run_monitoring_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)