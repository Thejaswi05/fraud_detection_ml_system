import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *
import shap
import lime
import lime.lime_tabular
from datetime import datetime
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame, model, feature_columns: list):
        """
        Initialize model monitor
        """
        self.reference_data = reference_data
        self.model = model
        self.feature_columns = feature_columns
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.LinearExplainer(
            model, 
            reference_data[feature_columns],
            feature_perturbation="interventional"
        )
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            reference_data[feature_columns].values,
            feature_names=feature_columns,
            class_names=['legitimate', 'fraud'],
            mode='classification'
        )

    def generate_data_drift_report(self, current_data: pd.DataFrame) -> Report:
        """Generate comprehensive data drift report"""
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            ColumnDriftMetric(column_name="Amount"),
            ColumnDriftMetric(column_name="Time"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data[self.feature_columns], 
            current_data=current_data[self.feature_columns]
        )
        return data_drift_report

    def generate_target_drift_report(self, current_data: pd.DataFrame, 
                                   predictions: np.ndarray) -> Report:
        """Generate target drift report"""
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
        ])
        
        # Add predictions to current data
        current_data_with_predictions = current_data.copy()
        current_data_with_predictions['prediction'] = predictions
        
        target_drift_report.run(
            reference_data=self.reference_data, 
            current_data=current_data_with_predictions,
            column_mapping={
                'target': 'Class',
                'prediction': 'prediction'
            }
        )
        return target_drift_report

    def generate_shap_values(self, instance: pd.DataFrame) -> Dict[str, float]:
        """Generate SHAP values for a prediction"""
        try:
            shap_values = self.shap_explainer.shap_values(instance)
            
            # For binary classification, we take the values for class 1 (fraud)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            return dict(zip(self.feature_columns, shap_values[0]))
        except Exception as e:
            logger.error(f"Error generating SHAP values: {str(e)}")
            return {}

    def generate_lime_explanation(self, instance: np.ndarray) -> Dict[str, float]:
        """Generate LIME explanation for a prediction"""
        try:
            explanation = self.lime_explainer.explain_instance(
                instance[0], 
                self.model.predict_proba,
                num_features=len(self.feature_columns)
            )
            
            return dict(explanation.as_list())
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return {}

    def analyze_prediction(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of a single prediction"""
        try:
            analysis = {
                'shap_values': self.generate_shap_values(instance),
                'lime_explanation': self.generate_lime_explanation(instance.values),
                'timestamp': datetime.now().isoformat()
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in prediction analysis: {str(e)}")
            return {}

    def save_report(self, report: Report, report_type: str):
        """Save monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            # Save HTML report
            report.save_html(f"monitoring/reports/{report_type}_{timestamp}.html")
            
            # Save JSON metrics
            report.save_json(f"monitoring/reports/{report_type}_{timestamp}.json")
            
            logger.info(f"Saved {report_type} report for timestamp {timestamp}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

    def check_drift_threshold(self, report: Report, threshold: float = 0.1) -> bool:
        """Check if drift exceeds threshold"""
        try:
            metrics = report.as_dict()
            
            # Check data drift metric
            if 'data_drift' in metrics:
                drift_score = metrics['data_drift'].get('data_drift_score', 0)
                return drift_score > threshold
                
            return False
        except Exception as e:
            logger.error(f"Error checking drift threshold: {str(e)}")
            return False

    def generate_monitoring_dashboard(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive monitoring dashboard data"""
        try:
            # Generate all reports
            data_drift_report = self.generate_data_drift_report(current_data)
            predictions = self.model.predict(current_data[self.feature_columns])
            target_drift_report = self.generate_target_drift_report(current_data, predictions)
            
            # Sample explanation for recent predictions
            sample_idx = np.random.choice(len(current_data), min(5, len(current_data)))
            sample_explanations = [
                self.analyze_prediction(current_data.iloc[[idx]]) 
                for idx in sample_idx
            ]
            
            dashboard_data = {
                'data_drift_metrics': data_drift_report.as_dict(),
                'target_drift_metrics': target_drift_report.as_dict(),
                'sample_explanations': sample_explanations,
                'timestamp': datetime.now().isoformat()
            }
            
            return dashboard_data
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}")
            return {}
