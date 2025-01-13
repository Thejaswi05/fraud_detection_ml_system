from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
import shap
import lime
import lime.lime_tabular
from datetime import datetime
import pandas as pd
import numpy as np
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
        report = Report(metrics=[
            DataDriftPreset(),
            DataDriftTable(),
            DatasetDriftMetric()
        ])
        
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = self.feature_columns
        
        report.run(
            reference_data=self.reference_data[self.feature_columns], 
            current_data=current_data[self.feature_columns],
            column_mapping=column_mapping
        )
        return report

    def generate_drift_test_suite(self, current_data: pd.DataFrame) -> TestSuite:
        """Generate drift test suite"""
        test_suite = TestSuite(tests=[
            DataDriftTestPreset(),
        ])
        
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = self.feature_columns
        
        test_suite.run(
            reference_data=self.reference_data[self.feature_columns],
            current_data=current_data[self.feature_columns],
            column_mapping=column_mapping
        )
        return test_suite

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