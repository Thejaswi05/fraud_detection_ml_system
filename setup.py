from setuptools import setup, find_packages

setup(
    name="fraud_detection_ml_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas==2.1.4",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "evidently==0.5.1",
        "pydantic==2.5.2",
        "shap==0.43.0",
        "lime==0.2.0.1",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "joblib==1.3.2",
        "python-dotenv==1.0.0",
    ],
)