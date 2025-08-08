import os

import mlflow


def setup_mlflow():
    """Setup MLflow configuration"""
    # For local development
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # For production (uncomment when ready)
    # mlflow.set_tracking_uri("postgresql://mlflow_user:password@34.76.186.238:5432/mlflow")

    # Set experiment
    mlflow.set_experiment("breast-cancer-classification")


def get_mlflow_client():
    """Get MLflow client"""
    return mlflow.tracking.MlflowClient()
