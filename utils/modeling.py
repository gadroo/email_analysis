"""
Modeling utilities for email campaign prediction.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    config: Dict
) -> RandomForestClassifier:
    """
    Train a predictive model for email clicks.
    
    Args:
        X_train: Feature DataFrame for training
        y_train: Target Series for training
        config: Configuration dictionary with model parameters
        
    Returns:
        Trained model
    """
    model_config = config["model"]
    
    if model_config["type"] == "random_forest":
        model = RandomForestClassifier(**model_config["params"])
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    """
    Evaluate the model and calculate various performance metrics.
    
    Args:
        model: Trained model
        X_test: Feature DataFrame for testing
        y_test: Target Series for testing
        
    Returns:
        Tuple containing:
            - Dictionary with performance metrics
            - DataFrame with classification report
            - Array with feature importances
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    # Get classification report as DataFrame
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    return metrics, report_df, feature_importances


def get_feature_importance(
    model: RandomForestClassifier, 
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        DataFrame with feature importances sorted in descending order
    """
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return feature_importance


def save_model(model: RandomForestClassifier, output_path: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        output_path: Path to save the model
        
    Returns:
        None
    """
    import joblib
    
    joblib.dump(model, output_path)
    
    
def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    import joblib
    
    return joblib.load(model_path) 