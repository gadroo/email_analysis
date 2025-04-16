"""
Unit tests for utility functions.
"""
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from utils.analysis import (
    analyze_factor_performance,
    calculate_campaign_metrics,
    estimate_targeting_improvement,
)
from utils.data_loader import create_output_dir, engineer_features, prepare_data
from utils.modeling import get_feature_importance, train_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Sample email data
    email_df = pd.DataFrame({
        "email_id": [1, 2, 3, 4, 5],
        "email_text": ["short_email", "long_email", "short_email", "long_email", "short_email"],
        "email_version": ["personalized", "generic", "personalized", "generic", "personalized"],
        "hour": [9, 12, 15, 18, 21],
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "user_country": ["US", "UK", "US", "FR", "ES"],
        "user_past_purchases": [0, 2, 5, 8, 10],
    })
    
    # Sample opened emails
    opened_df = pd.DataFrame({
        "email_id": [1, 3, 5]
    })
    
    # Sample clicked emails
    clicked_df = pd.DataFrame({
        "email_id": [1, 5]
    })
    
    return email_df, opened_df, clicked_df


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "data": {
            "email_table": "email (1)/email_table.csv",
            "opened_table": "email (1)/email_opened_table.csv",
            "clicked_table": "email (1)/link_clicked_table.csv",
            "output_dir": "output"
        },
        "analysis": {
            "min_segment_size": 2,
            "min_timing_samples": 1,
            "random_state": 42,
            "test_size": 0.3
        },
        "model": {
            "type": "random_forest",
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
                "random_state": 42
            }
        },
        "features": {
            "categorical_cols": [
                "email_text",
                "email_version",
                "weekday",
                "user_country"
            ],
            "purchase_bins": [-1, 0, 3, 7, 100],
            "purchase_labels": ["0", "1-3", "4-7", "8+"]
        },
        "visualization": {
            "dpi": 100,
            "figsize": [8, 5],
            "palette": "viridis",
            "save_format": "png"
        }
    }


def test_prepare_data(sample_data):
    """Test prepare_data function."""
    email_df, opened_df, clicked_df = sample_data
    
    # Prepare data
    df = prepare_data(email_df, opened_df, clicked_df)
    
    # Check that flags were added correctly
    assert "opened" in df.columns
    assert "clicked" in df.columns
    assert df["opened"].sum() == 3
    assert df["clicked"].sum() == 2
    assert df.loc[0, "opened"] == 1
    assert df.loc[0, "clicked"] == 1
    assert df.loc[1, "opened"] == 0
    assert df.loc[1, "clicked"] == 0


def test_calculate_campaign_metrics(sample_data):
    """Test calculate_campaign_metrics function."""
    email_df, opened_df, clicked_df = sample_data
    df = prepare_data(email_df, opened_df, clicked_df)
    
    # Calculate metrics
    metrics = calculate_campaign_metrics(df)
    
    # Check metrics
    assert metrics["total_emails"] == 5
    assert metrics["total_opened"] == 3
    assert metrics["total_clicked"] == 2
    assert metrics["open_rate"] == 60.0
    assert metrics["click_rate"] == 40.0
    assert metrics["click_to_open_rate"] == pytest.approx(66.67, rel=1e-2)


def test_analyze_factor_performance(sample_data):
    """Test analyze_factor_performance function."""
    email_df, opened_df, clicked_df = sample_data
    df = prepare_data(email_df, opened_df, clicked_df)
    
    # Analyze email_text factor
    performance = analyze_factor_performance(df, "email_text")
    
    # Check results
    assert "email_text" in performance.columns
    assert "total" in performance.columns
    assert "opened" in performance.columns
    assert "open_rate" in performance.columns
    assert "clicked" in performance.columns
    assert "click_rate" in performance.columns
    
    # Check specific values
    short_email_row = performance[performance["email_text"] == "short_email"]
    assert short_email_row["total"].values[0] == 3
    assert short_email_row["open_rate"].values[0] == pytest.approx(100.0)
    assert short_email_row["click_rate"].values[0] == pytest.approx(66.67, rel=1e-2)


def test_engineer_features(sample_data, sample_config):
    """Test engineer_features function."""
    email_df, opened_df, clicked_df = sample_data
    df = prepare_data(email_df, opened_df, clicked_df)
    
    # Engineer features
    df_encoded = engineer_features(df, sample_config)
    
    # Check that cyclic time features were added
    assert "hour_sin" in df_encoded.columns
    assert "hour_cos" in df_encoded.columns
    
    # Check that purchase buckets were created
    assert "purchase_bucket" in df_encoded.columns
    
    # Check that categorical variables were one-hot encoded
    assert "email_text_short_email" in df_encoded.columns
    assert "email_version_personalized" in df_encoded.columns
    assert "weekday_Monday" in df_encoded.columns
    assert "user_country_US" in df_encoded.columns


def test_create_output_dir(sample_config):
    """Test create_output_dir function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Modify config to use temp directory
        config = sample_config.copy()
        config["data"] = config["data"].copy()
        config["data"]["output_dir"] = os.path.join(temp_dir, "test_output")
        
        # Create output directory
        output_dir = create_output_dir(config)
        
        # Check that directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert str(output_dir) == os.path.join(temp_dir, "test_output")


def test_train_model(sample_data, sample_config):
    """Test train_model function."""
    email_df, opened_df, clicked_df = sample_data
    df = prepare_data(email_df, opened_df, clicked_df)
    df_encoded = engineer_features(df, sample_config)
    
    # Prepare features and target
    X = df_encoded.drop(["email_id", "hour", "opened", "clicked", "purchase_bucket"], axis=1)
    y = df_encoded["clicked"]
    
    # Train model
    model = train_model(X, y, sample_config)
    
    # Check model
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_get_feature_importance():
    """Test get_feature_importance function."""
    # Create mock model and data
    model = mock.Mock()
    model.feature_importances_ = np.array([0.3, 0.2, 0.5])
    
    X = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "feature3": [7, 8, 9]
    })
    
    # Get feature importance
    importance = get_feature_importance(model, X)
    
    # Check results
    assert list(importance["feature"]) == ["feature3", "feature1", "feature2"]
    assert list(importance["importance"]) == [0.5, 0.3, 0.2]


def test_estimate_targeting_improvement():
    """Test estimate_targeting_improvement function."""
    # Create test data
    y_test = pd.Series([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    y_prob = pd.Series([0.2, 0.1, 0.9, 0.3, 0.8, 0.4, 0.5, 0.3, 0.7, 0.2])
    current_ctr = 30.0
    
    # Estimate improvement
    percentiles = [20, 50, 100]
    targeting_results, improvement_metrics = estimate_targeting_improvement(
        y_test, y_prob, current_ctr, percentiles
    )
    
    # Check results
    assert len(targeting_results) == 3
    assert list(targeting_results["percentile"]) == [20, 50, 100]
    assert targeting_results["ctr"].iloc[0] == 100.0  # Top 20% has perfect CTR
    assert targeting_results["ctr"].iloc[1] == 60.0   # Top 50% has 60% CTR
    assert targeting_results["ctr"].iloc[2] == 30.0   # All 100% has 30% CTR
    
    assert improvement_metrics["current_ctr"] == 30.0
    assert improvement_metrics["best_percentile"] == 20
    assert improvement_metrics["best_targeted_ctr"] == 100.0
    assert improvement_metrics["improvement_percentage"] == pytest.approx(233.33, rel=1e-2) 