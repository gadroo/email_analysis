"""
Data loading and preparation utilities for email campaign analysis.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_datasets(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw datasets from CSV files.
    
    Args:
        config: Configuration dictionary containing file paths
        
    Returns:
        Tuple containing email_df, opened_df, and clicked_df DataFrames
    """
    data_paths = config["data"]
    
    email_df = pd.read_csv(data_paths["email_table"])
    opened_df = pd.read_csv(data_paths["opened_table"])
    clicked_df = pd.read_csv(data_paths["clicked_table"])
    
    return email_df, opened_df, clicked_df


def prepare_data(
    email_df: pd.DataFrame, 
    opened_df: pd.DataFrame, 
    clicked_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare the raw data by adding flags for opened and clicked emails.
    
    Args:
        email_df: DataFrame containing email information
        opened_df: DataFrame containing opened email IDs
        clicked_df: DataFrame containing clicked email IDs
        
    Returns:
        Prepared DataFrame with opened and clicked flags
    """
    # Add binary flags for opened and clicked emails
    email_df = email_df.copy()
    email_df["opened"] = email_df["email_id"].isin(opened_df["email_id"]).astype(int)
    email_df["clicked"] = email_df["email_id"].isin(clicked_df["email_id"]).astype(int)
    
    return email_df


def engineer_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Create purchase buckets
    bins = config["features"]["purchase_bins"]
    labels = config["features"]["purchase_labels"]
    
    df["purchase_bucket"] = pd.cut(
        df["user_past_purchases"], 
        bins=bins,
        labels=labels
    )
    
    # Convert hour to cyclical features to better capture time periodicity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # One-hot encode categorical variables
    categorical_cols = config["features"]["categorical_cols"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    return df_encoded


def prepare_train_test_data(
    df: pd.DataFrame, 
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare features and target variables and split into training and testing sets.
    
    Args:
        df: Input DataFrame with engineered features
        config: Configuration dictionary
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test
    """
    # Prepare features and target
    X = df.drop(["email_id", "hour", "opened", "clicked", "purchase_bucket"], axis=1)
    y = df["clicked"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=config["analysis"]["test_size"],
        random_state=config["analysis"]["random_state"]
    )
    
    return X_train, X_test, y_train, y_test


def create_output_dir(config: Dict) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path object for the output directory
    """
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir 