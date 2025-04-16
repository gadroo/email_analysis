"""
Analysis utilities for email campaign performance.
"""
from typing import Dict, List, Tuple

import pandas as pd


def calculate_campaign_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall email campaign performance metrics.
    
    Args:
        df: DataFrame with opened and clicked flags
        
    Returns:
        Dictionary containing campaign metrics
    """
    total_emails = len(df)
    total_opened = df["opened"].sum()
    total_clicked = df["clicked"].sum()
    
    open_rate = total_opened / total_emails * 100
    click_rate = total_clicked / total_emails * 100
    click_to_open_rate = (total_clicked / total_opened * 100 
                          if total_opened > 0 else 0)
    
    return {
        "total_emails": total_emails,
        "total_opened": total_opened,
        "total_clicked": total_clicked,
        "open_rate": open_rate,
        "click_rate": click_rate,
        "click_to_open_rate": click_to_open_rate
    }


def analyze_factor_performance(
    df: pd.DataFrame, 
    factor: str
) -> pd.DataFrame:
    """
    Analyze performance metrics grouped by a specific factor.
    
    Args:
        df: DataFrame with opened and clicked flags
        factor: Column name to group by
        
    Returns:
        DataFrame with performance metrics by factor
    """
    performance = df.groupby(factor).agg({
        "opened": ["count", "sum", lambda x: x.sum() / x.count() * 100],
        "clicked": ["sum", lambda x: x.sum() / x.count() * 100]
    }).reset_index()
    
    performance.columns = [
        factor, "total", "opened", "open_rate", "clicked", "click_rate"
    ]
    
    return performance.sort_values("click_rate", ascending=False)


def analyze_segment_performance(
    df: pd.DataFrame, 
    segment_cols: List[str],
    min_segment_size: int = 100
) -> pd.DataFrame:
    """
    Analyze performance metrics for segments based on multiple factors.
    
    Args:
        df: DataFrame with opened and clicked flags
        segment_cols: List of column names to create segments
        min_segment_size: Minimum number of emails for a valid segment
        
    Returns:
        DataFrame with performance metrics by segment
    """
    # Create segment identifier
    df = df.copy()
    df["segment"] = df[segment_cols].astype(str).agg("_".join, axis=1)
    
    # Analyze segment performance
    segment_performance = df.groupby("segment").agg({
        "opened": ["count", "sum", lambda x: x.sum() / x.count() * 100],
        "clicked": ["sum", lambda x: x.sum() / x.count() * 100]
    }).reset_index()
    
    segment_performance.columns = [
        "segment", "total", "opened", "open_rate", "clicked", "click_rate"
    ]
    
    # Filter by minimum segment size
    valid_segments = segment_performance[
        segment_performance["total"] >= min_segment_size
    ].sort_values("click_rate", ascending=False)
    
    return valid_segments


def analyze_timing_performance(
    df: pd.DataFrame, 
    min_sample_size: int = 50
) -> pd.DataFrame:
    """
    Analyze performance metrics for different timing combinations.
    
    Args:
        df: DataFrame with opened and clicked flags
        min_sample_size: Minimum sample size for timing combinations
        
    Returns:
        DataFrame with performance metrics by time
    """
    timing_performance = df.groupby(["weekday", "hour"]).agg({
        "opened": ["count", "sum", lambda x: x.sum() / x.count() * 100],
        "clicked": ["sum", lambda x: x.sum() / x.count() * 100]
    }).reset_index()
    
    timing_performance.columns = [
        "weekday", "hour", "total", "opened", "open_rate", "clicked", "click_rate"
    ]
    
    # Filter by minimum sample size
    valid_timing = timing_performance[
        timing_performance["total"] >= min_sample_size
    ].sort_values("click_rate", ascending=False)
    
    return valid_timing


def estimate_targeting_improvement(
    y_test: pd.Series, 
    y_prob: pd.Series, 
    current_ctr: float,
    percentiles: List[int] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Estimate potential improvement from targeted email sending.
    
    Args:
        y_test: Actual values from test set
        y_prob: Predicted probabilities from model
        current_ctr: Current click-through rate
        percentiles: List of percentiles to evaluate (default: [10, 20, ..., 100])
        
    Returns:
        Tuple containing:
            - DataFrame with CTR at different targeting thresholds
            - Dictionary with improvement metrics
    """
    if percentiles is None:
        percentiles = list(range(10, 101, 10))
    
    # Combine actual values and predicted probabilities
    test_results = pd.DataFrame({
        "actual": y_test,
        "probability": y_prob
    }).sort_values("probability", ascending=False)
    
    # Calculate CTR for different targeting percentiles
    targeted_ctrs = []
    
    for p in percentiles:
        threshold = int(len(test_results) * p / 100)
        top_p = test_results.iloc[:threshold]
        targeted_ctr = top_p["actual"].mean() * 100
        targeted_ctrs.append(targeted_ctr)
    
    # Create results DataFrame
    targeting_results = pd.DataFrame({
        "percentile": percentiles,
        "ctr": targeted_ctrs
    })
    
    # Calculate improvement metrics
    best_percentile = percentiles[targeting_results["ctr"].argmax()]
    best_targeted_ctr = targeting_results["ctr"].max()
    improvement = (best_targeted_ctr - current_ctr) / current_ctr * 100
    
    improvement_metrics = {
        "current_ctr": current_ctr,
        "best_percentile": best_percentile,
        "best_targeted_ctr": best_targeted_ctr,
        "improvement_percentage": improvement
    }
    
    return targeting_results, improvement_metrics 