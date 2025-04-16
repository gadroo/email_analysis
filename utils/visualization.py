"""
Visualization utilities for email campaign analysis.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def setup_visualization_style(config: Dict) -> None:
    """
    Set up the visualization style for consistent plots.
    
    Args:
        config: Configuration dictionary with visualization settings
        
    Returns:
        None
    """
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = config["visualization"]["figsize"]
    plt.rcParams["figure.dpi"] = config["visualization"]["dpi"]
    

def plot_campaign_metrics(
    metrics: Dict[str, float], 
    output_dir: Path,
    config: Dict
) -> None:
    """
    Plot overall campaign metrics.
    
    Args:
        metrics: Dictionary with campaign metrics
        output_dir: Directory to save the plot
        config: Configuration dictionary
        
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=config["visualization"]["figsize"])
    
    # Create barplot for rates
    rate_metrics = {
        "Open Rate": metrics["open_rate"],
        "Click Rate": metrics["click_rate"],
        "Click-to-Open Rate": metrics["click_to_open_rate"]
    }
    
    colors = sns.color_palette(config["visualization"]["palette"], len(rate_metrics))
    bars = ax.bar(rate_metrics.keys(), rate_metrics.values(), color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}%",
            ha="center", 
            va="bottom"
        )
    
    # Customize plot
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Email Campaign Performance Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"campaign_metrics.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close()


def plot_factor_performance(
    performance_df: pd.DataFrame,
    factor: str,
    output_dir: Path,
    config: Dict,
    top_n: int = None
) -> None:
    """
    Plot performance metrics grouped by a specific factor.
    
    Args:
        performance_df: DataFrame with factor performance metrics
        factor: Name of the factor being analyzed
        output_dir: Directory to save the plot
        config: Configuration dictionary
        top_n: Optional number of top factors to show
        
    Returns:
        None
    """
    # Limit to top N if specified
    if top_n is not None and len(performance_df) > top_n:
        df = performance_df.head(top_n)
    else:
        df = performance_df.copy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=config["visualization"]["figsize"],
        gridspec_kw={"width_ratios": [1, 1.5]}
    )
    
    # Plot open rate and click rate
    colors = sns.color_palette(config["visualization"]["palette"], 2)
    
    # Left plot: Rates
    df.plot(
        x=factor, 
        y=["open_rate", "click_rate"], 
        kind="bar", 
        ax=ax1, 
        color=colors
    )
    ax1.set_title(f"Open and Click Rates by {factor}")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xlabel(factor)
    ax1.legend(["Open Rate", "Click Rate"])
    
    # Right plot: Counts with CTR
    ax2_bars = ax2.bar(df[factor], df["total"], alpha=0.7)
    ax2.set_title(f"Email Count and Click Rate by {factor}")
    ax2.set_ylabel("Number of Emails")
    ax2.set_xlabel(factor)
    
    # Add click rate as line on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df[factor], df["click_rate"], "o-", color="red", linewidth=2)
    ax2_twin.set_ylabel("Click Rate (%)", color="red")
    ax2_twin.tick_params(axis="y", colors="red")
    
    # Add count labels
    for bar in ax2_bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center", 
            va="bottom"
        )
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{factor}_performance.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    top_n: int = 20
) -> None:
    """
    Plot feature importance from the model.
    
    Args:
        importance_df: DataFrame with feature importances
        output_dir: Directory to save the plot
        config: Configuration dictionary
        top_n: Number of top features to show
        
    Returns:
        None
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create barplot
    fig, ax = plt.subplots(figsize=config["visualization"]["figsize"])
    bars = ax.barh(
        top_features["feature"][::-1], 
        top_features["importance"][::-1], 
        color=sns.color_palette(config["visualization"]["palette"], 1)
    )
    
    # Customize plot
    ax.set_title(f"Top {top_n} Most Important Features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"feature_importance.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    output_dir: Path,
    config: Dict
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save the plot
        config: Configuration dictionary
        
    Returns:
        None
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=config["visualization"]["figsize"])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        ax=ax,
        cbar=False
    )
    
    # Customize plot
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["Not Clicked", "Clicked"])
    ax.set_yticklabels(["Not Clicked", "Clicked"])
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"confusion_matrix.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close()


def plot_roc_curve(
    y_true: Union[List, np.ndarray, pd.Series],
    y_prob: Union[List, np.ndarray, pd.Series],
    output_dir: Path,
    config: Dict
) -> None:
    """
    Plot ROC curve for model evaluation.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_dir: Directory to save the plot
        config: Configuration dictionary
        
    Returns:
        None
    """
    # Create plot
    fig, ax = plt.subplots(figsize=config["visualization"]["figsize"])
    RocCurveDisplay.from_predictions(
        y_true, 
        y_prob, 
        ax=ax,
        name="Random Forest"
    )
    
    # Add random baseline
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    
    # Customize plot
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"roc_curve.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close()


def plot_targeting_improvement(
    targeting_results: pd.DataFrame,
    improvement_metrics: Dict[str, float],
    output_dir: Path,
    config: Dict
) -> None:
    """
    Plot targeting improvement results.
    
    Args:
        targeting_results: DataFrame with CTR at different targeting thresholds
        improvement_metrics: Dictionary with improvement metrics
        output_dir: Directory to save the plot
        config: Configuration dictionary
        
    Returns:
        None
    """
    # Create plot
    fig, ax = plt.subplots(figsize=config["visualization"]["figsize"])
    
    # Plot targeted CTR by percentile
    ax.plot(
        targeting_results["percentile"], 
        targeting_results["ctr"],
        "o-", 
        linewidth=2,
        markersize=8
    )
    
    # Add horizontal line for current random CTR
    current_ctr = improvement_metrics["current_ctr"]
    ax.axhline(
        current_ctr, 
        color="red", 
        linestyle="--", 
        label=f"Current Random CTR: {current_ctr:.2f}%"
    )
    
    # Highlight best percentile
    best_percentile = improvement_metrics["best_percentile"]
    best_ctr = improvement_metrics["best_targeted_ctr"]
    ax.plot(
        best_percentile, 
        best_ctr, 
        "D", 
        color="green", 
        markersize=10,
        label=f"Best CTR: {best_ctr:.2f}% (Top {best_percentile}%)"
    )
    
    # Add improvement annotation
    improvement = improvement_metrics["improvement_percentage"]
    ax.annotate(
        f"+{improvement:.1f}%",
        xy=(best_percentile, best_ctr),
        xytext=(best_percentile - 5, best_ctr + 2),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=12,
        fontweight="bold",
        backgroundcolor="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black")
    )
    
    # Customize plot
    ax.set_title("Click-Through Rate Improvement with Targeted Sending")
    ax.set_xlabel("Percentage of Users Targeted (%)")
    ax.set_ylabel("Click-Through Rate (%)")
    ax.set_xticks(targeting_results["percentile"])
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        output_dir / f"targeting_improvement.{config['visualization']['save_format']}",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight"
    )
    plt.close() 