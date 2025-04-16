"""
Utilities for email campaign analysis.
"""
from utils.analysis import (
    analyze_factor_performance,
    analyze_segment_performance,
    analyze_timing_performance,
    calculate_campaign_metrics,
    estimate_targeting_improvement,
)
from utils.data_loader import (
    create_output_dir,
    engineer_features,
    load_datasets,
    prepare_data,
    prepare_train_test_data,
)
from utils.modeling import (
    evaluate_model,
    get_feature_importance,
    load_model,
    save_model,
    train_model,
)
from utils.visualization import (
    plot_campaign_metrics,
    plot_confusion_matrix,
    plot_factor_performance,
    plot_feature_importance,
    plot_roc_curve,
    plot_targeting_improvement,
    setup_visualization_style,
)

__all__ = [
    # Data loading
    "load_datasets",
    "prepare_data",
    "engineer_features",
    "prepare_train_test_data",
    "create_output_dir",
    
    # Analysis
    "calculate_campaign_metrics",
    "analyze_factor_performance",
    "analyze_segment_performance",
    "analyze_timing_performance",
    "estimate_targeting_improvement",
    
    # Modeling
    "train_model",
    "evaluate_model",
    "get_feature_importance",
    "save_model",
    "load_model",
    
    # Visualization
    "setup_visualization_style",
    "plot_campaign_metrics",
    "plot_factor_performance",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_targeting_improvement",
] 