#!/usr/bin/env python3
"""
Email Campaign Analysis

This script analyzes email campaign performance and builds a predictive model
to optimize future email campaigns.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from utils import (
    analyze_factor_performance,
    analyze_segment_performance,
    analyze_timing_performance,
    calculate_campaign_metrics,
    create_output_dir,
    engineer_features,
    estimate_targeting_improvement,
    evaluate_model,
    get_feature_importance,
    load_datasets,
    plot_campaign_metrics,
    plot_confusion_matrix,
    plot_factor_performance,
    plot_feature_importance,
    plot_roc_curve,
    plot_targeting_improvement,
    prepare_data,
    prepare_train_test_data,
    save_model,
    setup_visualization_style,
    train_model,
)


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_campaign_performance(df, output_dir, config):
    """Analyze overall campaign performance."""
    logging.info("Analyzing overall campaign performance...")
    
    # Calculate overall metrics
    metrics = calculate_campaign_metrics(df)
    
    # Log metrics
    logging.info(f"Total emails sent: {metrics['total_emails']}")
    logging.info(f"Total emails opened: {metrics['total_opened']} ({metrics['open_rate']:.2f}%)")
    logging.info(f"Total emails clicked: {metrics['total_clicked']} ({metrics['click_rate']:.2f}%)")
    logging.info(f"Click-to-open rate: {metrics['click_to_open_rate']:.2f}%")
    
    # Plot metrics
    plot_campaign_metrics(metrics, output_dir, config)
    
    return metrics


def analyze_basic_factors(df, output_dir, config):
    """Analyze performance by basic factors (excluding purchase_bucket)."""
    logging.info("Analyzing performance by basic factors...")
    
    # Analyze email text (long vs short)
    logging.info("Analyzing by email text...")
    text_performance = analyze_factor_performance(df, "email_text")
    logging.info(f"\n{text_performance}")
    plot_factor_performance(text_performance, "email_text", output_dir, config)
    
    # Analyze email version (personalized vs generic)
    logging.info("Analyzing by email version...")
    version_performance = analyze_factor_performance(df, "email_version")
    logging.info(f"\n{version_performance}")
    plot_factor_performance(version_performance, "email_version", output_dir, config)
    
    # Analyze hour of day
    logging.info("Analyzing by hour of day...")
    hour_performance = analyze_factor_performance(df, "hour")
    logging.info(f"Top 5 hours by click rate:\n{hour_performance.head()}")
    plot_factor_performance(hour_performance, "hour", output_dir, config)
    
    # Analyze day of week
    logging.info("Analyzing by day of week...")
    day_performance = analyze_factor_performance(df, "weekday")
    logging.info(f"\n{day_performance}")
    plot_factor_performance(day_performance, "weekday", output_dir, config)
    
    # Analyze user country
    logging.info("Analyzing by country...")
    country_performance = analyze_factor_performance(df, "user_country")
    logging.info(f"\n{country_performance}")
    plot_factor_performance(country_performance, "user_country", output_dir, config)
    
    return {
        "text_performance": text_performance,
        "version_performance": version_performance,
        "hour_performance": hour_performance,
        "day_performance": day_performance,
        "country_performance": country_performance,
    }


def analyze_purchase_factor(df, output_dir, config):
    """Analyze performance by purchase bucket."""
    logging.info("Analyzing by purchase history...")
    purchase_performance = analyze_factor_performance(df, "purchase_bucket")
    logging.info(f"\n{purchase_performance}")
    plot_factor_performance(purchase_performance, "purchase_bucket", output_dir, config)
    
    return purchase_performance


def build_predictive_model(df_encoded, output_dir, config):
    """Build and evaluate a predictive model."""
    logging.info("Building predictive model...")
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_encoded, config)
    
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Testing set shape: {X_test.shape}")
    
    # Train model
    logging.info("Training model...")
    start_time = time.time()
    model = train_model(X_train, y_train, config)
    training_time = time.time() - start_time
    logging.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics, report_df, _ = evaluate_model(model, X_test, y_test)
    
    # Log metrics
    logging.info(f"Model performance:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    # Get predictions for visualization
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Get feature importance
    feature_importance = get_feature_importance(model, X_train)
    logging.info(f"Top 10 most important features:\n{feature_importance.head(10)}")
    
    # Plot results
    plot_feature_importance(feature_importance, output_dir, config)
    plot_confusion_matrix(y_test, y_pred, output_dir, config)
    plot_roc_curve(y_test, y_prob, output_dir, config)
    
    # Save model
    model_path = output_dir / "model.joblib"
    save_model(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    return model, metrics, feature_importance, y_test, y_prob


def estimate_improvement(y_test, y_prob, campaign_metrics, output_dir, config):
    """Estimate potential improvement with model-based targeting."""
    logging.info("Estimating potential improvement with targeting...")
    
    # Current CTR
    current_ctr = campaign_metrics["click_rate"]
    
    # Estimate improvement with targeting
    targeting_results, improvement_metrics = estimate_targeting_improvement(
        y_test, y_prob, current_ctr
    )
    
    # Log results
    logging.info(f"Current random CTR: {current_ctr:.2f}%")
    logging.info(
        f"Best targeted CTR (top {improvement_metrics['best_percentile']}%): "
        f"{improvement_metrics['best_targeted_ctr']:.2f}%"
    )
    logging.info(
        f"Potential improvement: {improvement_metrics['improvement_percentage']:.2f}%"
    )
    
    # Plot results
    plot_targeting_improvement(targeting_results, improvement_metrics, output_dir, config)
    
    return targeting_results, improvement_metrics


def analyze_segments(df, output_dir, config):
    """Analyze performance for different segments."""
    logging.info("Analyzing segments...")
    
    # For encoded dataframe, we need to create segments differently
    # Check if the dataframe is one-hot encoded
    is_encoded = ('email_text' not in df.columns and 
                 'email_version' not in df.columns and
                 'user_country' not in df.columns)
    
    if is_encoded:
        logging.info("Using encoded dataframe for segment analysis")
        # Create a categorical column for email_text based on one-hot columns
        if 'email_text_short_email' in df.columns:
            df['email_text'] = df['email_text_short_email'].map({1: 'short_email', 0: 'long_email'})
        
        # Create a categorical column for email_version based on one-hot columns
        if 'email_version_personalized' in df.columns:
            df['email_version'] = df['email_version_personalized'].map({1: 'personalized', 0: 'generic'})
        
        # Create a categorical column for user_country
        country_columns = [col for col in df.columns if col.startswith('user_country_')]
        if country_columns:
            # Create an empty string column
            df['user_country'] = ''
            
            # For each country column, set the value if it's 1
            for col in country_columns:
                country = col.replace('user_country_', '')
                mask = df[col] == 1
                df.loc[mask, 'user_country'] = country
    
    # Create segment analysis with multiple factors
    segment_cols = ["email_text", "email_version", "user_country"]
    min_segment_size = config["analysis"]["min_segment_size"]
    
    segment_analysis = analyze_segment_performance(
        df, segment_cols, min_segment_size
    )
    
    logging.info(f"Top 5 highest performing segments:")
    logging.info(f"\n{segment_analysis.head(5)}")
    
    # Analyze timing combinations
    min_timing_samples = config["analysis"]["min_timing_samples"]
    
    # Timing analysis uses 'weekday' and 'hour', which need to be handled similarly
    if is_encoded:
        # Create a categorical column for weekday based on one-hot columns
        weekday_columns = [col for col in df.columns if col.startswith('weekday_')]
        if weekday_columns:
            df['weekday'] = ''
            for col in weekday_columns:
                day = col.replace('weekday_', '')
                mask = df[col] == 1
                df.loc[mask, 'weekday'] = day
    
    timing_analysis = analyze_timing_performance(
        df, min_timing_samples
    )
    
    logging.info(f"Top 5 best times to send emails:")
    logging.info(f"\n{timing_analysis.head(5)}")
    
    # Save results to CSV
    segment_analysis.to_csv(output_dir / "segment_analysis.csv", index=False)
    timing_analysis.to_csv(output_dir / "timing_analysis.csv", index=False)
    
    return segment_analysis, timing_analysis


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Email Campaign Analysis")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        # Start time
        start_time = time.time()
        logging.info("Starting email campaign analysis...")
        
        # Load configuration
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
        
        # Create output directory
        output_dir = create_output_dir(config)
        logging.info(f"Output directory: {output_dir}")
        
        # Set up visualization style
        setup_visualization_style(config)
        
        # Load datasets
        logging.info("Loading datasets...")
        email_df, opened_df, clicked_df = load_datasets(config)
        logging.info(
            f"Loaded datasets: "
            f"emails={len(email_df)}, "
            f"opened={len(opened_df)}, "
            f"clicked={len(clicked_df)}"
        )
        
        # Prepare data
        logging.info("Preparing data...")
        df = prepare_data(email_df, opened_df, clicked_df)
        
        # Analyze campaign performance
        campaign_metrics = analyze_campaign_performance(df, output_dir, config)
        
        # Analyze basic factors (before feature engineering)
        basic_factor_results = analyze_basic_factors(df, output_dir, config)
        
        # Feature engineering
        logging.info("Performing feature engineering...")
        df_encoded = engineer_features(df, config)
        
        # Now analyze purchase bucket factor (after feature engineering)
        purchase_performance = analyze_purchase_factor(df_encoded, output_dir, config)
        
        # Combine factor results
        factor_results = {**basic_factor_results, "purchase_performance": purchase_performance}
        
        # Save processed data
        df.to_csv(output_dir / "processed_data.csv", index=False)
        logging.info(f"Saved processed data to {output_dir / 'processed_data.csv'}")
        
        # Build predictive model
        model, model_metrics, feature_importance, y_test, y_prob = build_predictive_model(
            df_encoded, output_dir, config
        )
        
        # Estimate improvement
        targeting_results, improvement_metrics = estimate_improvement(
            y_test, y_prob, campaign_metrics, output_dir, config
        )
        
        # Analyze segments
        segment_analysis, timing_analysis = analyze_segments(df_encoded, output_dir, config)
        
        # Save important results
        pd.DataFrame([campaign_metrics]).to_csv(
            output_dir / "campaign_metrics.csv", index=False
        )
        pd.DataFrame([model_metrics]).to_csv(
            output_dir / "model_metrics.csv", index=False
        )
        pd.DataFrame([improvement_metrics]).to_csv(
            output_dir / "improvement_metrics.csv", index=False
        )
        feature_importance.to_csv(
            output_dir / "feature_importance.csv", index=False
        )
        
        # Complete
        execution_time = time.time() - start_time
        logging.info(f"Analysis completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logging.exception(f"Error in analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 