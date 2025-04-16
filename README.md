# Email Marketing Campaign Analysis

This project analyzes the performance of an email marketing campaign and builds a predictive model to optimize future campaigns. The analysis aims to answer the following questions:

1. What percentage of users opened the email and what percentage clicked on the link within the email?
2. How can we optimize future email campaigns to maximize the probability of users clicking on the link inside the email?
3. By how much would a targeted approach improve click-through rates compared to random sending?
4. Are there any interesting patterns in how the email campaign performed for different segments of users?

## Project Structure

```
.
├── config.yaml               # Configuration parameters
├── main.py                   # Main script to run the analysis
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py           # Package initialization
│   ├── analysis.py           # Analysis functions
│   ├── data_loader.py        # Data loading and preparation
│   ├── modeling.py           # Model training and evaluation
│   └── visualization.py      # Visualization functions
└── output/                   # Output directory (created on run)
    ├── campaign_metrics.csv  # Overall campaign metrics
    ├── model_metrics.csv     # Model performance metrics
    ├── feature_importance.csv# Feature importance rankings
    ├── processed_data.csv    # Processed dataset
    ├── segment_analysis.csv  # Segment performance analysis
    ├── timing_analysis.csv   # Timing performance analysis
    ├── model.joblib          # Saved model
    └── *_performance.png     # Performance visualization plots
```

## Data Description

The analysis uses three datasets:

1. **email_table.csv**: Information about each email sent
   - email_id: Unique identifier for each email
   - email_text: Email content type (long_email or short_email)
   - email_version: Personalization type (personalized or generic)
   - hour: Local time when the email was sent (0-23)
   - weekday: Day of the week when the email was sent
   - user_country: Recipient's country
   - user_past_purchases: Number of previous purchases by the recipient

2. **email_opened_table.csv**: IDs of emails that were opened at least once
   - email_id: Unique identifier for opened emails

3. **link_clicked_table.csv**: IDs of emails whose link was clicked at least once
   - email_id: Unique identifier for emails with clicked links

## Requirements

- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`):
  - pandas==2.1.4
  - numpy==1.26.3
  - matplotlib==3.8.2
  - seaborn==0.13.1
  - scikit-learn==1.3.2
  - pyyaml==6.0.1
  - joblib==1.3.2

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd email-marketing-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python main.py
   ```

   Optional arguments:
   - `--config`: Path to configuration file (default: `config.yaml`)
   - `--debug`: Enable debug logging

4. Check the results in the `output/` directory.

## Configuration

You can customize the analysis by modifying the `config.yaml` file:

- **Data paths**: Locations of input and output files
- **Analysis parameters**: Minimum sample sizes, random state, etc.
- **Model parameters**: Algorithm type and hyperparameters
- **Feature engineering**: Categorical columns and bucketing settings
- **Visualization settings**: Plot formats, sizes, and styles

## Analysis Workflow

1. **Data Preparation**:
   - Load the three datasets and combine them
   - Add binary flags for opened and clicked emails
   - Perform feature engineering (categorical encoding, cyclic features for time)

2. **Exploratory Analysis**:
   - Calculate overall campaign metrics
   - Analyze performance by various factors

3. **Predictive Modeling**:
   - Train a Random Forest classifier to predict email clicks
   - Evaluate model performance
   - Identify important features

4. **Targeting Improvement**:
   - Estimate potential CTR improvement with targeted sending
   - Compare with current random approach

5. **Segment Analysis**:
   - Identify high-performing segments
   - Analyze optimal timing combinations

## Results

The analysis produces the following key outputs:

- **Campaign metrics**: Overall open rate, click rate, and click-to-open rate
- **Factor performance**: How different factors affect email performance
- **Model metrics**: Performance of the predictive model
- **Feature importance**: Ranking of factors that influence click-through rates
- **Targeting improvement**: Estimated CTR improvement with targeted sending
- **Segment analysis**: Performance metrics for different user segments
- **Timing analysis**: Optimal times to send emails 