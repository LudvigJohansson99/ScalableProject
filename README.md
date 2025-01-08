# ScalableProject: WTI Crude Oil Price Prediction

## Introduction

This project focuses on developing a predictive model for daily crude oil prices, particularly West Texas Intermediate (WTI) prices, using data from Alpha Vantage’s WTI API. The project aims to analyze historical trends, leverage machine learning techniques for forecasting, and provide a user interface for visualization. 

### Objectives

1. Retrieve daily crude oil price data from Alpha Vantage’s WTI API.
2. Build and evaluate a machine learning model to forecast WTI crude oil prices.
3. Develop a web-based application to display historical data and predictions.
4. Ensure dynamic updates with new data from the API.

---

## Dataset

The primary data source is the Alpha Vantage WTI API, providing daily crude oil price data. The data is derived from the U.S. Energy Information Administration and accessed via the Federal Reserve Bank of St. Louis (FRED). 

Sample data :
```
"2024-12-10": 71.27,
"2024-12-11": 70.30,
"2024-12-12": 67.73,
"2024-12-13": 69.07,
"2024-12-14": 70.60,
"2024-12-15": 69.89,
"2024-12-16": 70.68
```

---

## Methodology

### Data Handling
1. **Fetching Data**: Historical daily crude oil prices are retrieved using the Alpha Vantage API through a script (`data.py`).
2. **Data Cleaning**: Missing values are handled using forward fill (`ffill`), and the data is sorted by date.
3. **Storage**: Processed data is saved in `oil_prices.json`.

### Model Development
1. **Pre-trained Model Selection**: A time-series transformer model supporting regression tasks is fine-tuned.
2. **Training Configuration**: 
   - **Context Length**: 40 days
   - **Prediction Length**: 7 days
   - **Lags**: Days 1-7
   - **Architecture**: 6 encoder and decoder layers with 128 feed-forward dimension each.
3. **Training Execution**: Implemented in `update_predictions.py` with a loss function optimized for 30,000 epochs.

### Dynamic Updates
1. **GitHub Actions**: Automated workflows (`updatePredictions.yml` and `update-hf-space.yml`) are configured to update predictions and synchronize the Hugging Face Space.

---

## Results

### Model Predictions
A sample of future predictions:
- **2024-12-17**: 68.62
- **2024-12-18**: 70.80
- **2024-12-19**: 70.10

### Observations
The model seems to predict a lower price than what the true price ends up at more often than higher

### Visualization
The Gradio-based UI in Hugging Face Spaces provides a comprehensive view of historical trends and forecasted data.

---

## How to Run the Code

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
2. **Fetch and Process Data**:
   Execute `data.py` to retrieve and preprocess the data.
3. **Run Predictions**:
   Use `update_predictions.py` to train and generate predictions.
4. **Automate Updates**:
   Ensure workflows in `updatePredictions.yml` and `update-hf-space.yml` are correctly configured.
