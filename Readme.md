# House Price Prediction

A machine learning project for predicting house prices using the California Housing dataset.

This project was built as a learning-focused ML project to practice the complete workflow from data exploration and preprocessing to model evaluation and comparison.

## Project Overview

The goal of this project is to predict `median_house_value` using housing and location-related features.

Two regression models were implemented and compared:

- Linear Regression
- Random Forest Regressor

Random Forest achieved significantly better performance and was selected as the final model.

## Final Results

### Linear Regression

- MAE: ~49,450
- RMSE: ~68,843
- R²: ~0.638

### Random Forest

- `max_depth = 20`
- MAE: ~31,579
- RMSE: ~48,758
- R²: ~0.819

## What I Learned

This project was mainly focused on understanding and implementing core machine learning concepts, including:

- Train / Test Split
- Data Leakage
- Missing Value Imputation
- One-Hot Encoding
- Feature Scaling
- Feature Engineering
- Polynomial Features
- Multicollinearity and VIF
- Linear Regression
- Random Forest
- Overfitting and Underfitting
- Cross-Validation
- Model Evaluation using MAE, RMSE and R²

## Project Structure

```text
House Price Prediction/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   ├── exploration.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── train.py
│   └── metrics.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
