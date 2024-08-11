# Customer Churn Prediction

## Overview
This project aims to predict customer churn in a telecommunications company using various machine learning algorithms. The dataset used is the Telco Customer Churn dataset.

## Dataset
The dataset contains customer data, including demographics, account information, and services used. The target variable is `Churn`, which indicates whether a customer has left the company.

## Project Structure
- **Data Preprocessing:** 
  - Handle missing values by replacing blank entries with NaN and dropping rows with missing data.
  - Convert categorical variables into binary or one-hot encoded features.
  - Create new features such as `NumServices` and `TenureMonths` for better predictive power.
  
- **Exploratory Data Analysis:**
  - Visualize the distribution of the target variable `Churn`.
  - Examine correlations and feature importance.

- **Modeling:**
  - Implement various machine learning models including:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Machine (SVM)
  - Evaluate models using metrics such as accuracy, precision, recall, F1 score, ROC curve, and AUC.

## Installation
To run this project, you'll need to install the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
