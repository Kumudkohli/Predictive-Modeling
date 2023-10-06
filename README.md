# Predictive-Modeling-
This repository contains code for building a predictive model using Python and scikit-learn library. The model aims to predict whether a customer is a "great customer" based on various features.

## Table of Contents
1. [Data Loading and Inspection](#data-loading-and-inspection)
2. [Data Cleaning](#data-cleaning)
3. [Feature Selection](#feature-selection)
4. [Preprocessing](#preprocessing)
5. [Model Building](#model-building)
6. [Ensemble Learning Technique](#ensemble-learning-technique)
7. [Metric to Evaluate Your Prediction Model](#metric-to-evaluate-your-prediction-model)

## Data Loading and Inspection
### Importing Pandas library
```python
import pandas as pd
```
## Load the dataset
```python
def data_loader(file_path):
    data = pd.read_csv(file_path)
    return data

# Load the data
file_path = "great_customers.csv"
data = data_loader(file_path)
def data_loader(file_path):
    data = pd.read_csv(file_path)
    return data

# Load the data
file_path = "great_customers.csv"
data = data_loader(file_path)
```
## Inspect the data
```python
print(data.head())
print(data.info())
```

