#!/usr/bin/env python
# coding: utf-8

# ## DATA LOADING AND INSPECTION

# In[21]:


# Importing Pandas library
import pandas as pd


# In[3]:


# Load the dataset
def data_loader(file_path):
    data = pd.read_csv(file_path)
    return data


# In[4]:


# Load the data
file_path = "great_customers.csv"
data = data_loader(file_path)


# In[5]:


# Inspect the data
print(data.head())


# In[6]:


print(data.info())


# ## STEP 2: Data Cleaning

# In[7]:


# Data Cleaning
def data_cleaning(data):
    # Handle missing values (if any)
    data.dropna(inplace=True)
    
    # Remove duplicate rows (if any)
    data.drop_duplicates(inplace=True)
    
    # Check and handle outliers
    
    return data


# In[8]:


# Perform data cleaning
data = data_cleaning(data)


# ## STEP 3: FEATURE SELECTION

# In[11]:


# Feature Selection
def feature_selection(data):
    # Select the relevant features
    selected_features = data[['age', 'salary', 'education_rank', 'marital-status', 'occupation', 'sex', 'tea_per_year', 'coffee_per_year']]

    
    return selected_features

# Perform feature selection
selected_features = feature_selection(data)


# ## STEP 4: PREPROCESSING

# In[31]:


# Importing Numpy library:
import numpy as np
# Preprocessing

def preprocess_data(data):
    # Handle categorical variables using one-hot encoding
    categorical_cols = ['marital-status', 'occupation']
    numeric_cols = ['age', 'salary', 'education_rank', 'tea_per_year', 'coffee_per_year']
    
    # Encode categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data_encoded = encoder.fit_transform(data[categorical_cols])
    
    # Select only the numeric columns for imputation
    numeric_data = data[numeric_cols]
    
    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    
    # Combine encoded categorical features and imputed numeric features
    processed_data = np.concatenate([categorical_data_encoded, numeric_data_imputed], axis=1)
    
    return processed_data


# In[27]:


# Preprocessing
def preprocess_data(data):
    # Handle categorical variables using one-hot encoding
    categorical_cols = ['marital-status', 'occupation']
    numeric_cols = ['age', 'salary', 'education_rank', 'tea_per_year', 'coffee_per_year']
    
    # Encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_data_encoded = encoder.fit_transform(data[categorical_cols])
    
    # Select only the numeric columns for imputation
    numeric_data = data[numeric_cols]
    
    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    
    # Check if any of the arrays are empty
    if categorical_data_encoded.shape[0] == 0:
        processed_data = numeric_data_imputed
    elif numeric_data_imputed.shape[0] == 0:
        processed_data = categorical_data_encoded
    else:
        # Combine encoded categorical features and imputed numeric features
        processed_data = np.concatenate([categorical_data_encoded, numeric_data_imputed], axis=1)
    
    return processed_data


# In[32]:


# Preprocess the data
processed_data = preprocess_data(selected_features)


# ## STEP 5: MODEL BUILDING

# In[45]:


# Importing the relevant files for model building:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np


# In[46]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_data, data['great_customer_class'], test_size=0.2, random_state=42)


# In[47]:


# Split the data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(selected_features, data['great_customer_class'], test_size=0.2, random_state=42)


# In[49]:


# Define a dictionary with models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

# Evaluate models
model_metrics = {
    'Model': [],
    'Precision': [],
    'ROC-AUC': [],
    'Accuracy': [],
    'F1-Score': []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    model_metrics['Model'].append(name)
    model_metrics['Precision'].append(precision)
    model_metrics['ROC-AUC'].append(roc_auc)
    model_metrics['Accuracy'].append(accuracy)
    model_metrics['F1-Score'].append(f1)


# ## STEP 6: ENSEMBLE LEARNING TECHNIQUE

# In[51]:


# Apply ensemble learning (Voting)
ensemble.fit(X_train, y_train)
ensemble_predictions = ensemble.predict(X_test)
ensemble_precision = precision_score(y_test, ensemble_predictions)
ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_f1 = f1_score(y_test, ensemble_predictions)

model_metrics['Model'].append('Ensemble (Voting)')
model_metrics['Precision'].append(ensemble_precision)
model_metrics['ROC-AUC'].append(ensemble_roc_auc)
model_metrics['Accuracy'].append(ensemble_accuracy)
model_metrics['F1-Score'].append(ensemble_f1)


# ## STEP 7: Metric to evaluate your prediction model
# 

# In[53]:


# Create a DataFrame to store the results
results_df = pd.DataFrame(model_metrics)

# Save the results to a CSV file
results_df.to_csv('model_results.csv', index=False)
print("Results saved to model_results.csv")


# In[ ]:




