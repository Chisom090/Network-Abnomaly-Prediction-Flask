# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:03:07 2024

@author: chisom
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Handling missing values in 'attack' and 'last_flag' columns in test dataset
columns_null = ['attack', 'last_flag']
imputer = SimpleImputer(strategy='most_frequent')
test[columns_null] = imputer.fit_transform(test[columns_null])

# Handling missing values in train dataset
null_columns = train.columns[train.isnull().any()]
columns_to_drop = list(null_columns)
imputer = SimpleImputer(strategy='most_frequent')
train[columns_to_drop] = imputer.fit_transform(train[columns_to_drop])

# Selecting object columns for training
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = ['num_compromised', 'root_shell', 'su_attempted', 'num_root', 
                      'num_file_creations', 'num_shells', 'num_access_files', 
                      'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
                      'count', 'srv_count', 'serror rate', 'srv_serror_rate', 
                      'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                      'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
                      'dst_host_srv_count', 'dst_host_same_srv_rate', 
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                      'dst_host_srv_rerror_rate']

# Separating features and target variable
X_train = train[categorical_features + numerical_features]
y_train = train['attack']
X_test = test[categorical_features + numerical_features]
y_test = test['attack']

# Print shapes for debugging
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Print column names for debugging
print("X_train columns:", X_train.columns)
print("X_test columns:", X_test.columns)

# Preprocessing pipelines for both numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Creating the final pipeline with preprocessing and KNN classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=4))
])

# Training the model
model_pipeline.fit(X_train, y_train)

# Saving the model pipeline
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Saving the preprocessor separately
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Making predictions (model score was 70 percent; which is okay)
y_pred = model_pipeline.predict(X_test)

# Evaluating the model 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
