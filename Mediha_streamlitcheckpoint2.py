

import pandas as pd
import numpy as np
#from pandas_profiling import ProfileReport

# Load dataset
data = pd.read_csv('/content/Financial_inclusion_dataset.csv')

# Display general information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Create a pandas profiling report
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
profile.to_file("report.html")

# Handle missing values
data.fillna(method='ffill', inplace=True)  # Forward fill, you can choose another method if appropriate

# Check for missing values again
print(data.isnull().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Verify removal
print(data.duplicated().sum())

# Assuming numerical columns are identified
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Use z-score to identify outliers
from scipy import stats
z_scores = np.abs(stats.zscore(data[numerical_cols]))

# Remove outliers
data = data[(z_scores < 3).all(axis=1)]

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Perform one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Define features and target
X = data.drop("bank_account_Yes", axis=1)  # Replace 'target_column' with the actual target column name
y = data["bank_account_Yes"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save the model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
