# Social Media Ads Prediction using Logistic Regression

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Starting program...")

# Load dataset
try:
    data = pd.read_csv("data.csv")
    print("Data loaded successfully")
except Exception as e:
    print("Error loading data:", e)
    exit()

# Check columns
print("Columns in dataset:", data.columns)

# Features and target
try:
    X = data[['Age', 'Salary']]
    y = data['Purchased']
    print("Features extracted")
except Exception as e:
    print("Column error:", e)
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Data split done")

# -------- CHANGE THIS PART FOR VERSION --------
VERSION = 2   # change to 1 or 2

if VERSION == 1:
    print("Training Model Version 1")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pickle.dump(model, open("model_v1.pkl", "wb"))

elif VERSION == 2:
    print("Training Model Version 2")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    pickle.dump(model, open("model_v2.pkl", "wb"))

else:
    print("Invalid VERSION selected. Use 1 or 2.")
    exit()

print("Model trained successfully")

# Prediction
try:
    y_pred = model.predict(X_test)
    print("Prediction done")
except Exception as e:
    print("Prediction error:", e)
    exit()

# Accuracy
try:
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
except Exception as e:
    print("Accuracy calculation error:", e)