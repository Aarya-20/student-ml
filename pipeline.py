# ML Pipeline (Simple Version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Preprocessing
def preprocess():
    print("Step 1: Data Preprocessing")
    data = pd.read_csv("data.csv")
    X = data[['Age', 'Salary']]
    y = data['Purchased']
    return train_test_split(X, y, test_size=0.2)

# Step 2: Training
def train(X_train, y_train):
    print("Step 2: Model Training")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

# Step 3: Evaluation
def evaluate(model, X_test, y_test):
    print("Step 3: Model Evaluation")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Pipeline execution
X_train, X_test, y_train, y_test = preprocess()
model = train(X_train, y_train)
evaluate(model, X_test, y_test)