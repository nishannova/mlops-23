import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, metrics
from utils import preprocess_data
import os

os.makedirs("q2_models", exist_ok=True)

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def save_model(model, filename):
    joblib.dump(model, filename)

# Read and preprocess the dataset
X, y = read_digits()
X = preprocess_data(X)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

print(f"*"*50)
print(f"[Q2] ANSWER IS FOLLOWING:\n\n")
for solver in solvers:
    # Initialize and train logistic regression model
    clf = LogisticRegression(solver=solver, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(f"\t", "*"*25)
    print(f"\t Accuracy with solver '{solver}': {accuracy}")

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\t Mean and Std with solver with 5-CROSS CV '{solver}': {cv_scores.mean()}, {cv_scores.std()}")
    print(f"\t", "*"*25, "\n")
    # Save the model
    filename = f"m22aie208_lr_{solver}.joblib"
    save_model(clf, os.path.join("q2_models", filename))
print(f"\n\n[Q2] END OF ANSWER")
print(f"*"*50)