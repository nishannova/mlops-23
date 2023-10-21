import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def tune_model(model, parameters, X_train, y_train):
    print(f"BEGINING TO TUNE: {model}")
    import time

    start = time.time()
    scorer = make_scorer(f1_score, average='macro')
    clf = GridSearchCV(model, parameters, scoring=scorer, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"TUNING MODEL TOOK: {time.time() - start} Secs")
    return clf.best_estimator_


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    return predictions, accuracy, f1

def main():
    X_train, X_test, y_train, y_test = load_data()

    prod_model = SVC()
    prod_parameters = {'kernel': ['rbf'], 'C': [1, 10]}
    print(f"SVC MDOEL AND PARAMETERS SET")

    cand_model = DecisionTreeClassifier()
    cand_parameters = {'criterion': ['gini'], 'max_depth': [None, 10]}
    print(f"DECISION TREE MDOEL AND PARAMETERS SET")

    best_prod_model = tune_model(prod_model, prod_parameters, X_train, y_train)
    
    best_cand_model = tune_model(cand_model, cand_parameters, X_train, y_train)

    prod_predictions, prod_accuracy, prod_f1 = evaluate_model(best_prod_model, X_test, y_test)
    cand_predictions, cand_accuracy, cand_f1 = evaluate_model(best_cand_model, X_test, y_test)

    confusion_inter = confusion_matrix(prod_predictions, cand_predictions, labels=np.unique(y_test))
    confusion_2x2 = confusion_matrix(prod_predictions == y_test, cand_predictions == y_test)

    print("SVC model's accuracy: ", prod_accuracy)
    print("DECISION TREE model's accuracy: ", cand_accuracy)
    print("Confusion Matrix between models' predictions: \n", confusion_inter)
    print("2x2 Confusion Matrix: \n", confusion_2x2)
    print("SVC model's F1 score: ", prod_f1)
    print("DECISION TREE model's F1 score: ", cand_f1)

if __name__ == "__main__":
    main()
