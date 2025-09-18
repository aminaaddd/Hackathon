import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def train_model_classification(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return y_pred, y_test, clf, X_test

#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Worse","Stable","Better"]))


def train_model_with_weights(
    X,
    y,
    test_size= 0.2,
    random_state= 42,
    weight_pos= 10, 
    n_estimators= 280):
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=test_size, random_state=random_state, stratify=y
    )

    # OPTION A (simple): class_weight explicite
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight={0: 1.0, 1: float(weight_pos)}, 
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return y_pred, y_test, clf, X_test, X_train

