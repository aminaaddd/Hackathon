from .clean_data import clean_data, prepare_features
from .model import train_model_classification, train_model_with_weights
from .evaluate_model import evaluate_model
import pandas as pd
import joblib


df = pd.read_csv("Prediction_model/player_injuries_impact.csv")

if __name__ == "__main__":
    cleaned_df = clean_data(df)
    X, y = prepare_features(cleaned_df)
    
    y_pred, y_test, clf, X_test = train_model_classification(X, y)
    y_pred_weights, y_test_weights, clf_weights, X_test_weights, X_train_weights = train_model_with_weights(X, y)
    
    #cm, cr, cr_prob, precision, recall = evaluate_model(y_test, y_pred, clf, X_test)
    cm_weights, cr_weights = evaluate_model(y_test_weights, y_pred_weights, clf_weights, X_test_weights)
    
    #print("Confusion Matrix:\n", cm)
    #print("\nClassification Report:\n", cr)
    #print("\nClassification Report with custom threshold:\n", cr_prob)
    
    print("\nClassification Report:\n", cr_weights)

    
    
    #print("\nClassification Report with custom threshold:\n", cr_prob_weights)
    
    #plot_precision_recall_curve(precision_weights, recall_weights)
    
    
    # Save the model
    joblib.dump(clf_weights, "model/injuries_prediction.pkl")
    joblib.dump(X_train_weights.columns, "model/train_columns.pkl")
