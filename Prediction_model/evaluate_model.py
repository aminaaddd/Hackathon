from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import numpy as np

def evaluate_model(y_test, y_pred, clf, X_test):

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["No next injury","Next injury"])

    #pos_idx = np.where(clf.classes_ == 1)[0][0]
    #y_score = clf.predict_proba(X_test)[:, pos_idx]

    #y_pred_custom = (y_score >= 0.38).astype(int)
    #cr_prob = classification_report(y_test, y_pred_custom, zero_division=0)

    #precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    #avg_precision = average_precision_score(y_test, y_score)
    

    return cm, cr

