import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, matthews_corrcoef, roc_auc_score, average_precision_score

def calculate_metrics(y_probs: np.ndarray, y_preds: np.ndarray, y_truths: np.ndarray):

    conf_mat = confusion_matrix(y_true=y_truths, y_pred=y_preds)
    tn, fp, fn, tp = conf_mat.ravel()             

    return {
        "accuracy": accuracy_score(y_true=y_truths, y_pred=y_preds),
        "fnr": fn / (fn + tp),
        "fpr": fp / (fp + tn),
        "npv": tn / (tn + fn),
        "ppv": tp / (tp + fp),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1_score": fbeta_score(y_true=y_truths, y_pred=y_preds, beta=1),
        "f2_score": fbeta_score(y_true=y_truths, y_pred=y_preds, beta=2),
        "mcc": matthews_corrcoef(y_true=y_truths, y_pred=y_preds),
        "auc_roc": roc_auc_score(y_true=y_truths, y_score=y_probs),
        "auc_pr": average_precision_score(y_true=y_truths, y_score=y_probs)
    }  

