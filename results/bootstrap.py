import numpy as np

from math import floor
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

from math import floor
from sklearn.utils import resample
from calculate_metrics import calculate_metrics

def bootstrap(y_probs: np.ndarray, y_preds: np.ndarray, y_truths: np.ndarray, n_resample: int = 10000, p_value: float = 0.05, random_seed: int = 29):
    np.random.seed(random_seed)
    
    distributions = {
        "accuracy": [],
        "fnr": [],
        "fpr": [],
        "npv": [],
        "ppv": [],
        "sensitivity": [],
        "specificity": [],
        "f1_score": [],
        "f2_score": [],
        "mcc": [],
        "auc_roc": [],
        "auc_pr": []
    }

    for n in range(n_resample):
        # (Re)sample with replacement
        sample_y_probs, sample_y_preds, sample_y_truths = resample(y_probs, y_preds, y_truths, stratify=y_truths)

        # Calculate metrics
        metrics = calculate_metrics(y_probs=sample_y_probs, y_preds=sample_y_preds, y_truths=sample_y_truths)

        # Append to individual distributions
        distributions["accuracy"].append(metrics["accuracy"])
        distributions["fnr"].append(metrics["fnr"])
        distributions["fpr"].append(metrics["fpr"])
        distributions["npv"].append(metrics["npv"])
        distributions["ppv"].append(metrics["ppv"])
        distributions["sensitivity"].append(metrics["sensitivity"])
        distributions["specificity"].append(metrics["specificity"])
        distributions["f1_score"].append(metrics["f1_score"])
        distributions["f2_score"].append(metrics["f2_score"])
        distributions["mcc"].append(metrics["mcc"])
        distributions["auc_roc"].append(metrics["auc_roc"])
        distributions["auc_pr"].append(metrics["auc_pr"])

    # Sort the distributions
    for d in distributions.values():
        d.sort()

    # Calculate confidence intervals for each metric
    lower_index = floor(n_resample * (p_value / 2))
    upper_index = floor(n_resample * (1 - (p_value / 2)))
    
    intervals = {
        "accuracy": [distributions["accuracy"][lower_index], distributions["accuracy"][upper_index]],
        "fnr": [distributions["fnr"][lower_index], distributions["fnr"][upper_index]],
        "fpr": [distributions["fpr"][lower_index], distributions["fpr"][upper_index]],
        "npv": [distributions["npv"][lower_index], distributions["npv"][upper_index]],
        "ppv": [distributions["ppv"][lower_index], distributions["ppv"][upper_index]],
        "sensitivity": [distributions["sensitivity"][lower_index], distributions["sensitivity"][upper_index]],
        "specificity": [distributions["specificity"][lower_index], distributions["specificity"][upper_index]],
        "f1_score": [distributions["f1_score"][lower_index], distributions["f1_score"][upper_index]],
        "f2_score": [distributions["f2_score"][lower_index], distributions["f2_score"][upper_index]],
        "mcc": [distributions["mcc"][lower_index], distributions["mcc"][upper_index]],
        "auc_roc": [distributions["auc_roc"][lower_index], distributions["auc_roc"][upper_index]],
        "auc_pr": [distributions["auc_pr"][lower_index], distributions["auc_pr"][upper_index]]
    }

    return distributions, intervals