import pandas as pd

class TrainingMetrics:
    """Tracks and exports training/validation metrics across epochs."""
    
    def __init__(self):
        # Initialize all metrics as lists
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.val_fnr = []
        self.val_fpr = []
        self.val_npv = []
        self.val_ppv = []
        self.val_sensitivity = []
        self.val_specificity = []
        self.val_f1_score = []
        self.val_f2_score = []
        self.val_mcc = []
        self.val_auc_roc = []
        self.val_auc_pr = []
        
    def update_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
        val_fnr: float,
        val_fpr: float,
        val_npv: float,
        val_ppv: float,
        val_sensitivity: float,
        val_specificity: float,
        val_f1_score: float,
        val_f2_score: float,
        val_mcc: float,
        val_auc_roc: float,
        val_auc_pr: float
    ) -> None:
        """Update all metrics for a single epoch."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_accuracy)
        self.val_accuracy.append(val_accuracy)
        self.val_fnr.append(val_fnr)
        self.val_fpr.append(val_fpr)
        self.val_npv.append(val_npv)
        self.val_ppv.append(val_ppv)
        self.val_sensitivity.append(val_sensitivity)
        self.val_specificity.append(val_specificity)
        self.val_f1_score.append(val_f1_score)
        self.val_f2_score.append(val_f2_score)
        self.val_mcc.append(val_mcc)
        self.val_auc_roc.append(val_auc_roc)
        self.val_auc_pr.append(val_auc_pr)

    def export_to_csv(self, filename: str = 'training_metrics.csv') -> None:
        """Export all metrics to a CSV file."""
        data = {
            'Epoch': list(range(1, len(self.train_loss) + 1)),
            'Train Loss': self.train_loss,
            'Validation Loss': self.val_loss,
            'Train Accuracy': self.train_accuracy,
            'Validation Accuracy': self.val_accuracy,
            'F1 Score': self.val_f1_score,
            'F2 Score': self.val_f2_score,
            'False Negative Rate': self.val_fnr,
            'False Positive Rate': self.val_fpr,
            'Negative Predictive Value': self.val_npv,
            'Positive Predictive Value': self.val_ppv,
            'Sensitivity': self.val_sensitivity,
            'Specificity': self.val_specificity,
            'Matthews Correlation Coefficient': self.val_mcc,
            'AUC-ROC': self.val_auc_roc,
            'AUC-PR': self.val_auc_pr,
        }
        pd.DataFrame(data).to_csv(filename, index=False)