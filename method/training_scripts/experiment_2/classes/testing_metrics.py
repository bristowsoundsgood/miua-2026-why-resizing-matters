import pandas as pd

class TestingMetrics():
    
    def __init__(self, loss=0, accuracy=0, fnr=0, fpr=0, npv=0, ppv=0, sensitivity=0, specificity=0, f1_score=0, f2_score=0, mcc=0, auc_roc=0, auc_pr=0, training_time=0):
        self.loss = loss,
        self.accuracy = accuracy,
        self.fnr = fnr,
        self.fpr = fpr,
        self.npv = npv,
        self.ppv = ppv,
        self.sensitivity = sensitivity,
        self.specificity = specificity,
        self.f1_score = f1_score,
        self.f2_score = f2_score,
        self.mcc = mcc,
        self.auc_roc = auc_roc,
        self.auc_pr = auc_pr,
        self.training_time = training_time
        
    def print_performance(self):
        print(f'\n******** PERFORMANCE SUMMARY ********')
        print(f'> Loss: {self.loss}')
        print(f'> Accuracy: {self.accuracy}')
        print(f'> False Negative Rate: {self.fnr}')
        print(f'> False Positive Rate: {self.fpr}')
        print(f'> Negative Predictive Value: {self.npv}')
        print(f'> Positive Predictive Value: {self.ppv}')
        print(f'> Sensitivity: {self.sensitivity}')
        print(f'> Specificity: {self.specificity}')
        print(f'> F1-Score: {self.f1_score}')
        print(f'> F2-Score: {self.f2_score}')
        print(f'> Matthews Correlation Coefficient: {self.mcc}')
        print(f'> AUC-ROC: {self.auc_roc}')
        print(f'> AUC-PR: {self.auc_pr}')
        print(f'> Model Training Time: {self.training_time}')
        
        
    def export_to_csv(self, filename: str = 'testing_metrics.csv') -> None:
        data = {
            'Loss': self.loss,
            'Accuracy': self.accuracy,
            'False Negative Rate': self.fnr,
            'False Positive Rate': self.fpr,
            'Negative Predictive Value': self.npv,
            'Positive Predictive Value': self.ppv,
            'Sensitivity': self.sensitivity,
            'Specificity': self.specificity,
            'F1-Score': self.f1_score,
            'F2-Score': self.f2_score,
            'Matthews Correlation Coefficient': self.mcc,
            'AUC-ROC': self.auc_roc,
            'AUC-PR': self.auc_pr,
            'Model Training Time': self.training_time
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
