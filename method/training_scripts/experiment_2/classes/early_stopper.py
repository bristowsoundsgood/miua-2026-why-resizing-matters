import torch

# Custom EarlyStopping class from scratch
class EarlyStopper:
    def __init__(self, path, patience=8, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_val_loss = None
        self.no_improvement_count = 0
    
    def check_early_stop(self, model, val_loss):
        
        if self.best_val_loss is None or val_loss < self.best_val_loss - self.delta:
            self.no_improvement_count = 0
            
            self.best_val_loss = val_loss
            
            # Save checkpoint if improvement observed
            torch.save(model.state_dict(), self.path)            
            if self.verbose:
                print(f"Model improved - checkpoint saved at loss {val_loss:.4f}")       
       
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                print("Early stopping triggered.")
                return True  # Signal to stop training
        return False