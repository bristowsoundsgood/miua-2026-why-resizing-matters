import torch

class WarmupCosineScheduler:
    def __init__(self, optimiser, warmup_epochs, max_epochs, max_lr, min_lr):
        self.optimiser = optimiser
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Create cosine scheduler for post-warmup
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max_epochs - warmup_epochs, eta_min=min_lr
        )
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr
        else:
            # Use cosine annealing
            self.cosine_scheduler.step()