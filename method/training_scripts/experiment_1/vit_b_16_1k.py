import os
import pathlib

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchinfo import summary

from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, matthews_corrcoef, roc_auc_score, average_precision_score

from tqdm import tqdm
from timeit import default_timer as timer

from classes.training_metrics import TrainingMetrics
from classes.testing_metrics import TestingMetrics
from classes.early_stopper import EarlyStopper
from classes.warmup_cosine_scheduler import WarmupCosineScheduler

## HYPERPARAMETERS ## 
N_FOLDS = 5
MAX_EPOCHS = 100
WARMUP_EPOCHS = 5
BATCH_SIZE = 32
LR_MAXIMUM = 1e-4
LR_MINIMUM = 1e-5
RANDOM_SEED = 29
# N_THREADS = 3

# torch.set_num_threads(N_THREADS)

## 1. LOAD MODEL ##
pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
preprocess_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VIT_B_16_1K = torchvision.models.vit_b_16(weights=pretrained_weights)

print(f'******* RAW PRE-TRAINED MODEL *******')
summary(model=VIT_B_16_1K,
        input_size=(1, 3, 224, 224),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

## 2.LOAD DATASET - ENTER YOUR OWN DIRECTORIES  ## 
dir_train_dataset = ""
dir_val_dataset = ""
dir_test_dataset = ""

TRAIN_DATASET = ImageFolder(root=dir_train_dataset, transform=preprocess_transform)
VAL_DATASET = ImageFolder(root=dir_val_dataset, transform=preprocess_transform)
TEST_DATASET = ImageFolder(root=dir_test_dataset, transform=preprocess_transform)

train_dataloader = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False)

## 3. TWEAK MODEL FOR FINE-TUNING ## 
for param in VIT_B_16_1K.parameters():
    param.requires_grad = False
    
# Change classifier head to suit problem domain (1 class prediction 0->1)
torch.manual_seed(RANDOM_SEED)
new_classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=1)
)
VIT_B_16_1K.heads = new_classifier

print(f'******* ADAPTED MODEL FOR FINE-TUNING *******')
summary(model=VIT_B_16_1K,
        input_size=(1, 3, 224, 224),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

# Set up class weighting for loss function. Addresses class imbalance.
class_weighting = compute_class_weight(class_weight='balanced', classes=np.unique(TRAIN_DATASET.targets), y=TRAIN_DATASET.targets)
positive_weighting = torch.tensor([class_weighting[1] / class_weighting[0]])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weighting)                         

optimiser = torch.optim.Adam(params=VIT_B_16_1K.parameters(), lr=LR_MAXIMUM)
lr_scheduler = WarmupCosineScheduler(optimiser=optimiser, warmup_epochs=WARMUP_EPOCHS, max_epochs=MAX_EPOCHS, max_lr=LR_MAXIMUM, min_lr=LR_MINIMUM)

CURRENT_DIR = pathlib.Path(__file__).parent
checkpoint_path = os.path.join(CURRENT_DIR, "trained_checkpoints", "vit_b_16_1k.pt")
early_stopper = EarlyStopper(path=checkpoint_path, patience=10, delta=0.001, verbose=True)


training_metrics = TrainingMetrics()        # For tracking how validation performance improves throughout training  
testing_metrics = TestingMetrics()

start_time = timer()
for epoch in tqdm(range(MAX_EPOCHS), desc='TRAINING'):
    
    # Train model
    VIT_B_16_1K.train()
    train_loss = 0
    train_preds = []
    train_truths = []
    
    for X_train, y_train in train_dataloader:
        y_train = y_train.unsqueeze(dim=1).float()
        y_logits = VIT_B_16_1K(X_train)
        
        loss = loss_fn(y_logits, y_train)
        train_loss += loss.item()        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        y_probs = torch.sigmoid(y_logits)
        y_preds = torch.round(y_probs)
        
        train_preds.extend(y_preds.detach().numpy())
        train_truths.extend(y_train.detach().numpy())
        
    # After all batches, lightly track performance
    train_loss /= len(train_dataloader)                 # Average loss per epoch
    train_accuracy = accuracy_score(y_true=train_truths, y_pred=train_preds) 
    
    lr_scheduler.step(epoch)
    print(f'Learning Rate: {optimiser.param_groups[0]['lr']}')
    
    # Validate model    
    VIT_B_16_1K.eval()
    val_loss = 0
    val_probs = []
    val_preds = []
    val_truths = []
    
    with torch.inference_mode():        
        for X_val, y_val in val_dataloader:
            y_val = y_val.unsqueeze(dim=1).float()
            y_logits = VIT_B_16_1K(X_val)
            
            loss = loss_fn(y_logits, y_val)
            val_loss += loss.item()
            
            y_probs = torch.sigmoid(y_logits)
            y_preds = torch.round(y_probs)
        
            val_probs.extend(y_probs.numpy())
            val_preds.extend(y_preds.numpy())
            val_truths.extend(y_val.numpy())
            
        # After all batches, track performance thoroughly for plotting (val metrics also determine model selection)
        val_loss /= len(val_dataloader)
        val_accuracy = accuracy_score(y_true=val_truths, y_pred=val_preds)
        val_f1_score = fbeta_score(y_true=val_truths, y_pred=val_preds, beta=1)
        val_f2_score = fbeta_score(y_true=val_truths, y_pred=val_preds, beta=2) 
                    
        val_conf_mat = confusion_matrix(y_true=val_truths, y_pred=val_preds)
        tn, fp, fn, tp = val_conf_mat.ravel()             
        val_fnr = fn / (fn + tp)
        val_fpr = fp / (fp + tn)
        val_npv = tn / (tn + fn)
        val_ppv = tp / (tp + fp)
        val_sensitivity = tp / (tp + fn)
        val_specificity = tn / (tn + fp)
        val_mcc = matthews_corrcoef(y_true=val_truths, y_pred=val_preds)
        val_auc_roc = roc_auc_score(y_true=val_truths, y_score=val_probs)
        val_auc_pr = average_precision_score(y_true=val_truths, y_score=val_probs)
        
        training_metrics.update_epoch(
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            val_fnr,
            val_fpr,
            val_npv,
            val_ppv,
            val_sensitivity,
            val_specificity,
            val_f1_score,
            val_f2_score,
            val_mcc,
            val_auc_roc,
            val_auc_pr
        )
        
        # Print out loss every 5 epochs
        if epoch % 5 == 0:
            print(f'\nEPOCH: {epoch}')
            print(f'Train Loss: {train_loss}')
            print(f'Validation Loss: {val_loss}')
        
        # Check for early stopping (if model is overfitting)
        if early_stopper.check_early_stop(model=VIT_B_16_1K, val_loss=val_loss) == True:
            print(f'EARLY STOPPING: training stopped after {epoch} epochs')       
            break
        
end_time = timer()
training_time = end_time - start_time
        
# After training, load the weights from the best version of the model (i.e. before it began overfitting)
VIT_B_16_1K.load_state_dict(state_dict=torch.load(early_stopper.path))
        
## 5. TEST MODEL ##
VIT_B_16_1K.eval()
with torch.inference_mode():
    
    test_loss = 0
    test_probs = []
    test_preds = []     
    test_truths = []
    
    for X_test, y_test in tqdm(test_dataloader, desc='TESTING'):
        y_test = y_test.unsqueeze(dim=1).float()
        y_logits = VIT_B_16_1K(X_test)
        
        loss = loss_fn(y_logits, y_test)
        test_loss += loss.item()
        
        y_probs = torch.sigmoid(y_logits)
        y_preds = torch.round(y_probs)
        
        test_probs.extend(y_probs.numpy())      # Accumulate prediction probabilities, class predictions, and ground truths as they come. Used to calculate performance metrics.
        test_preds.extend(y_preds.numpy())
        test_truths.extend(y_test.numpy())
        
    # Calculate performance metrics after all predictions made
    test_loss /= len(test_dataloader)
    accuracy = accuracy_score(y_true=test_truths, y_pred=test_preds)            
    f1_score = fbeta_score(y_true=test_truths, y_pred=test_preds, beta=1)
    f2_score = fbeta_score(y_true=test_truths, y_pred=test_preds, beta=2) 
                
    conf_mat = confusion_matrix(y_true=test_truths, y_pred=test_preds)
    tn, fp, fn, tp = conf_mat.ravel()             
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true=test_truths, y_pred=test_preds)
    auc_roc = roc_auc_score(y_true=test_truths, y_score=test_probs)
    auc_pr = average_precision_score(y_true=test_truths, y_score=test_probs)
        
    testing_metrics = TestingMetrics(test_loss, accuracy, fnr, fpr, npv, ppv, sensitivity, specificity, f1_score, f2_score, mcc, auc_roc, auc_pr, training_time)
        
torch.save(obj=VIT_B_16_1K.state_dict(), f='trained_models/vit_b_16_1k.pt')

testing_metrics.print_performance()
testing_metrics.export_to_csv(filename='results/metrics_test_vit_b_16_1k')
training_metrics.export_to_csv(filename='results/metrics_train_vit_b_16_1k')