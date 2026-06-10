import torch
from torch import nn

class ViTWithCustomClassifier(nn.Module):
    """Adds a trainable classifier head to a ViT model.
       This is useful for models provided by Google's Big Transfer, such as ViT-Tiny/16-21K, ViT-Base/16-21K, and ViT-Large/16-21K. 
    """
    
    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.classifier = classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        class_token = outputs.last_hidden_state[:, 0, :]    # As per the huggingface documentation, classifier is run on the [CLS] token from the last hidden state (located at 0 index)
        logits = self.classifier(class_token)
        return logits
