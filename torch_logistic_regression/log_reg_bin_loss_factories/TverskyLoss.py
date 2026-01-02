import torch

class TverskyLoss(torch.nn.Module):
    """
    Tversky Loss - A generalization of Dice Loss.
    Allows adjusting penalties for False Positives (FP) vs False Negatives (FN).
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # Penalty for False Positives
        self.beta = beta    # Penalty for False Negatives (Higher = higher Recall)
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        # Flatten tensors to treat all samples in the batch as a single vector
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # True Positives
        TP = (probs * targets).sum()
        # False Positives
        FP = ((1 - targets) * probs).sum()
        # False Negatives
        FN = (targets * (1 - probs)).sum()
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index
