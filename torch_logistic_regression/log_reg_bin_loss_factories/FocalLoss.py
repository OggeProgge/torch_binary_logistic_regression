import torch

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for dense object detection and imbalanced binary classification.
    Lowers the loss for well-classified examples, focusing training on 'hard' negatives.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha      # Weighting factor for the rare class (range 0-1)
        self.gamma = gamma      # Focusing parameter (higher = focuses more on hard examples)
        self.reduction = reduction

    def forward(self, logits, targets):

        # BCEWithLogitsLoss handles numerical stability better than manual Sigmoid + Log
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Get probabilities to calculate the "modulating factor" (pt)
        pt = torch.exp(-bce_loss) 
        
        # Focal term: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss