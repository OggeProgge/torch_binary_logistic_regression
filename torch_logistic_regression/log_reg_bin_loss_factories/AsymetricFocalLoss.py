import torch 

class AsymmetricFocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma_pos=0.5, gamma_neg=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Robust masking for floats
        pos_mask = targets > 0.5
        neg_mask = targets <= 0.5
        
        focal_loss = torch.zeros_like(bce_loss)
        focal_loss[pos_mask] = self.alpha * (1 - pt[pos_mask]) ** self.gamma_pos * bce_loss[pos_mask]
        focal_loss[neg_mask] = (1 - self.alpha) * (1 - pt[neg_mask]) ** self.gamma_neg * bce_loss[neg_mask]
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss