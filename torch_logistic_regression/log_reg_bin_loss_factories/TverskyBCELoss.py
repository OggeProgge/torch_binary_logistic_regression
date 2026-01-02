import torch
from log_reg_bin_loss_factories.TverskyLoss import TverskyLoss  # NOTE: assumes same directory

class ComboLoss(torch.nn.Module):

    """
    Combo Loss: Weighted combination of BCE and Tversky.
    Stabilizes the gradient (via BCE) while optimizing for overlap (via Tversky).
    """
    def __init__(self, bce_weight=1.0, tversky_weight=1.0, alpha=0.3, beta=0.7, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        
        # Handle pos_weight for BCE part if provided
        pw_tensor = torch.tensor([pos_weight]) if pos_weight else None
        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)


    def forward(self, logits, targets):
        # We need to make sure the pos_weight tensor is on the same device as inputs
        if self.bce_loss.pos_weight is not None:
             self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits.device)

        loss_bce = self.bce_loss(logits, targets)
        loss_tversky = self.tversky_loss(logits, targets)
        
        return (self.bce_weight * loss_bce) + (self.tversky_weight * loss_tversky)