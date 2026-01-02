import torch

class JaccardLoss(torch.nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        TP = (probs * targets).sum()
        # Optimization: Union = Sum(A) + Sum(B) - Intersection
        union = probs.sum() + targets.sum() - TP
        iou = (TP + self.smooth) / (union + self.smooth)
        return 1 - iou