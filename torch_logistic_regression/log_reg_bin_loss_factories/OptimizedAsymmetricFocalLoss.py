import torch 

class OptimizedAsymmetricFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma_pos=0.5, gamma_neg=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        
        # We store pos_weight as a buffer so it moves to device automatically with the module
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        # 1. Calculate Standard BCE (No weights yet!) to get accurate probabilities
        #    reduction='none' is crucial to keep the shape [batch_size]
        bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Derive true p_t from the raw loss
        pt = torch.exp(-bce_raw)
        
        # 3. Create Masks
        pos_mask = targets > 0.5
        neg_mask = targets <= 0.5
        
        # 4. Initialize Loss Vector
        loss = torch.zeros_like(bce_raw)
        
        # 5. Calculate Asymmetric Focal Terms
        #    (1 - pt) is the "modulating factor"
        
        # Positives: alpha * (1-pt)^gamma_pos * BCE
        # Optional: Apply pos_weight here if strictly necessary
        pos_loss = self.alpha * (1 - pt[pos_mask]) ** self.gamma_pos * bce_raw[pos_mask]
        
        if self.pos_weight is not None:
            pos_loss = pos_loss * self.pos_weight
            
        loss[pos_mask] = pos_loss

        # Negatives: (1-alpha) * (1-pt)^gamma_neg * BCE
        loss[neg_mask] = (1 - self.alpha) * (1 - pt[neg_mask]) ** self.gamma_neg * bce_raw[neg_mask]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss