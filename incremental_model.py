import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model_v1 import SASRec

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model_v1 import SASRec

class IncrementalSASRec(SASRec):
    """
    Extends SASRec model with incremental learning capabilities
    """
    def __init__(self, usernum, itemnum, args):
        super(IncrementalSASRec, self).__init__(usernum, itemnum, args)
        
        # Additional parameters for incremental learning
        self.distill_temp = getattr(args, 'distill_temp', 2.0)
        self.distill_alpha = getattr(args, 'distill_alpha', 0.5)
        
        # For tracking performance
        self.seen_items = set()
        self.new_items = set()
    
    def forward(self, u, seq, pos, neg, is_training=True, base_model=None):
        """
        Forward pass with optional knowledge distillation
        """
        loss, attention_weights, auc, l2_loss = super().forward(
            u, seq, pos, neg, is_training=is_training)
        
        # Add knowledge distillation if base model is provided and in training mode
        if is_training and base_model is not None:
            # Get base model predictions
            with torch.no_grad():
                seq_emb = base_model.item_emb(seq)
                u_emb = base_model.user_emb(u)
                u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                
                # Get logits from base model
                base_logits = base_model(u, seq, pos, neg, is_training=False)
                
            # Knowledge distillation loss (simplified)
            distill_loss = F.kl_div(
                F.log_softmax(attention_weights[0] / self.distill_temp, dim=-1),
                F.softmax(base_logits[1][0] / self.distill_temp, dim=-1),
                reduction='batchmean'
            )
            
            loss += self.distill_alpha * distill_loss
        
        return loss, attention_weights, auc, l2_loss
    
    def update_item_sets(self, old_items, new_items):
        """
        Update seen item sets for adaptive weighting
        """
        self.seen_items = set(old_items)
        self.new_items = set(new_items)


class JointSASRec(nn.Module):
    """
    Joint recommendation model that combines a frozen base model with 
    a fine-tuned incremental model for improved performance
    """
    def __init__(self, base_model, incremental_model):
        super(JointSASRec, self).__init__()
        self.base_model = base_model
        self.incremental_model = incremental_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reference to device for consistency
        self.dev = incremental_model.dev
        
    def predict(self, u, seq, item_idx, alpha=0.5, adaptive=True):
        """
        Make joint predictions using both models with adaptive weighting
        
        Args:
            u: User indices
            seq: Sequence of items
            item_idx: Items to score
            alpha: Base weight for old model
            adaptive: Whether to use adaptive weighting based on item recency
        """
        # Get predictions from both models
        with torch.no_grad():
            base_scores = self.base_model.predict(u, seq, item_idx)
            incr_scores = self.incremental_model.predict(u, seq, item_idx)
            
            if not adaptive:
                # Static weighting
                combined_scores = alpha * base_scores + (1 - alpha) * incr_scores
            else:
                # Adaptive weighting based on item recency
                weights = []
                for idx in item_idx.cpu().numpy().flatten():
                    if idx in self.incremental_model.new_items:
                        # New item: rely more on incremental model
                        weights.append(0.2)  # 20% base, 80% incremental
                    elif idx in self.incremental_model.seen_items:
                        # Old item: balance both models
                        weights.append(0.6)  # 60% base, 40% incremental
                    else:
                        # Unknown: use equal weighting
                        weights.append(0.5)
                        
                # Convert to tensor and reshape to match scores
                weights = torch.tensor(weights, device=self.dev)
                weights = weights.view(base_scores.shape)
                
                # Weighted combination
                combined_scores = weights * base_scores + (1 - weights) * incr_scores
            
        return combined_scores
        
    def forward(self, u, seq, pos, neg, is_training=True):
        """
        For training, only update incremental model
        """
        return self.incremental_model(u, seq, pos, neg, is_training, self.base_model)