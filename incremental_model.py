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
                    try:
                        # Ensure inputs are properly typed for the base model
                        # Convert to the correct device and dtype consistently
                        u_tensor = u.to(self.dev).long()
                        seq_tensor = seq.to(self.dev).long()
                        pos_tensor = pos.reshape(-1).to(self.dev).long()  # Ensure it's flattened as the model expects
                        neg_tensor = neg.reshape(-1).to(self.dev).long()  # Ensure it's flattened as the model expects
                        
                        # Run the base model with properly prepared inputs
                        base_output = base_model(u_tensor, seq_tensor, pos_tensor, neg_tensor, is_training=False)
                        base_attention = base_output[1][0]  # Get attention weights
                        
                        # Knowledge distillation loss on attention weights
                        distill_temp = getattr(self.args, 'distill_temp', 2.0)
                        distill_alpha = getattr(self.args, 'distill_alpha', 0.5)
                        
                        # Match dimensions for KL divergence
                        # Make sure we're comparing attention weights of the same shape
                        if attention_weights[0].shape == base_attention.shape:
                            distill_loss = F.kl_div(
                                F.log_softmax(attention_weights[0] / distill_temp, dim=-1),
                                F.softmax(base_attention / distill_temp, dim=-1),
                                reduction='batchmean'
                            )
                            
                            loss += distill_alpha * distill_loss
                        else:
                            print(f"Warning: Attention shape mismatch. Current: {attention_weights[0].shape}, Base: {base_attention.shape}")
                            
                    except Exception as e:
                        print(f"Error during knowledge distillation: {e}")
                        # Continue without distillation if it fails
                        pass
                        
            # Return the updated loss and other values
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