import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Embedding, MultiHeadAttention, FeedForward, LayerNorm

class SASRec(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init__()
        
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args  # Store args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The hidden units are now just the item embedding dimension
        self.hidden_units = args.item_hidden_units
        
        # Item and position embeddings
        self.item_emb = Embedding(itemnum + 1, args.item_hidden_units, zero_pad=True, scale=True)
        self.pos_emb = Embedding(args.maxlen, args.item_hidden_units, zero_pad=False, scale=False)
        
        # No user embedding needed
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.hidden_units, 
                             num_heads=args.num_heads,
                             dropout_rate=args.dropout_rate,
                             causality=True)
            for _ in range(args.num_blocks)
        ])
        
        # Feed forward layers
        self.feed_forward_layers = nn.ModuleList([
            FeedForward(self.hidden_units,
                       num_units=[self.hidden_units, self.hidden_units],
                       dropout_rate=args.dropout_rate)
            for _ in range(args.num_blocks)
        ])
        
        self.last_layernorm = LayerNorm(self.hidden_units)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # For prediction
        self.maxlen = args.maxlen
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, u, seq, pos, neg, is_training=True):
        # Create masks
        mask = (seq > 0)  # This will create the mask on the same device as seq
        batch_size = seq.size(0)
        
        # Get embeddings
        seq_emb = self.item_emb(seq)  # [B, L, H_item]
        
        # Position encoding
        pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(self.dev)
        pos_ids = pos_ids.unsqueeze(0).expand_as(seq)  # [B, L]
        pos_emb = self.pos_emb(pos_ids)  # [B, L, H]
        
        # Combine embeddings (no user embedding to add)
        seq = seq_emb + pos_emb
        
        # Apply dropout
        if is_training:
            seq = self.dropout(seq)
        
        # Mask padding
        mask_float = mask.unsqueeze(-1).float()
        seq = seq * mask_float
        
        attention_weights = []
        
        # Self-attention blocks
        for i in range(len(self.attention_layers)):
            seq = self.last_layernorm(seq)
            seq, attention = self.attention_layers[i](seq, seq, is_training=is_training)
            attention_weights.append(attention)
            
            seq = self.feed_forward_layers[i](seq, is_training=is_training)
            seq = seq * mask_float
        
        seq = self.last_layernorm(seq)
        
        # Reshape for prediction
        seq = seq.reshape(-1, self.hidden_units)  # [B*L, H]
        pos = pos.reshape(-1)  # [B*L]
        neg = neg.reshape(-1)  # [B*L]
        
        # Get embeddings for positive and negative items
        pos_emb = self.item_emb(pos)  # [B*L, H_item]
        neg_emb = self.item_emb(neg)  # [B*L, H_item]
        
        # No user embeddings to add
        
        # Compute logits
        pos_logits = torch.sum(pos_emb * seq, dim=-1)  # [B*L]
        neg_logits = torch.sum(neg_emb * seq, dim=-1)  # [B*L]
        
        # Compute loss
        istarget = (pos > 0).float()  # [B*L]
        
        loss = torch.sum(
            -(torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget) -
            (torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget)
        ) / torch.sum(istarget)
        
        # Add L2 regularization for embeddings
        l2_loss = 0
        if hasattr(self, 'args') and self.args.l2_emb > 0:
            l2_loss = self.args.l2_emb * (
                torch.sum(self.item_emb.embedding.weight ** 2) +
                torch.sum(self.pos_emb.embedding.weight ** 2)
                # No user embedding regularization
            )
            loss += l2_loss
        
        # Compute AUC for training info
        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        return loss, attention_weights, auc, l2_loss if hasattr(self, 'args') and self.args.l2_emb > 0 else 0.0
    
    def predict(self, u, seq, item_idx):
        # Set evaluation mode
        self.eval()
        
        with torch.no_grad():
            mask = (seq > 0)  # This will create the mask on the same device as seq
            batch_size = seq.size(0)
            
            # Get embeddings
            seq_emb = self.item_emb(seq)
            
            # Position encoding
            pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(self.dev)
            pos_ids = pos_ids.unsqueeze(0).expand_as(seq)
            pos_emb = self.pos_emb(pos_ids)
            
            # Combine embeddings (no user embedding to add)
            seq = seq_emb + pos_emb
            
            # Mask padding
            if mask is not None:
                seq = seq * mask.unsqueeze(-1).float()
            
            # Self-attention blocks
            for i in range(len(self.attention_layers)):
                seq = self.last_layernorm(seq)
                seq, _ = self.attention_layers[i](seq, seq, is_training=False)
                seq = self.feed_forward_layers[i](seq, is_training=False)
                if mask is not None:
                    seq = seq * mask.unsqueeze(-1).float()
            
            seq = self.last_layernorm(seq)
            
            # Get sequence representation from last position
            seq_emb = seq[:, -1, :]  # [B, H]
            
            # Get test item embeddings
            test_item_emb = self.item_emb(item_idx)  # [N_items, H_item]
            
            # No user embeddings to add
            
            # Compute scores (make them negative since we want target items to have higher scores)
            # Reshape test_item_emb to match batch dimension
            test_item_emb = test_item_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_items, H]
            scores = -torch.sum(seq_emb.unsqueeze(1) * test_item_emb, dim=-1)  # [B, N_items]
            
        # Set back to training mode
        self.train()
        
        return scores    

