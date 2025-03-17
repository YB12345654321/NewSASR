import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model_v1 import SASRec

class PromptBank(nn.Module):
    """
    A bank of learnable prompts that capture different user behavior patterns
    """
    def __init__(self, prompt_dim, num_prompts=8):
        super(PromptBank, self).__init__()
        # Initialize learnable prompt vectors
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, query_embedding):
        """
        Select relevant prompts based on query embedding
        
        Args:
            query_embedding: Embedding used to query the prompt bank
            
        Returns:
            Selected prompt embeddings
        """
        # Calculate similarity between query and prompts
        similarity = torch.matmul(query_embedding, self.prompts.T) / self.temperature
        
        # Apply softmax to get attention weights
        attention = F.softmax(similarity, dim=-1)
        
        # Get weighted combination of prompts
        selected_prompts = torch.matmul(attention, self.prompts)
        
        return selected_prompts, attention


class PromptBaseSASRec(SASRec):
    """
    Extends SASRec model with prompt-based incremental learning
    """
    def __init__(self, usernum, itemnum, args):
        super(PromptBaseSASRec, self).__init__(usernum, itemnum, args)
        
        # Define prompt dimensions
        prompt_dim = args.item_hidden_units + args.user_hidden_units
        self.num_prompts = getattr(args, 'num_prompts', 8)
        
        # Create prompt bank
        self.prompt_bank = PromptBank(prompt_dim, self.num_prompts)
        
        # Importance weights for each prompt (for continual learning)
        self.register_buffer('prompt_importance', torch.zeros(self.num_prompts, device=self.dev))
        
        # For tracking performance
        self.seen_items = set()
        self.new_items = set()
        
        # Parameters for continual learning
        self.prompt_mix_ratio = getattr(args, 'prompt_mix_ratio', 0.3)
        
    def forward(self, u, seq, pos, neg, is_training=True, base_model=None):
        """
        Forward pass with prompt augmentation
        """
        # Get mask for valid sequence items
        mask = (seq > 0)
        batch_size = seq.size(0)
        
        # Get embeddings
        seq_emb = self.item_emb(seq)  # [B, L, H_item]
        
        # Position encoding
        pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(self.dev)
        pos_ids = pos_ids.unsqueeze(0).expand_as(seq)  # [B, L]
        pos_emb = self.pos_emb(pos_ids)  # [B, L, H_total]
        
        # User encoding
        u_emb = self.user_emb(u)  # [B, H_user]
        u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B, L, H_user]
        
        # Combine embeddings
        seq = torch.cat([seq_emb, u_emb_expand], dim=-1)  # [B, L, H_total]
        seq += pos_emb
        
        # Get a summary of user's sequence for prompt selection
        # Use the last position in the sequence as query vector
        valid_positions = mask.sum(dim=1) - 1  # Get index of last valid position
        valid_positions = torch.clamp(valid_positions, min=0)  # Ensure non-negative
        
        batch_indices = torch.arange(batch_size).to(self.dev)
        query_vectors = seq[batch_indices, valid_positions]
        
        # Select prompts based on the sequence representation
        selected_prompts, prompt_weights = self.prompt_bank(query_vectors)
        
        # Update prompt importance if in training mode
        if is_training:
            with torch.no_grad():
                self.prompt_importance += prompt_weights.sum(dim=0)
        
        # Apply dropout
        if is_training:
            seq = self.dropout(seq)
            selected_prompts = self.dropout(selected_prompts)
        
        # Mask padding
        mask_float = mask.unsqueeze(-1).float()
        seq = seq * mask_float
        
        # Add prompts to the sequence representation
        # Mix the prompts with the sequence representation at each position
        selected_prompts_expand = selected_prompts.unsqueeze(1).expand(-1, seq.size(1), -1)
        seq = seq * (1 - self.prompt_mix_ratio) + selected_prompts_expand * self.prompt_mix_ratio
        
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
        
        # Add user embeddings
        user_emb = u_emb_expand.reshape(-1, u_emb.size(-1))  # [B*L, H_user]
        pos_emb = torch.cat([pos_emb, user_emb], dim=1)  # [B*L, H_total]
        neg_emb = torch.cat([neg_emb, user_emb], dim=1)  # [B*L, H_total]
        
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
                torch.sum(self.user_emb.embedding.weight ** 2) +
                torch.sum(self.pos_emb.embedding.weight ** 2) +
                torch.sum(self.prompt_bank.prompts ** 2)  # Also regularize prompts
            )
            loss += l2_loss
        
        # Compute AUC for training info
        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        # Add knowledge distillation if base model is provided and in training mode
        if is_training and base_model is not None:
            # Get base model predictions
            with torch.no_grad():
                # Run only the encoding part of the base model
                base_seq_emb = base_model.item_emb(seq)
                base_u_emb = base_model.user_emb(u)
                base_u_emb_expand = base_u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                
                # Get logits from base model
                base_output = base_model(u, seq, pos, neg, is_training=False)
                base_attention = base_output[1][0]  # Get attention weights
            
            # Knowledge distillation loss on attention weights
            distill_temp = getattr(self.args, 'distill_temp', 2.0)
            distill_alpha = getattr(self.args, 'distill_alpha', 0.5)
            
            distill_loss = F.kl_div(
                F.log_softmax(attention_weights[0] / distill_temp, dim=-1),
                F.softmax(base_attention / distill_temp, dim=-1),
                reduction='batchmean'
            )
            
            loss += distill_alpha * distill_loss
        
        return loss, attention_weights, auc, l2_loss
    
    def predict(self, u, seq, item_idx):
        """
        Make predictions with prompt enhancement
        """
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
            
            # User encoding
            u_emb = self.user_emb(u)  # [B, H_user]
            u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B, L, H_user]
            
            # Combine embeddings
            seq_repr = torch.cat([seq_emb, u_emb_expand], dim=-1)  # [B, L, H_total]
            seq_repr += pos_emb
            
            # Get a summary of user's sequence for prompt selection
            # Use the last position in the sequence as query vector
            valid_positions = mask.sum(dim=1) - 1  # Get index of last valid position
            valid_positions = torch.clamp(valid_positions, min=0)  # Ensure non-negative
            
            batch_indices = torch.arange(batch_size).to(self.dev)
            query_vectors = seq_repr[batch_indices, valid_positions]
            
            # Select prompts based on the sequence representation
            selected_prompts, _ = self.prompt_bank(query_vectors)
            
            # Mask padding
            if mask is not None:
                seq_repr = seq_repr * mask.unsqueeze(-1).float()
            
            # Add prompts to the sequence representation
            # Mix the prompts with the sequence representation at each position
            selected_prompts_expand = selected_prompts.unsqueeze(1).expand(-1, seq_repr.size(1), -1)
            seq_repr = seq_repr * (1 - self.prompt_mix_ratio) + selected_prompts_expand * self.prompt_mix_ratio
            
            # Self-attention blocks
            for i in range(len(self.attention_layers)):
                seq_repr = self.last_layernorm(seq_repr)
                seq_repr, _ = self.attention_layers[i](seq_repr, seq_repr, is_training=False)
                seq_repr = self.feed_forward_layers[i](seq_repr, is_training=False)
                if mask is not None:
                    seq_repr = seq_repr * mask.unsqueeze(-1).float()
            
            seq_repr = self.last_layernorm(seq_repr)
            
            # Get sequence representation from last position
            seq_emb = seq_repr[:, -1, :]  # [B, H_total]
            
            # Get test item embeddings
            test_item_emb = self.item_emb(item_idx)  # [N_items, H_item]
            
            # Add user embeddings to test items
            test_u_emb = u_emb.unsqueeze(1).expand(-1, test_item_emb.size(0), -1)  # [B, N_items, H_user]
            test_item_emb = torch.cat([
                test_item_emb.unsqueeze(0).expand(batch_size, -1, -1),  # [B, N_items, H_item]
                test_u_emb  # [B, N_items, H_user]
            ], dim=-1)  # [B, N_items, H_total]
            
            # Compute scores (make them negative since we want target items to have higher scores)
            scores = -torch.sum(seq_emb.unsqueeze(1) * test_item_emb, dim=-1)  # [B, N_items]
            
        # Set back to training mode
        self.train()
        
        return scores
        
    def update_item_sets(self, old_items, new_items):
        """
        Update seen item sets for adaptive weighting
        """
        self.seen_items = set(old_items)
        self.new_items = set(new_items)
        
    def freeze_important_prompts(self, importance_threshold=0.7):
        """
        Create gradient masks for prompt parameters to prevent updates to important prompts
        
        Args:
            importance_threshold: Threshold for determining which prompts to freeze
        """
        # Normalize importance scores
        if torch.sum(self.prompt_importance) > 0:
            normalized_importance = self.prompt_importance / torch.sum(self.prompt_importance)
            
            # Identify prompts to keep frozen (ensure mask is on the right device)
            keep_mask = normalized_importance > importance_threshold
            
            # Apply hooks to parameters
            def hook(grad, mask):
                # Ensure mask is on the same device as grad
                mask = mask.to(grad.device)
                return grad * (~mask).float().unsqueeze(1)
            
            # Register hook
            self.prompt_bank.prompts.register_hook(
                lambda grad: hook(grad, keep_mask)
            )
    def train_with_separate_prompt_phases(self, train_data, valid_data, args, device):
        """
        Train the model with separate phases for base model and prompts
        """
        # Create samplers and optimizers
        t1_sampler = WarpSampler(train_data, args.usernum, args.itemnum,
                            batch_size=args.batch_size, maxlen=args.maxlen,
                            threshold_user=args.threshold_user,
                            threshold_item=args.threshold_item,
                            n_workers=3, device=device)
        
        # Phase 1: Train base model with frozen prompts
        print("=== Phase 1: Training base model with frozen prompts ===")
        # Freeze prompt parameters
        for param in self.prompt_bank.parameters():
            param.requires_grad = False
            
        # Create optimizer for non-prompt parameters
        phase1_optimizer = torch.optim.Adam(
            [p for n, p in self.named_parameters() if 'prompt_bank' not in n],
            lr=args.lr, betas=(0.9, 0.98)
        )
        
        # Disable prompt mixing during phase 1
        original_mix_ratio = self.prompt_mix_ratio
        self.prompt_mix_ratio = 0.0
        
        # Train for half the epochs
        num_batch = max(len(train_data) // args.batch_size, 1)
        phase1_epochs = args.num_epochs // 2
        
        for epoch in range(1, phase1_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = t1_sampler.next_batch()
                
                phase1_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase1_optimizer.step()
                
            if epoch % args.print_freq == 0:
                t_test = evaluate(self, [train_data, valid_data, {}, args.usernum, args.itemnum], args, device)
                print(f"[Phase 1 epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}")
        
        # Phase 2: Train prompts with frozen base model
        print("=== Phase 2: Training prompts with frozen base model ===")
        # Freeze non-prompt parameters
        for name, param in self.named_parameters():
            if 'prompt_bank' not in name:
                param.requires_grad = False
        
        # Unfreeze prompt parameters
        for param in self.prompt_bank.parameters():
            param.requires_grad = True
        
        # Restore prompt mixing
        self.prompt_mix_ratio = original_mix_ratio
        
        # Create optimizer for prompt parameters only
        phase2_optimizer = torch.optim.Adam(
            self.prompt_bank.parameters(),
            lr=args.lr, betas=(0.9, 0.98)
        )
        
        # Train for remaining epochs
        phase2_epochs = args.num_epochs - phase1_epochs
        
        for epoch in range(1, phase2_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = t1_sampler.next_batch()
                
                phase2_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase2_optimizer.step()
                
            if epoch % args.print_freq == 0:
                t_test = evaluate(self, [train_data, valid_data, {}, args.usernum, args.itemnum], args, device)
                print(f"[Phase 2 epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}")
        
        # Close sampler
        t1_sampler.close()
        
        # Calculate prompt importance
        self.calculate_prompt_importance(valid_data, args, device)
        
    def calculate_prompt_importance(self, valid_data, args, device):
        """
        Calculate prompt importance based on validation performance
        """
        print("=== Calculating prompt importance ===")
        self.eval()
        
        # First, measure baseline performance
        baseline_ndcg, _ = evaluate(self, [valid_data, valid_data, {}, args.usernum, args.itemnum], args, device)
        
        # Initialize importance scores
        importance_scores = torch.zeros(self.num_prompts, device=self.dev)
        
        # For each prompt, measure performance when it's masked out
        for i in range(self.num_prompts):
            with torch.no_grad():
                # Create temporary copy of prompts
                original_prompts = self.prompt_bank.prompts.clone()
                
                # Zero out the specific prompt
                self.prompt_bank.prompts[i].zero_()
                
                # Evaluate with this prompt disabled
                masked_ndcg, _ = evaluate(self, [valid_data, valid_data, {}, args.usernum, args.itemnum], args, device)
                
                # Restore original prompts
                self.prompt_bank.prompts = nn.Parameter(original_prompts)
                
                # Calculate importance (performance drop when prompt is removed)
                importance_scores[i] = max(0, baseline_ndcg - masked_ndcg)
        
        # Normalize importance scores
        if torch.sum(importance_scores) > 0:
            self.prompt_importance = importance_scores / torch.sum(importance_scores)
        else:
            # If all scores are zero, use uniform importance
            self.prompt_importance = torch.ones(self.num_prompts, device=self.dev) / self.num_prompts
        
        print(f"Prompt importance: {self.prompt_importance.cpu().numpy()}")
        
    def prepare_for_incremental_learning(self):
        """
        Prepare for incremental learning by completely freezing all prompts
        """
        print("=== Freezing all prompts for incremental learning ===")
        
        # Completely freeze the prompt bank
        for param in self.prompt_bank.parameters():
            param.requires_grad = False
        
        # The rest of the model can still learn
        for name, param in self.named_parameters():
            if 'prompt_bank' not in name:
                param.requires_grad = True


    def calculate_prompt_importance(self, validation_data, args, device):
        """
        Calculate importance of each prompt based on validation performance
        """
        print("=== Calculating prompt importance ===")
        self.eval()
        
        # First get baseline performance
        with torch.no_grad():
            baseline_ndcg, _ = evaluate(self, [validation_data, {}, {}, self.usernum, self.itemnum], args, device)
        
        # Measure performance drop when each prompt is disabled
        importance = torch.zeros(self.num_prompts, device=self.dev)
        for i in range(self.num_prompts):
            with torch.no_grad():
                # Store original prompt
                original_prompt = self.prompt_bank.prompts[i].clone()
                
                # Temporarily zero out this prompt
                self.prompt_bank.prompts[i].fill_(0)
                
                # Check performance without this prompt
                masked_ndcg, _ = evaluate(self, [validation_data, {}, {}, self.usernum, self.itemnum], args, device)
                
                # Restore original prompt
                self.prompt_bank.prompts[i].copy_(original_prompt)
                
                # Higher performance drop means higher importance
                importance[i] = max(0, baseline_ndcg - masked_ndcg)
        
        # Normalize importance
        if torch.sum(importance) > 0:
            self.prompt_importance = importance / torch.sum(importance)
        else:
            # If all zeros, use uniform importance
            self.prompt_importance = torch.ones(self.num_prompts, device=self.dev) / self.num_prompts
        
        print(f"Prompt importance: {self.prompt_importance.cpu().numpy()}")
        return self.prompt_importance

    def freeze_prompts_for_incremental(self):
        """
        Completely freeze all prompts for incremental learning
        """
        print("=== Freezing all prompts for incremental learning ===")
        
        # Completely freeze the prompt bank
        for param in self.prompt_bank.parameters():
            param.requires_grad = False