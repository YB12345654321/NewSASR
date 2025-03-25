import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from model_v1 import SASRec
from util import evaluate
from sampler import WarpSampler

class PromptBank(nn.Module):
    """
    A bank of learnable prompts that capture different user behavior patterns
    """
    def __init__(self, prompt_dim, num_prompts=8):
        super(PromptBank, self).__init__()
        # Initialize learnable prompt vectors
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, query_embedding, top_k=10):
        """
        Select relevant prompts based on query embedding
        
        Args:
            query_embedding: Embedding used to query the prompt bank
            
        Returns:
            Selected prompt embeddings
        """
        # Calculate similarity between query and prompts
        similarity = torch.matmul(query_embedding, self.prompts.T) / self.temperature
        
        # Get top-k indices and scores
        top_k_values, top_k_indices = torch.topk(similarity, k=min(top_k, similarity.size(-1)), dim=-1)
        
        # Create a sparse attention weights tensor
        attention = torch.zeros_like(similarity).scatter_(-1, top_k_indices, F.softmax(top_k_values, dim=-1))
        
        # Get weighted combination of prompts
        selected_prompts = torch.matmul(attention, self.prompts)
        
        return selected_prompts, attention, top_k_indices


class PromptBaseSASRec(SASRec):
    """
    Extends SASRec model with prompt-based incremental learning
    """
    def __init__(self, usernum, itemnum, args, initial_prompts=None):
        super(PromptBaseSASRec, self).__init__(usernum, itemnum, args)
        
        # Define prompt dimensions
        prompt_dim = args.item_hidden_units # + args.user_hidden_units
        self.num_prompts = getattr(args, 'num_prompts', 8)
        
        # Create prompt bank
        self.prompt_bank = PromptBank(prompt_dim, self.num_prompts)#, initial_prompts)
        
        # For tracking which prompts should be frozen
        self.frozen_prompt_indices = []
        
        # For tracking performance
        self.seen_items = set()
        self.new_items = set()
        
        # Parameters for continual learning
        self.prompt_mix_ratio = getattr(args, 'prompt_mix_ratio', 0.3)
        
    def forward(self, u, seq, pos, neg, is_training=True, base_model=None):
        """
        Forward pass with prompt augmentation and diversity loss
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
        
        # # User encoding
        # u_emb = self.user_emb(u)  # [B, H_user]
        # u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B, L, H_user]
        
        # Combine embeddings
        # seq = torch.cat([seq_emb, u_emb_expand], dim=-1)  # [B, L, H_total]
        # seq += pos_emb
        seq = seq_emb + pos_emb
        
        # Get a summary of user's sequence for prompt selection
        # Use the last position in the sequence as query vector
        valid_positions = mask.sum(dim=1) - 1  # Get index of last valid position
        valid_positions = torch.clamp(valid_positions, min=0)  # Ensure non-negative
        
        batch_indices = torch.arange(batch_size, device=self.dev)
        query_vectors = seq[batch_indices, valid_positions]
        
        # Calculate similarity between query and prompts
        similarities = torch.matmul(query_vectors, self.prompt_bank.prompts.T) / self.prompt_bank.temperature
        
        # Apply softmax to get attention weights
        prompt_weights = F.softmax(similarities, dim=-1)
        
        # Get weighted combination of prompts
        selected_prompts = torch.matmul(prompt_weights, self.prompt_bank.prompts)
        
        # Add diversity loss to encourage prompt specialization
        diversity_loss = 0.0
        if is_training:
            # Calculate entropy of prompt selection (higher entropy means more diverse)
            prompt_entropy = -torch.sum(prompt_weights * torch.log(prompt_weights + 1e-10), dim=-1).mean()
            
            # Calculate prompt correlation matrix
            prompt_corr = torch.matmul(prompt_weights.T, prompt_weights) / batch_size
            
            # Get off-diagonal elements (correlations between different prompts)
            off_diag_mask = 1.0 - torch.eye(self.num_prompts, device=self.dev)
            off_diag_corr = prompt_corr * off_diag_mask
            
            # Diversity loss: minimize correlations between prompts, maximize entropy
            diversity_loss = off_diag_corr.sum() / (self.num_prompts * (self.num_prompts - 1)) - 0.5 * prompt_entropy
            
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
        # user_emb = u_emb_expand.reshape(-1, u_emb.size(-1))  # [B*L, H_user]
        # pos_emb = torch.cat([pos_emb, user_emb], dim=1)  # [B*L, H_total]
        # neg_emb = torch.cat([neg_emb, user_emb], dim=1)  # [B*L, H_total]
        
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
                # torch.sum(self.user_emb.embedding.weight ** 2) +
                torch.sum(self.pos_emb.embedding.weight ** 2) +
                torch.sum(self.prompt_bank.prompts ** 2)  # Also regularize prompts
            )
            loss += l2_loss
        
        # Add diversity loss to total loss
        if is_training:
            diversity_weight = 0.1  # Adjust as needed
            loss += diversity_weight * diversity_loss
        
        # Compute AUC for training info
        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        # Add knowledge distillation if base model is provided and in training mode
        if is_training and base_model is not None:
            # Get base model predictions
            with torch.no_grad():
                # Run only the encoding part of the base model
                u_tensor = u.to(self.dev).long()
                seq_tensor = seq.to(self.dev).long()
                pos_tensor = pos.reshape(-1).to(self.dev).long()  # Ensure it's flattened as the model expects
                neg_tensor = neg.reshape(-1).to(self.dev).long() 
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
            
            # # User encoding
            # u_emb = self.user_emb(u)  # [B, H_user]
            # u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B, L, H_user]
            
            # Combine embeddings
            # seq_repr = torch.cat([seq_emb, u_emb_expand], dim=-1)  # [B, L, H_total]
            seq_repr = seq_emb+pos_emb
            
            # Get a summary of user's sequence for prompt selection
            # Use the last position in the sequence as query vector
            valid_positions = mask.sum(dim=1) - 1  # Get index of last valid position
            valid_positions = torch.clamp(valid_positions, min=0)  # Ensure non-negative
            
            batch_indices = torch.arange(batch_size).to(self.dev)
            query_vectors = seq_repr[batch_indices, valid_positions]
            
            # Select prompts based on the sequence representation
            similarities = torch.matmul(query_vectors, self.prompt_bank.prompts.T) / self.prompt_bank.temperature
            prompt_weights = F.softmax(similarities, dim=-1)
            selected_prompts = torch.matmul(prompt_weights, self.prompt_bank.prompts)
            
            # Mask padding
            if mask is not None:
                seq_repr = seq_repr * mask.unsqueeze(-1).float()
            
            # Apply prompts with position-aware integration
            # Clone sequence to avoid modifying the original
            enhanced_seq = seq_repr.clone()
            
            # For each position, apply position-dependent prompt influence
            seq_len = seq_repr.size(1)
            for pos in range(seq_len):
                # Calculate position-dependent weight
                # More recent items get more prompt influence
                pos_weight = 1.0 - 0.8 * (seq_len - 1 - pos) / max(1, seq_len - 1)
                
                # Mix prompts with the representation at this position
                mix_ratio = self.prompt_mix_ratio * pos_weight
                enhanced_seq[:, pos, :] = seq_repr[:, pos, :] * (1 - mix_ratio) + selected_prompts * mix_ratio
            
            # Use the enhanced sequence
            seq_repr = enhanced_seq
            
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
            
            # # Add user embeddings to test items
            # # test_u_emb = u_emb.unsqueeze(1).expand(-1, test_item_emb.size(0), -1)  # [B, N_items, H_user]
            # test_item_emb = torch.cat([
            #     test_item_emb.unsqueeze(0).expand(batch_size, -1, -1),  # [B, N_items, H_item]
            #     test_u_emb  # [B, N_items, H_user]
            # ], dim=-1)  # [B, N_items, H_total]
            test_item_emb = test_item_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_items, H_item]

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




    def train_with_separate_prompt_phases(self, train_data, valid_data, test_data, args, device):
        """
        Enhanced three-phase training approach:
        1. Train base model with frozen prompts
        2. Train prompts with frozen base model
        3. Fine-tune everything together
        """
        # Create sampler
        t1_sampler = WarpSampler(train_data, self.usernum, self.itemnum,
                            batch_size=args.batch_size, maxlen=args.maxlen,
                            threshold_user=args.threshold_user,
                            threshold_item=args.threshold_item,
                            n_workers=3, device=device)
        
        # Phase 1: Train base model with minimal prompt influence
        print("=== Phase 1: Training base model with minimal prompt influence ===")
        
        # Store original prompt_mix_ratio
        original_mix_ratio = self.prompt_mix_ratio
        
        # Set a very low mix ratio for Phase 1
        self.prompt_mix_ratio = 0.05  # Just enough to start learning prompt representations
        
        # Create optimizer for all parameters
        phase1_optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98))
        
        # Train for ~1/3 of the total epochs
        num_batch = max(len(train_data) // args.batch_size, 1)
        phase1_epochs = args.num_epochs // 3
        
        for epoch in range(1, phase1_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = t1_sampler.next_batch()
                
                phase1_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase1_optimizer.step()
                
            if epoch % args.print_freq == 0:
                # Create dataset with self.usernum and self.itemnum instead of args.usernum and args.itemnum
                t_test = evaluate(self, [train_data, valid_data, test_data, self.usernum, self.itemnum], args, device)
                print(f"[Phase 1 epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}")
        
        # Phase 2: Train prompts with a higher mix ratio while freezing base model
        print("=== Phase 2: Freezing base model, training prompts ===")
        
        # Restore original prompt mixing
        self.prompt_mix_ratio = original_mix_ratio
        
        # Freeze all non-prompt parameters
        for name, param in self.named_parameters():
            if 'prompt_bank' not in name:
                param.requires_grad = False
        
        # Create optimizer for prompt parameters only
        phase2_optimizer = torch.optim.Adam(
            self.prompt_bank.parameters(),
            lr=args.lr, betas=(0.9, 0.98)
        )
        
        # Train for ~1/3 of the total epochs
        phase2_epochs = args.num_epochs // 3
        
        for epoch in range(1, phase2_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = t1_sampler.next_batch()
                
                phase2_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase2_optimizer.step()
                
            if epoch % args.print_freq == 0:
                # Create dataset with self.usernum and self.itemnum instead of args.usernum and args.itemnum
                t_test = evaluate(self, [train_data, valid_data, test_data, self.usernum, self.itemnum], args, device)
                print(f"[Phase 2 epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}")
        
        # Phase 3: Fine-tune everything together
        print("=== Phase 3: Fine-tuning entire model with prompts ===")
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        
        # Use a lower learning rate for fine-tuning
        phase3_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.lr * 0.1, betas=(0.9, 0.98)
        )
        
        # Train for the remaining epochs
        phase3_epochs = max(1, args.num_epochs - phase1_epochs - phase2_epochs)
        
        for epoch in range(1, phase3_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = t1_sampler.next_batch()
                
                phase3_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase3_optimizer.step()
                
            if epoch % args.print_freq == 0:
                # Evaluate and print results
                t_test = evaluate(self, [train_data, valid_data, test_data, self.usernum, self.itemnum], args, device)
                print(f"[Phase 3 epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}")
        
        # Close sampler
        t1_sampler.close()



    def train_two_phase(self, train_data, valid_data, args, device, num_epochs=None):
        """
        Two-phase training: 
        1. Train the whole model
        2. Freeze the base model and train only prompts
        """
        from util import evaluate
        
        if num_epochs is None:
            num_epochs = args.num_epochs
            
        print("=== Phase 1: Training full model ===")
        
        # Create optimizer for all parameters
        phase1_optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98))
        
        # Create sampler
        sampler = WarpSampler(train_data, self.usernum, self.itemnum,
                            batch_size=args.batch_size, maxlen=args.maxlen,
                            threshold_user=args.threshold_user,
                            threshold_item=args.threshold_item,
                            n_workers=3, device=device)
        
        # Train for half the epochs
        phase1_epochs = num_epochs // 2
        num_batch = max(len(train_data) // args.batch_size, 1)
        
        for epoch in range(1, phase1_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                
                phase1_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase1_optimizer.step()
            
            if epoch % args.print_freq == 0:
                valid_data_format = [train_data, valid_data, {}, self.usernum, self.itemnum]
                ndcg, hr = evaluate(self, valid_data_format, args, device)
                print(f"[Phase 1 epoch {epoch}] NDCG={ndcg:.4f}, HR={hr:.4f}, Loss={loss:.4f}")
        
        print("=== Phase 2: Freezing base model, training prompts ===")
        
        # Freeze all parameters except prompts
        for name, param in self.named_parameters():
            if 'prompt_bank' not in name:
                param.requires_grad = False
        
        # Create optimizer for prompt parameters only
        phase2_optimizer = torch.optim.Adam(
            self.prompt_bank.parameters(), 
            lr=args.lr * 0.5,  # Lower learning rate for fine-tuning
            betas=(0.9, 0.98)
        )
        
        # Train for remaining epochs
        phase2_epochs = num_epochs - phase1_epochs
        
        for epoch in range(1, phase2_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                
                phase2_optimizer.zero_grad()
                loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase2_optimizer.step()
            
            if epoch % args.print_freq == 0:
                valid_data_format = [train_data, valid_data, {}, self.usernum, self.itemnum]
                ndcg, hr = evaluate(self, valid_data_format, args, device)
                print(f"[Phase 2 epoch {epoch}] NDCG={ndcg:.4f}, HR={hr:.4f}, Loss={loss:.4f}")
        
        # Unfreeze all parameters for future incremental learning
        for param in self.parameters():
            param.requires_grad = True
        
        # Close sampler
        sampler.close()


    def generate_new_prompts(self, slice_data, num_new_prompts=4, device='cuda'):
        """Generate new prompts from current slice data patterns"""
        print(f"Generating {num_new_prompts} new prompts from current slice data")
        
        # Extract representations from current slice data
        user_train = slice_data[0]
        
        # Create sampler for this slice
        from sampler import WarpSampler
        sampler = WarpSampler(user_train, self.usernum, self.itemnum,
                        batch_size=128, maxlen=self.args.maxlen,
                        threshold_user=self.args.threshold_user,
                        threshold_item=self.args.threshold_item,
                        n_workers=1, device=device)
        
        # Collect sequence representations
        sequence_reprs = []
        num_samples = min(1000, len(user_train))
        
        with torch.no_grad():
            for _ in range(num_samples // 128 + 1):
                u, seq, _, _ = sampler.next_batch()
                if len(u) == 0:
                    break
                    
                # Get embeddings
                mask = (seq > 0)
                seq_emb = self.item_emb(seq)
                pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(device)
                pos_ids = pos_ids.unsqueeze(0).expand_as(seq)
                pos_emb = self.pos_emb(pos_ids)
                # u_emb = self.user_emb(u)
                # u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                seq_repr = seq_emb # torch.cat([seq_emb, u_emb_expand], dim=-1)
                seq_repr += pos_emb
                
                # Get last valid position
                valid_positions = mask.sum(dim=1) - 1
                valid_positions = torch.clamp(valid_positions, min=0)
                batch_indices = torch.arange(seq.size(0), device=device)
                
                # Extract representations
                for i in range(len(u)):
                    pos = valid_positions[i].item()
                    repr = seq_repr[i, pos, :].detach().cpu().numpy()
                    sequence_reprs.append(repr)
                    
                    if len(sequence_reprs) >= num_samples:
                        break
                
                if len(sequence_reprs) >= num_samples:
                    break
        
        sampler.close()
        
        # Apply clustering to find new prompt patterns
        if len(sequence_reprs) > num_new_prompts:
            from sklearn.cluster import KMeans
            sequence_reprs = np.array(sequence_reprs)
            kmeans = KMeans(n_clusters=num_new_prompts, random_state=42).fit(sequence_reprs)
            new_prompts = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        else:
            # Not enough samples, just use what we have
            new_prompts = torch.tensor(np.array(sequence_reprs), dtype=torch.float32, device=device)
        
        # Store the current prompts
        current_prompts = self.prompt_bank.prompts.detach().clone()
        
        # Create a new prompt bank with the combined prompts
        combined_prompts = torch.cat([current_prompts, new_prompts])
        
        # Create a completely new Parameter for the prompt bank
        old_prompt_count = current_prompts.size(0)
        new_prompt_count = combined_prompts.size(0)
        
        # Replace the entire prompts parameter
        self.prompt_bank.prompts = torch.nn.Parameter(combined_prompts)
        self.num_prompts = new_prompt_count
        
        print(f"Prompt bank expanded to {self.num_prompts} prompts")
        
        # Handle hybrid freezing through gradient masking in forward pass
        # instead of trying to set requires_grad on individual tensor elements
        
        return self.num_prompts


    def analyze_prompt_usage(self, user_train, args, device):
        """Analyze how frequently each prompt is selected as important"""
        prompt_usage = torch.zeros(self.num_prompts, device=device)
        
        # Create a minimal loader for analysis
        from sampler import WarpSampler
        sampler = WarpSampler(user_train, self.usernum, self.itemnum,
                        batch_size=args.batch_size, maxlen=args.maxlen,
                        threshold_user=args.threshold_user,
                        threshold_item=args.threshold_item,
                        n_workers=1, device=device)
        
        # Sample some batches
        num_batches = min(100, max(len(user_train) // args.batch_size, 10))
        
        with torch.no_grad():
            for _ in range(num_batches):
                u, seq, _, _ = sampler.next_batch()
                
                # Get mask for valid items
                mask = (seq > 0)
                
                # Get embeddings and prepare for prompt selection
                seq_emb = self.item_emb(seq)
                pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(device)
                pos_ids = pos_ids.unsqueeze(0).expand_as(seq)
                pos_emb = self.pos_emb(pos_ids)
                # u_emb = self.user_emb(u)
                # u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                seq_repr = seq_emb # torch.cat([seq_emb, u_emb_expand], dim=-1)
                seq_repr += pos_emb
                
                # Get query vectors
                valid_positions = mask.sum(dim=1) - 1
                valid_positions = torch.clamp(valid_positions, min=0)
                batch_indices = torch.arange(seq.size(0), device=device)
                query_vectors = seq_repr[batch_indices, valid_positions]
                
                # Get prompt weights
                _, prompt_weights, top_k_indices = self.prompt_bank(query_vectors)
                
                # Update usage statistics
                for indices in top_k_indices:
                    for idx in indices:
                        prompt_usage[idx] += 1
                        
        # Normalize usage
        if torch.sum(prompt_usage) > 0:
            prompt_usage = prompt_usage / torch.sum(prompt_usage)
        
        sampler.close()
        return prompt_usage
    


    def setup_prompt_gradient_masking(self, frozen_indices):
        """
        Set up a hook for gradient masking to implement hybrid prompt freezing
        
        Args:
            frozen_indices: List of prompt indices to freeze
        """
        # Create a mask for the prompts parameter
        self.frozen_prompt_indices = frozen_indices
        
        # Remove existing hooks if any
        if hasattr(self, '_prompt_grad_hook'):
            self._prompt_grad_hook.remove()
        
        # Register the hook
        self._prompt_grad_hook = self.prompt_bank.prompts.register_hook(self.mask_prompt_grads)
        
        print(f"Gradient masking set up: {len(frozen_indices)} prompts frozen, "
            f"{self.num_prompts - len(frozen_indices)} prompts trainable")
        
        return True
    
    # Define the hook function
    def mask_prompt_grads(self, grad):
        # Create a mask initialized with ones (all prompts trainable by default)
        mask = torch.ones_like(grad)
        
        # Set mask to zero for prompts that should be frozen
        for idx in self.frozen_prompt_indices:
            if idx < grad.size(0):
                mask[idx] = 0.0
                
        # Apply the mask to zero out gradients for frozen prompts
        return grad * mask
    
    def ensure_hybrid_prompt_freezing(self):
        """
        Ensure hybrid prompt freezing during incremental learning
        """
        print("=== Ensuring hybrid prompt freezing during incremental learning ===")
        
        # Get the frozen_prompt_count from args, defaulting to 1024 if not specified
        frozen_count = getattr(self.args, 'frozen_prompt_count', 1024)
        
        # Determine which prompts to freeze (memory prompts)
        # Freeze the first frozen_count prompts (or all if fewer than frozen_count)
        frozen_count = min(frozen_count, self.num_prompts)
        frozen_indices = list(range(frozen_count))
        
        # Set up gradient masking for these prompts
        self.setup_prompt_gradient_masking(frozen_indices)
        
        return frozen_indices
    
    # For the train_incremental_phases method in PromptBaseSASRec class
def train_incremental_phases(self, train_data, valid_data, test_data, args, device, base_model=None):
    """
    Three-phase training approach for incremental slices with optimized learning rates and epoch allocation:
    1. Train base model with minimal prompt influence (30% of epochs)
    2. Train prompts with frozen base model (50% of epochs) - higher learning rate
    3. Fine-tune everything together with a lower learning rate (20% of epochs)
    """
    # Create sampler
    sampler = WarpSampler(train_data, self.usernum, self.itemnum,
                         batch_size=args.batch_size, maxlen=args.maxlen,
                         threshold_user=args.threshold_user,
                         threshold_item=args.threshold_item,
                         n_workers=3, device=device)
    
    # Prepare evaluation data
    eval_data = [train_data, valid_data, test_data, self.usernum, self.itemnum]
    
    # Get number of epochs for each phase with new distribution
    total_epochs = args.num_epochs // 4  # Fewer epochs for incremental learning
    phase1_epochs = max(1, int(total_epochs * 0.3))  # 30% for Phase 1
    phase2_epochs = max(1, int(total_epochs * 0.5))  # 50% for Phase 2
    phase3_epochs = max(1, total_epochs - phase1_epochs - phase2_epochs)  # Remaining for Phase 3
    
    print(f"Phase distribution: Phase 1 = {phase1_epochs} epochs, Phase 2 = {phase2_epochs} epochs, Phase 3 = {phase3_epochs} epochs")
    
    # Store original prompt mixing ratio
    original_mix_ratio = self.prompt_mix_ratio
    
    # ===========================================================
    # Phase 1: Train with minimal prompt influence
    # ===========================================================
    print("=== Phase 1: Training with minimal prompt influence ===")
    
    # Reduce prompt influence
    self.prompt_mix_ratio = 0.05
    
    # Make all parameters trainable
    for param in self.parameters():
        param.requires_grad = True
    
    # Ensure hybrid freezing is still respected
    self.ensure_hybrid_prompt_freezing()
    
    # Create optimizer - keep Phase 1 learning rate as is
    phase1_lr = args.lr * 0.1  # Keeping the same: 10% of original LR
    phase1_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=phase1_lr, betas=(0.9, 0.98)
    )
    
    # Train Phase 1
    num_batch = max(len(train_data) // args.batch_size, 1)
    for epoch in range(1, phase1_epochs + 1):
        total_loss = 0
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            
            phase1_optimizer.zero_grad()
            loss, _, _, _ = self(u, seq, pos, neg, is_training=True, base_model=base_model)
            loss.backward()
            phase1_optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate after each epoch or at specified intervals
        if epoch % args.print_freq == 0 or epoch == phase1_epochs:
            avg_loss = total_loss / num_batch
            ndcg, hr = evaluate(self, eval_data, args, device)
            print(f"[Phase 1 epoch {epoch}/{phase1_epochs}] NDCG={ndcg:.4f}, HR={hr:.4f}, Loss={avg_loss:.4f}")
    
    # ===========================================================
    # Phase 2: Train prompts with frozen base model
    # ===========================================================
    print("=== Phase 2: Training prompts with frozen base model ===")
    
    # Restore original prompt mix ratio with a boost to emphasize prompt influence
    self.prompt_mix_ratio = original_mix_ratio * 1.2  # Increase by 20% to emphasize prompts
    
    # Freeze all non-prompt parameters
    for name, param in self.named_parameters():
        if 'prompt_bank' not in name:
            param.requires_grad = False
    
    # Create optimizer for prompt parameters with higher learning rate
    phase2_lr = args.lr * 0.2  # Increased from 0.05 to 0.2 (4x higher)
    phase2_optimizer = torch.optim.Adam(
        self.prompt_bank.parameters(),
        lr=phase2_lr, betas=(0.9, 0.98)
    )
    
    print(f"Phase 2 learning rate increased to {phase2_lr:.6f} (from {args.lr * 0.05:.6f})")
    
    # Train Phase 2 - now with more epochs
    for epoch in range(1, phase2_epochs + 1):
        total_loss = 0
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            
            phase2_optimizer.zero_grad()
            loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
            loss.backward()
            phase2_optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate after each epoch or at specified intervals
        if epoch % args.print_freq == 0 or epoch == phase2_epochs:
            avg_loss = total_loss / num_batch
            ndcg, hr = evaluate(self, eval_data, args, device)
            print(f"[Phase 2 epoch {epoch}/{phase2_epochs}] NDCG={ndcg:.4f}, HR={hr:.4f}, Loss={avg_loss:.4f}")
    
    # ===========================================================
    # Phase 3: Fine-tune entire model with prompts
    # ===========================================================
    print("=== Phase 3: Fine-tuning entire model with prompts ===")
    
    # Restore normal prompt mix ratio
    self.prompt_mix_ratio = original_mix_ratio
    
    # Get the frozen_prompt_count from args
    frozen_count = getattr(args, 'frozen_prompt_count', 256)  # Default to 256 instead of 1024
    
    # Unfreeze all parameters (except those that should remain frozen)
    for name, param in self.named_parameters():
        if name.startswith('prompt_bank.prompts'):
            prompt_idx = int(name.split('.')[2]) if len(name.split('.')) > 2 else -1
            if prompt_idx >= 0 and prompt_idx < frozen_count:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = True
    
    # Re-apply hybrid prompt freezing to ensure correct gradient masking
    self.ensure_hybrid_prompt_freezing()
    
    # Create optimizer with slightly higher learning rate than before
    phase3_lr = args.lr * 0.02  # Increased from 0.01 to 0.02 (2x higher)
    phase3_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=phase3_lr, betas=(0.9, 0.98)
    )
    
    print(f"Phase 3 learning rate increased to {phase3_lr:.6f} (from {args.lr * 0.01:.6f})")
    
    # Train Phase 3
    for epoch in range(1, phase3_epochs + 1):
        total_loss = 0
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            
            phase3_optimizer.zero_grad()
            loss, _, _, _ = self(u, seq, pos, neg, is_training=True)
            loss.backward()
            phase3_optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate after each epoch or at specified intervals
        if epoch % args.print_freq == 0 or epoch == phase3_epochs:
            avg_loss = total_loss / num_batch
            ndcg, hr = evaluate(self, eval_data, args, device)
            print(f"[Phase 3 epoch {epoch}/{phase3_epochs}] NDCG={ndcg:.4f}, HR={hr:.4f}, Loss={avg_loss:.4f}")
    
    # Close sampler
    sampler.close()
    
    # Return final performance metrics
    final_ndcg, final_hr = evaluate(self, eval_data, args, device)
    return final_ndcg, final_hr


    
class EnsemblePromptSASRec(nn.Module):
    """
    Ensemble model that combines predictions from a frozen T1 model and an incrementally trained model
    """
    def __init__(self, frozen_t1_model, incremental_model, alpha=0.3):
        super(EnsemblePromptSASRec, self).__init__()
        self.frozen_t1_model = frozen_t1_model
        self.incremental_model = incremental_model
        self.alpha = alpha  # Weight for the frozen model (0-1)
        
        # Ensure frozen model stays frozen
        for param in self.frozen_t1_model.parameters():
            param.requires_grad = False
            
        # Reference to device for consistency
        self.dev = incremental_model.dev
        
    def predict(self, u, seq, item_idx):
        """
        Make ensemble predictions using both models
        
        Args:
            u: User indices
            seq: Sequence of items
            item_idx: Items to score
        """
        # Get predictions from both models
        with torch.no_grad():
            frozen_scores = self.frozen_t1_model.predict(u, seq, item_idx)
            incremental_scores = self.incremental_model.predict(u, seq, item_idx)
            
            # Weighted combination
            combined_scores = self.alpha * frozen_scores + (1 - self.alpha) * incremental_scores
            
        return combined_scores
        
    def forward(self, u, seq, pos, neg, is_training=True):
        """
        For training, only update incremental model
        """
        return self.incremental_model(u, seq, pos, neg, is_training=is_training)
    

class EnhancedEnsemblePromptSASRec(nn.Module):
    """
    Enhanced ensemble model that adaptively combines predictions from a frozen T1 model
    and an incrementally trained model based on item characteristics
    """
    def __init__(self, frozen_t1_model, incremental_model, default_alpha=0.3):
        super(EnhancedEnsemblePromptSASRec, self).__init__()
        self.frozen_t1_model = frozen_t1_model
        self.incremental_model = incremental_model
        self.default_alpha = default_alpha  # Default weight for the frozen model (0-1)
        
        # Ensure frozen model stays frozen
        for param in self.frozen_t1_model.parameters():
            param.requires_grad = False
            
        # Reference to device for consistency
        self.dev = incremental_model.dev
        
        # Get item sets from incremental model
        self.seen_items = incremental_model.seen_items if hasattr(incremental_model, 'seen_items') else set()
        self.new_items = incremental_model.new_items if hasattr(incremental_model, 'new_items') else set()
        
    def predict(self, u, seq, item_idx):
        """
        Make ensemble predictions using both models with adaptive weighting
        
        Args:
            u: User indices
            seq: Sequence of items
            item_idx: Items to score
        """
        # Get predictions from both models
        with torch.no_grad():
            frozen_scores = self.frozen_t1_model.predict(u, seq, item_idx)
            incremental_scores = self.incremental_model.predict(u, seq, item_idx)
            
            # Create item-specific alpha values
            alphas = torch.ones_like(frozen_scores) * self.default_alpha
            
            # Adjust alpha based on item characteristics
            for batch_idx in range(item_idx.size(0)):
                for item_pos in range(item_idx.size(1)):
                    item = item_idx[batch_idx, item_pos].item()
                    
                    if item in self.seen_items:
                        # Old items from T1: higher weight for frozen model
                        alphas[batch_idx, item_pos] = 0.7
                    elif item in self.new_items:
                        # New items from incremental slices: lower weight for frozen model
                        alphas[batch_idx, item_pos] = 0.1
                    # Otherwise, use default alpha for unseen items
            
            # Weighted combination
            combined_scores = alphas * frozen_scores + (1 - alphas) * incremental_scores
            
        return combined_scores
        
    def forward(self, u, seq, pos, neg, is_training=True):
        """
        For training, only update incremental model
        """
        return self.incremental_model(u, seq, pos, neg, is_training=is_training)