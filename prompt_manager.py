import torch
import numpy as np
import torch.nn.functional as F
import copy

class PromptManager:
    """
    Manager for handling prompt selection and update strategies in incremental learning
    """
    def __init__(self, prompt_model, num_prompts=8, importance_threshold=0.7):
        self.prompt_model = prompt_model
        self.num_prompts = num_prompts
        self.importance_threshold = importance_threshold
        
        # Get the device from the model
        self.device = next(prompt_model.parameters()).device
        
        # Initialize prompt usage statistics
        self.prompt_usage = torch.zeros(num_prompts, device=self.device)
        
        # Track items in different time slices
        self.old_items = set()
        self.new_items = set()
        
    def update_item_sets(self, old_items, new_items):
        """
        Update tracked item sets
        """
        self.old_items = set(old_items)
        self.new_items = set(new_items)
        self.prompt_model.update_item_sets(old_items, new_items)
        

    def prepare_for_training(self, current_slice):
        """
        Prepare prompts for training on a new time slice
        
        Args:
            current_slice: Index of the current time slice
        """
        # Normalize prompt importance
        prompt_importance = self.prompt_model.prompt_importance
        if torch.sum(prompt_importance) > 0:
            normalized_importance = prompt_importance / torch.sum(prompt_importance)
            
            # Freeze important prompts to prevent catastrophic forgetting
            if current_slice > 1:  # Only start freezing after first incremental update
                for param in self.prompt_model.prompt_bank.parameters():
                    param.requires_grad = True  # Reset all parameters to trainable
                    
                # Register gradient hook for selective updating
                def grad_hook(grad):
                    # Create mask based on importance threshold
                    mask = normalized_importance > self.importance_threshold
                    mask = mask.to(grad.device)
                    # Zero out gradients for important prompts (freeze them)
                    mask_expanded = mask.unsqueeze(1).expand_as(grad)
                    return grad * (~mask_expanded)
                
                # Apply the hook
                self.prompt_model.prompt_bank.prompts.register_hook(grad_hook)
                
            print(f"Prompt importance: {normalized_importance.cpu().numpy()}")
            print(f"Freezing prompts with importance > {self.importance_threshold}")

            
    def reset_prompt_stats(self):
        """
        Reset prompt usage statistics
        """
        # Reset prompt importance (ensure it's on the correct device)
        self.prompt_model.prompt_importance.zero_()
        
    def analyze_prompts(self, dataset, args, device):
        """
        Analyze how prompts are used for different users/items
        
        Args:
            dataset: Dataset to analyze
            args: Command line arguments
            device: Device to run on
            
        Returns:
            Dictionary of analysis results
        """
        self.prompt_model.eval()
        
        results = {
            'prompt_usage': {},
            'item_type_usage': {
                'old_items': {},
                'new_items': {},
                'unseen_items': {}
            }
        }
        
        # Sample users from dataset
        user_train = dataset[0]
        all_users = list(user_train.keys())
        if len(all_users) > 100:
            sample_users = np.random.choice(all_users, 100, replace=False)
        else:
            sample_users = all_users
            
        # Track prompt usage (on the right device)
        prompt_usage = torch.zeros(self.num_prompts, device=device)
        old_item_usage = torch.zeros(self.num_prompts, device=device)
        new_item_usage = torch.zeros(self.num_prompts, device=device)
        unseen_item_usage = torch.zeros(self.num_prompts, device=device)
        
        total_old = 0
        total_new = 0
        total_unseen = 0
        
        with torch.no_grad():
            for u in sample_users:
                if u not in user_train or len(user_train[u]) < 1:
                    continue
                    
                # Prepare sequence
                seq = np.zeros([args.maxlen], dtype=np.int64)
                idx = args.maxlen - 1
                
                for i in reversed(user_train[u]):
                    seq[idx] = i
                    idx -= 1
                    if idx == -1: break
                
                # Convert to tensors
                seq = torch.LongTensor(seq).unsqueeze(0).to(device)
                u_tensor = torch.LongTensor([u]).to(device)
                
                # Get mask for valid sequence items
                mask = (seq > 0)
                
                # Get embeddings
                seq_emb = self.prompt_model.item_emb(seq)
                
                # Position encoding
                pos_ids = torch.arange(seq.size(1), dtype=torch.long).to(device)
                pos_ids = pos_ids.unsqueeze(0).expand_as(seq)
                pos_emb = self.prompt_model.pos_emb(pos_ids)
                
                # User encoding
                u_emb = self.prompt_model.user_emb(u_tensor)
                u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                
                # Combine embeddings
                seq_repr = torch.cat([seq_emb, u_emb_expand], dim=-1)
                seq_repr += pos_emb
                
                # Get a summary of user's sequence for prompt selection
                valid_positions = mask.sum(dim=1) - 1
                valid_positions = torch.clamp(valid_positions, min=0)
                
                batch_indices = torch.arange(seq.size(0)).to(device)
                query_vectors = seq_repr[batch_indices, valid_positions]
                
                # Select prompts based on the sequence representation
                _, prompt_weights = self.prompt_model.prompt_bank(query_vectors)
                
                # Update overall prompt usage
                prompt_usage += prompt_weights.squeeze(0)
                
                # Analyze usage patterns for different item types
                for i in seq.squeeze().cpu().numpy():
                    if i == 0:  # Skip padding
                        continue
                        
                    if i in self.old_items:
                        old_item_usage += prompt_weights.squeeze(0)
                        total_old += 1
                    elif i in self.new_items:
                        new_item_usage += prompt_weights.squeeze(0)
                        total_new += 1
                    else:
                        unseen_item_usage += prompt_weights.squeeze(0)
                        total_unseen += 1
        
        # Normalize usage statistics
        if torch.sum(prompt_usage) > 0:
            prompt_usage = prompt_usage / torch.sum(prompt_usage)
        
        if total_old > 0:
            old_item_usage = old_item_usage / total_old
        
        if total_new > 0:
            new_item_usage = new_item_usage / total_new
            
        if total_unseen > 0:
            unseen_item_usage = unseen_item_usage / total_unseen
        
        # Store results (convert to numpy for storage)
        results['prompt_usage'] = prompt_usage.cpu().numpy()
        results['item_type_usage']['old_items'] = old_item_usage.cpu().numpy()
        results['item_type_usage']['new_items'] = new_item_usage.cpu().numpy()
        results['item_type_usage']['unseen_items'] = unseen_item_usage.cpu().numpy()
        
        self.prompt_model.train()
        return results