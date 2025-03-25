import torch
import numpy as np
import torch.nn.functional as F
import copy

class PromptManager:
    """
    Manager for handling prompt selection and usage analysis in incremental learning
    """
    def __init__(self, prompt_model, num_prompts=8):
        self.prompt_model = prompt_model
        self.num_prompts = num_prompts
        
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
        
    def update_prompt_count(self):
        """Update the number of prompts based on the model's current prompt count"""
        current_count = self.prompt_model.prompt_bank.prompts.size(0)
        if self.num_prompts != current_count:
            self.num_prompts = current_count
            print(f"PromptManager: Updated prompt count to {self.num_prompts}")
        return self.num_prompts

    def analyze_prompts(self, dataset, args, device):
        """
        Analyze how prompts are used for different users/items
        Enhanced to track top-K prompt usage
        
        Args:
            dataset: Dataset to analyze
            args: Command line arguments
            device: Device to run on
            
        Returns:
            Dictionary of analysis results
        """
        self.prompt_model.eval()
        self.update_prompt_count()

        
        results = {
            'prompt_usage': {},
            'top_k_selections': {},
            'item_type_usage': {
                'old_items': {},
                'new_items': {},
                'unseen_items': {}
            }
        }
        # Get the current number of prompts from the model
        current_num_prompts = self.prompt_model.prompt_bank.prompts.size(0)
            
        # Track prompt usage (on the right device)
        prompt_usage = torch.zeros(self.num_prompts, device=device)
        top_k_selections = torch.zeros(self.num_prompts, device=device)
        
        # Track usage by item type
        old_item_usage = torch.zeros(self.num_prompts, device=device)
        new_item_usage = torch.zeros(self.num_prompts, device=device)
        unseen_item_usage = torch.zeros(self.num_prompts, device=device)
        
        # Track co-occurrence of prompts (how often are they selected together)
        prompt_cooccurrence = torch.zeros(self.num_prompts, self.num_prompts, device=device)
        
        # Sample users from dataset
        user_train = dataset[0]
        all_users = list(user_train.keys())
        if len(all_users) > 100:
            sample_users = np.random.choice(all_users, 100, replace=False)
        else:
            sample_users = all_users

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
                # u_emb = self.prompt_model.user_emb(u_tensor)
                # u_emb_expand = u_emb.unsqueeze(1).expand(-1, seq.size(1), -1)
                
                # Combine embeddings
                # seq_repr = torch.cat([seq_emb, u_emb_expand], dim=-1)
                seq_repr = seq_emb
                seq_repr += pos_emb
                
                # Get a summary of user's sequence for prompt selection
                valid_positions = mask.sum(dim=1) - 1
                valid_positions = torch.clamp(valid_positions, min=0)
                
                batch_indices = torch.arange(seq.size(0)).to(device)
                query_vectors = seq_repr[batch_indices, valid_positions]
                
                # Select prompts based on the sequence representation
                _, prompt_weights, top_k_indices = self.prompt_model.prompt_bank(query_vectors, top_k=3)
                
                # Update overall prompt usage
                # prompt_usage += prompt_weights.squeeze(0)
                # Get the current shape of prompt_weights after squeezing
                current_weights_size = prompt_weights.squeeze(0).size(0)

                # Ensure prompt_usage is the right size
                if prompt_usage.size(0) != current_weights_size:
                    # Create a new tensor of the right size and copy existing values where possible
                    new_prompt_usage = torch.zeros(current_weights_size, device=device)
                    min_size = min(prompt_usage.size(0), current_weights_size)
                    new_prompt_usage[:min_size] = prompt_usage[:min_size]
                    prompt_usage = new_prompt_usage
                    
                    # Also resize other tensors that depend on prompt count
                    new_top_k_selections = torch.zeros(current_weights_size, device=device)
                    new_top_k_selections[:min_size] = top_k_selections[:min_size]
                    top_k_selections = new_top_k_selections
                    
                    new_old_item_usage = torch.zeros(current_weights_size, device=device)
                    new_old_item_usage[:min_size] = old_item_usage[:min_size]
                    old_item_usage = new_old_item_usage
                    
                    new_new_item_usage = torch.zeros(current_weights_size, device=device)
                    new_new_item_usage[:min_size] = new_item_usage[:min_size]
                    new_item_usage = new_new_item_usage
                    
                    new_unseen_item_usage = torch.zeros(current_weights_size, device=device)
                    new_unseen_item_usage[:min_size] = unseen_item_usage[:min_size]
                    unseen_item_usage = new_unseen_item_usage
                    
                    # Resize cooccurrence matrix
                    new_cooccurrence = torch.zeros(current_weights_size, current_weights_size, device=device)
                    min_size = min(prompt_cooccurrence.size(0), current_weights_size)
                    new_cooccurrence[:min_size, :min_size] = prompt_cooccurrence[:min_size, :min_size]
                    prompt_cooccurrence = new_cooccurrence

                # Now we can safely add
                prompt_usage += prompt_weights.squeeze(0)
                                
                # Update top-k selection count
                for idx in top_k_indices.squeeze(0):
                    top_k_selections[idx] += 1
                    
                # Update co-occurrence matrix
                for i in top_k_indices.squeeze(0):
                    for j in top_k_indices.squeeze(0):
                        prompt_cooccurrence[i, j] += 1
                
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
        
        if torch.sum(top_k_selections) > 0:
            top_k_selections = top_k_selections / torch.sum(top_k_selections)
        
        if total_old > 0:
            old_item_usage = old_item_usage / total_old
        
        if total_new > 0:
            new_item_usage = new_item_usage / total_new
            
        if total_unseen > 0:
            unseen_item_usage = unseen_item_usage / total_unseen
        
        # Normalize co-occurrence matrix
        total_selections = prompt_cooccurrence.sum() / self.num_prompts  # Account for self-selection
        if total_selections > 0:
            prompt_cooccurrence = prompt_cooccurrence / total_selections
        
        # Calculate prompt specialization scores
        # Higher score means the prompt tends to be used for specific item types
        prompt_specialization = torch.zeros(current_num_prompts, device=device)
        if total_old > 0 and total_new > 0:
            for i in range(self.num_prompts):
                # Calculate how differently this prompt is used for old vs new items
                specialization = torch.abs(old_item_usage[i] - new_item_usage[i])
                prompt_specialization[i] = specialization
        
        # Store results (convert to numpy for storage)
        results['prompt_usage'] = prompt_usage.cpu().numpy()
        results['top_k_selections'] = top_k_selections.cpu().numpy()
        results['prompt_cooccurrence'] = prompt_cooccurrence.cpu().numpy()
        results['prompt_specialization'] = prompt_specialization.cpu().numpy()
        results['item_type_usage']['old_items'] = old_item_usage.cpu().numpy()
        results['item_type_usage']['new_items'] = new_item_usage.cpu().numpy()
        results['item_type_usage']['unseen_items'] = unseen_item_usage.cpu().numpy()
        
        # Calculate specialized roles based on usage patterns
        roles = []
        for i in range(current_num_prompts):
            if old_item_usage[i] > new_item_usage[i] * 1.5:
                roles.append("Old Items Specialist")
            elif new_item_usage[i] > old_item_usage[i] * 1.5:
                roles.append("New Items Specialist")
            elif unseen_item_usage[i] > (old_item_usage[i] + new_item_usage[i]) / 2 * 1.5:
                roles.append("Exploration Specialist")
            else:
                roles.append("General Purpose")
        
        results['prompt_roles'] = roles
        
        self.prompt_model.train()
        return results