import torch
import numpy as np
from sklearn.cluster import KMeans

class PromptExtractor:
    def __init__(self, base_model, layer_idx=0):
        """
        Initialize a prompt extractor that extracts representations from a specific transformer layer
        
        Args:
            base_model: The trained base model
            layer_idx: Index of the transformer layer to extract from (0-indexed)
        """
        self.base_model = base_model
        self.layer_idx = layer_idx
        self.representations = []
        
        # Register hook to extract representations
        self.hook = self.base_model.attention_layers[layer_idx].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        """Hook function to capture layer outputs"""
        # Output[0] is the sequence representation after this attention layer
        self.representations.append(output[0].detach().cpu())
    
    def extract_representations(self, data_loader):
        """
        Extract sequence representations from the model
        
        Args:
            data_loader: DataLoader providing (u, seq, pos, neg) batches
            
        Returns:
            Array of sequence representations
        """
        self.base_model.eval()
        all_reps = []
        
        with torch.no_grad():
            for u, seq, pos, neg in data_loader:
                # Move tensors to the same device as the model
                device = next(self.base_model.parameters()).device
                u = u.to(device)
                seq = seq.to(device)
                
                # Forward pass to trigger the hook
                self.base_model(u, seq, torch.zeros_like(seq), torch.zeros_like(seq), is_training=False)
                
                # Get last non-padding position for each sequence
                mask = (seq > 0)
                valid_positions = mask.sum(dim=1) - 1
                valid_positions = torch.clamp(valid_positions, min=0)
                
                # Extract representations for valid positions
                batch_size = seq.size(0)
                for i in range(batch_size):
                    pos = valid_positions[i].item()
                    rep = self.representations[-1][i, pos, :].numpy()
                    all_reps.append(rep)
                    
                # Clear stored representations
                self.representations = []
        
        return np.array(all_reps)
    
    def generate_prompts(self, data_loader, num_prompts=8):
        """
        Generate prompts via clustering sequence representations
        
        Args:
            data_loader: DataLoader providing training batches
            num_prompts: Number of prompts to generate
            
        Returns:
            Tensor of prompts
        """
        print(f"Extracting representations from layer {self.layer_idx}...")
        reps = self.extract_representations(data_loader)
        
        print(f"Running K-means clustering with k={num_prompts}...")
        kmeans = KMeans(n_clusters=num_prompts, random_state=0, n_init=10).fit(reps)
        
        # Get cluster centroids as prompts
        prompts = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        # Clean up
        self.hook.remove()
        
        return prompts

def create_data_loader(user_train, usernum, itemnum, args, device):
    """Create a data loader from user_train data"""
    from sampler import WarpSampler
    
    # Create sampler
    sampler = WarpSampler(user_train, usernum, itemnum,
                        batch_size=args.batch_size, maxlen=args.maxlen,
                        threshold_user=args.threshold_user,
                        threshold_item=args.threshold_item,
                        n_workers=3, device=device)
    
    # Create a wrapper to behave like a DataLoader
    class SamplerWrapper:
        def __init__(self, sampler, num_batches):
            self.sampler = sampler
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield self.sampler.next_batch()
        
        def __len__(self):
            return self.num_batches
    
    # Estimate number of batches
    num_batches = max(len(user_train) // args.batch_size, 100)
    
    return SamplerWrapper(sampler, num_batches), sampler

def analyze_prompt_diversity(prompts):
    """
    Analyze the diversity of prompts
    
    Args:
        prompts: Tensor of prompts [num_prompts, dim]
        
    Returns:
        Dictionary of diversity metrics
    """
    import torch.nn.functional as F
    
    # Normalize prompts
    normalized_prompts = F.normalize(prompts, p=2, dim=1)
    
    # Calculate cosine similarity matrix
    similarity = torch.matmul(normalized_prompts, normalized_prompts.t())
    
    # Exclude self-similarity (diagonal)
    mask = 1.0 - torch.eye(prompts.size(0), device=prompts.device)
    masked_similarity = similarity * mask
    
    # Calculate metrics
    mean_sim = masked_similarity.sum() / (prompts.size(0) * (prompts.size(0) - 1))
    max_sim = masked_similarity.max()
    min_sim = masked_similarity[masked_similarity > 0].min()
    
    return {
        'mean_similarity': mean_sim.item(),
        'max_similarity': max_sim.item(),
        'min_similarity': min_sim.item()
    }

def evaluate_prompts(model, dataset, args, device):
    """
    Evaluate how much prompts contribute to performance
    
    Args:
        model: Model with prompts
        dataset: Dataset to evaluate on
        args: Command line arguments
        device: Device to run on
        
    Returns:
        Tuple of (ndcg_contribution, hr_contribution)
    """
    from util import evaluate
    
    # First, store original mix ratio
    original_mix_ratio = model.prompt_mix_ratio
    
    # Evaluate without prompts
    model.prompt_mix_ratio = 0.0
    results_without_prompts = evaluate(model, dataset, args, device)
    
    # Evaluate with prompts
    model.prompt_mix_ratio = original_mix_ratio
    results_with_prompts = evaluate(model, dataset, args, device)
    
    # Calculate prompt contribution
    ndcg_contribution = results_with_prompts[0] - results_without_prompts[0]
    hr_contribution = results_with_prompts[1] - results_without_prompts[1]
    
    return ndcg_contribution, hr_contribution