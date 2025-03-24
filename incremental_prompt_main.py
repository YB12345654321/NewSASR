import os
import time
import pickle
import torch
import numpy as np
import copy
import random
import argparse
from sampler import WarpSampler
from model_v1 import SASRec
from util import evaluate, evaluate_valid, data_partition

# Import incremental learning components
from incremental_model import IncrementalSASRec, JointSASRec
from incremental_data import TimeSlicedData, ExperienceReplay
from eval_incremental import evaluate_incremental, evaluate_forgetting

# Import prompt-based components
from prompt_model import EnsemblePromptSASRec, PromptBaseSASRec, PromptBank #, setup_prompt_gradient_masking, ensure_hybrid_prompt_freezing, generate_new_prompts
from prompt_manager import PromptManager

parser = argparse.ArgumentParser()
# Original arguments
parser.add_argument('--dataset', default='ml1m')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--user_hidden_units', default=50, type=int)
parser.add_argument('--item_hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=20, type=int) # was 20
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--threshold_user', default=1.0, type=float)
parser.add_argument('--threshold_item', default=1.0, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--k', default=10, type=int)

# Incremental learning arguments
parser.add_argument('--incremental', action='store_true', help='Enable incremental learning')
parser.add_argument('--slice_ratios', nargs='+', type=float, default=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], help='Ratios for user-based time slices')
parser.add_argument('--min_interactions', default=10, type=int, help='Minimum interactions per user for valid data')
parser.add_argument('--distill_temp', default=2.0, type=float, help='Temperature for knowledge distillation')
parser.add_argument('--distill_alpha', default=0.5, type=float, help='Weight for distillation loss')
parser.add_argument('--replay_ratio', default=0.3, type=float, help='Ratio of experience replay')
parser.add_argument('--buffer_size', default=1000, type=int, help='Size of experience replay buffer')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
parser.add_argument('--load_splits', default=None, type=str, help='Path to load saved data splits')
parser.add_argument('--save_splits', action='store_true', help='Whether to save data splits for future use')

# Prompt-based arguments
parser.add_argument('--num_prompts', default=8, type=int, help='Number of prompts in the bank')
parser.add_argument('--prompt_mix_ratio', default=0.3, type=float, help='Mixing ratio for prompts')
parser.add_argument('--run_both', action='store_true', help='Run both regular incremental and prompt-based models')
parser.add_argument('--extract_prompts', action='store_true', help='Extract prompts from transformer layers')
parser.add_argument('--prompt_layer_idx', default=0, type=int, help='Layer index to extract prompts from')
parser.add_argument('--freeze_prompts', action='store_true', help='Freeze prompts after initial training')
parser.add_argument('--num_new_prompts', default=4, type=int, help='Number of new prompts to be added in each epoch')
parser.add_argument('--frozen_prompt_count', default=1024, type=int, 
                    help='Number of prompts to keep frozen during incremental learning')

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def check_dataset_validity(dataset):
    """
    Check if dataset has valid test users
    """
    [train, valid, test, usernum, itemnum] = dataset
    valid_test_users = []
    for u in train:
        if u in test and len(test[u]) > 0 and len(train[u]) > 0:
            valid_test_users.append(u)
    return len(valid_test_users) > 0

def prepare_time_sliced_data(args, output_dir):
    """
    Load or create time-sliced data
    
    Args:
        args: Command line arguments
        output_dir: Output directory for saving data
        
    Returns:
        TimeSlicedData object and list of time slices
    """
    # Set number of slices from slice ratios
    args.num_slices = len(args.slice_ratios)
    
    # Check if we should load pre-saved data splits
    time_data = None
    time_slices = None
    data_splits_path = os.path.join(output_dir, 'data_splits.pkl')
    
    if args.load_splits:
        # Load from specified path
        print(f"Loading data splits from {args.load_splits}...")
        try:
            with open(args.load_splits, 'rb') as f:
                saved_data = pickle.load(f)
                time_data = saved_data['time_data']
                time_slices = saved_data['time_slices']
                print("Data splits loaded successfully!")
        except Exception as e:
            print(f"Error loading data splits: {e}")
            print("Will create new data splits.")
            time_data = None
    
    # If not loaded, create new splits
    if time_data is None:
        print("Creating new data splits...")
        data_path = f'SSEPI/data/{args.dataset}.txt'
        time_data = TimeSlicedData(
            data_path, 
            num_slices=args.num_slices,
            slice_ratios=args.slice_ratios,
            min_interactions=args.min_interactions
        )
        time_slices = time_data.load_data()
        
        # Save data splits if requested
        if args.save_splits:
            print(f"Saving data splits to {data_splits_path}...")
            saved_data = {
                'time_data': time_data,
                'time_slices': time_slices,
                'args': {
                    'slice_ratios': args.slice_ratios,
                    'min_interactions': args.min_interactions,
                    'seed': args.seed
                }
            }
            with open(data_splits_path, 'wb') as f:
                pickle.dump(saved_data, f)
            print("Data splits saved successfully!")
    
    print(f"Total users: {len(time_data.all_users)}, Valid users: {len(time_data.valid_users)}")
    print(f"Total items: {time_data.itemnum}")
    
    return time_data, time_slices

def run_incremental_learning(args, time_data, output_dir, log_file):
    """
    Run incremental learning experiment with knowledge distillation
    
    Args:
        args: Command line arguments
        time_data: TimeSlicedData object
        output_dir: Output directory for saving results
        log_file: Log file object
        
    Returns:
        Dictionary of results
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Prepare T1 (first slice)
    t1_data = time_data.prepare_slice(0, include_previous=False)
    [t1_user_train, t1_user_valid, t1_user_test, usernum, itemnum] = t1_data
    
    print(f"T1 statistics:")
    print(f"Train users: {len(t1_user_train)}")
    cc = 0.0
    for u in t1_user_train:
        cc += len(t1_user_train[u])
    print(f"Average sequence length: {cc / len(t1_user_train) if len(t1_user_train) > 0 else 0}")
    
    # Initialize base model
    print("\n=== Training Base Model on T1 ===")
    base_model = SASRec(usernum, itemnum, args).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # T1 sampler
    t1_sampler = WarpSampler(t1_user_train, usernum, itemnum,
                           batch_size=args.batch_size, maxlen=args.maxlen,
                           threshold_user=args.threshold_user,
                           threshold_item=args.threshold_item,
                           n_workers=3, device=device)
    
    # Train on T1
    num_batch = max(len(t1_user_train) // args.batch_size, 1)
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        for step in range(num_batch):
            u, seq, pos, neg = t1_sampler.next_batch()
            
            base_optimizer.zero_grad()
            loss, attention_weights, auc, l2_loss = base_model(u, seq, pos, neg, is_training=True)
            loss.backward()
            base_optimizer.step()
            
        if epoch % args.print_freq == 0:
            t1 = time.time() - t0
            
            if check_dataset_validity(t1_data):
                t_test = evaluate(base_model, t1_data, args, device)
                log_file.write(f"Base model epoch {epoch}: NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}\n")
                log_file.flush()
                print(f"[Base model epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}, Time={t1:.1f}s")
            else:
                print(f"[Base model epoch {epoch}] No valid test users found for evaluation. Skipping.")
            
            t0 = time.time()
    
    # Close T1 sampler
    t1_sampler.close()
    
    # Save base model
    torch.save(base_model, os.path.join(output_dir, 'base_model.pt'))
    print(f"Base model saved to {os.path.join(output_dir, 'base_model.pt')}")
    
    # Get items from T1
    t1_users, t1_items = time_data.get_slice_data(0)
    print(f"T1 unique users: {len(t1_users)}, unique items: {len(t1_items)}")
    
    # Initialize incremental model from base model
    print("\n=== Initializing Incremental Model ===")
    incremental_model = IncrementalSASRec(usernum, itemnum, args).to(device)
    
    # Copy weights from base model
    incremental_model.load_state_dict(base_model.state_dict())
    
    # Create replay buffer
    replay_buffer = ExperienceReplay(args.buffer_size, args.replay_ratio)
    
    # Create dataloader for T1 buffer
    t1_buffer = time_data.create_replay_buffer(0, args.buffer_size, max_seq_length=1)
    replay_buffer.update_buffer(t1_buffer)
    
    # Results storage
    results = {
        'base_model': {},
        'incremental_model': {}
    }
    
    # Evaluate base model on all slices
    print("\n=== Evaluating Base Model on All Slices ===")
    for i in range(args.num_slices):
        slice_data = time_data.prepare_slice(i, include_previous=False)
        if check_dataset_validity(slice_data):
            ndcg, hr = evaluate(base_model, slice_data, args, device)
            results['base_model'][f'slice_{i}'] = {'ndcg': ndcg, 'hr': hr}
            print(f"Base model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}")
            log_file.write(f"Base model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}\n")
        else:
            results['base_model'][f'slice_{i}'] = {'ndcg': 0.0, 'hr': 0.0}
            print(f"Base model on slice {i}: No valid test users found. Skipping.")
    
    # Incremental learning on subsequent slices
    for slice_idx in range(1, args.num_slices):
        print(f"\n=== Incremental Learning on Slice {slice_idx} ===")
        
        # Get slice data
        slice_data = time_data.prepare_slice(slice_idx, include_previous=False)
        [slice_user_train, slice_user_valid, slice_user_test, _, _] = slice_data
        
        # Get slice items
        slice_users, slice_items = time_data.get_slice_data(slice_idx)
        print(f"Slice {slice_idx} unique users: {len(slice_users)}, unique items: {len(slice_items)}")
        
        # Update item sets for tracking
        incremental_model.update_item_sets(t1_items, slice_items)
        
        # Create sampler for this slice
        slice_sampler = WarpSampler(slice_user_train, usernum, itemnum,
                                  batch_size=args.batch_size, maxlen=args.maxlen,
                                  threshold_user=args.threshold_user,
                                  threshold_item=args.threshold_item,
                                  n_workers=3, device=device)
        
        # Create optimizer with lower learning rate
        inc_lr = args.lr * 0.1  # Lower learning rate for stability
        inc_optimizer = torch.optim.Adam(incremental_model.parameters(), lr=inc_lr, betas=(0.9, 0.98))
        
        # Create new replay buffer for this slice
        slice_buffer = time_data.create_replay_buffer(slice_idx, args.buffer_size, max_seq_length=1)
        replay_buffer.update_buffer(slice_buffer)
        
        # Train for fewer epochs
        inc_epochs = args.num_epochs // 4  # Fewer epochs for incremental learning
        t0 = time.time()

        for epoch in range(1, inc_epochs + 1):
            # Determine number of batches
            num_batch = max(len(slice_user_train) // args.batch_size, 1)
            
            for step in range(num_batch):
                # Get batch from current slice
                u, seq, pos, neg = slice_sampler.next_batch()
                
                # Optionally mix with replay samples
                if random.random() < args.replay_ratio and replay_buffer.buffer:
                    replay_batch = replay_buffer.sample_batch(args.batch_size)
                    if replay_batch:
                        r_users, r_seqs, r_poss, r_negs = replay_batch
                        
                        # Convert all to numpy arrays first
                        r_users_array = np.array(r_users)
                        r_poss_array = np.array(r_poss)
                        r_negs_array = np.array(r_negs)
                        
                        # Pad sequences properly
                        padded_seqs = np.zeros((len(r_seqs), args.maxlen), dtype=np.int64)
                        for i, seq_item in enumerate(r_seqs):
                            # Make sure we handle it as a sequence
                            if not isinstance(seq_item, (list, np.ndarray)):
                                seq_item = [seq_item]
                            
                            # Place at the end of the padded sequence
                            seq_length = min(len(seq_item), args.maxlen)
                            if seq_length > 0:
                                padded_seqs[i, args.maxlen - seq_length:] = np.array(seq_item[-seq_length:])
                        
                        # Get the shapes of the original tensors for proper reshaping
                        u_shape = u.shape
                        seq_shape = seq.shape
                        pos_shape = pos.shape
                        neg_shape = neg.shape
                        
                        # Convert to tensors, ensuring they have the right dtype and shape
                        r_users_tensor = torch.tensor(r_users_array, dtype=torch.long, device=device)
                        r_seqs_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=device)
                        
                        # Reshape pos and neg to match the original tensors
                        if len(pos_shape) > 1:  # If pos is 2D
                            # Reshape to match (batch_size, maxlen) if that's the shape
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device).reshape(-1, 1)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device).reshape(-1, 1)
                            
                            # Expand to match maxlen if needed
                            if pos_shape[1] > 1:
                                r_poss_tensor = r_poss_tensor.expand(-1, pos_shape[1])
                                r_negs_tensor = r_negs_tensor.expand(-1, neg_shape[1])
                        else:
                            # If pos is 1D, keep it 1D
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device)
                        
                        # Combine current and replay
                        u = torch.cat([u, r_users_tensor])
                        seq = torch.cat([seq, r_seqs_tensor])
                        pos = torch.cat([pos, r_poss_tensor])
                        neg = torch.cat([neg, r_negs_tensor])
                
                # Update model with knowledge distillation
                inc_optimizer.zero_grad()
                loss, _, auc, _ = incremental_model(u, seq, pos, neg, is_training=True, base_model=base_model)
                loss.backward()
                inc_optimizer.step()
            
            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                
                # Evaluate on current slice
                if check_dataset_validity(slice_data):
                    t_test = evaluate(incremental_model, slice_data, args, device)
                    log_file.write(f"Slice {slice_idx}, epoch {epoch}: NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}\n")
                    log_file.flush()
                    print(f"[Slice {slice_idx}, epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}, Time={t1:.1f}s")
                else:
                    print(f"[Slice {slice_idx}, epoch {epoch}] No valid test users found. Skipping.")
                
                t0 = time.time()
        
        # Close sampler
        slice_sampler.close()
        
        # Save incremental model
        torch.save(incremental_model, os.path.join(output_dir, f'incremental_model_slice_{slice_idx}.pt'))
        
        # Evaluate incremental model on all slices
        print(f"\n=== Evaluating Incremental Model After Slice {slice_idx} ===")
        for i in range(args.num_slices):
            eval_data = time_data.prepare_slice(i, include_previous=False)
            if check_dataset_validity(eval_data):
                ndcg, hr = evaluate(incremental_model, eval_data, args, device)
                results['incremental_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': ndcg, 'hr': hr}
                print(f"Incremental model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}")
                log_file.write(f"Incremental model (after slice {slice_idx}) on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}\n")
            else:
                results['incremental_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': 0.0, 'hr': 0.0}
                print(f"Incremental model on slice {i}: No valid test users found. Skipping.")
    
    # Save final results
    with open(os.path.join(output_dir, 'results_incremental.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== Incremental Learning Experiment Completed ===")
    
    return results, base_model, t1_items


def freeze_prompts(model, freeze=True):
    """
    Freeze or unfreeze prompts
    
    Args:
        model: Model with prompt_bank
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    for param in model.prompt_bank.parameters():
        param.requires_grad = not freeze
    
    if freeze:
        print("Prompts are now frozen")
    else:
        print("Prompts are now trainable")

def run_prompt_incremental_learning(args, time_data, base_model, t1_items, output_dir, log_file):
    """
    Run prompt-based incremental learning experiment with three-phase training for each slice
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Prepare T1 (first slice)
    t1_data = time_data.prepare_slice(0, include_previous=False)
    [t1_user_train, t1_user_valid, t1_user_test, usernum, itemnum] = t1_data
    
    # Initialize prompt-based model
    print("\n=== Training Prompt-Based Model on T1 ===")
    prompt_model = PromptBaseSASRec(usernum, itemnum, args).to(device)

    # Initialize prompt manager
    prompt_manager = PromptManager(prompt_model, args.num_prompts)

    # Use three-phase training for T1
    prompt_model.train_with_separate_prompt_phases(t1_user_train, t1_user_valid, t1_user_test, args, device)
    
    # Implement hybrid prompt freezing strategy
    print("=== Implementing hybrid prompt freezing strategy ===")
    prompt_model.ensure_hybrid_prompt_freezing()

    # Save prompt-based model
    torch.save(prompt_model, os.path.join(output_dir, 'prompt_base_model.pt'))
    print(f"Prompt-based model saved to {os.path.join(output_dir, 'prompt_base_model.pt')}")

    # Create a FROZEN COPY of the T1 model for ensemble prediction
    frozen_t1_model = copy.deepcopy(prompt_model)
    
    # Ensure the frozen model is in eval mode and has no gradients
    frozen_t1_model.eval()
    for param in frozen_t1_model.parameters():
        param.requires_grad = False
    
    # Save the frozen model separately
    torch.save(frozen_t1_model, os.path.join(output_dir, 'frozen_t1_model.pt'))
    print(f"Frozen T1 model saved to {os.path.join(output_dir, 'frozen_t1_model.pt')}")

    # Get items from T1
    t1_users, t1_items = time_data.get_slice_data(0)
    print(f"T1 unique users: {len(t1_users)}, unique items: {len(t1_items)}")

    # Update item sets
    prompt_manager.update_item_sets(t1_items, set())
    
    # Analyze prompt usage after T1 training
    prompt_analysis = prompt_manager.analyze_prompts(t1_data, args, device)
    with open(os.path.join(output_dir, 'prompt_analysis_t1.pkl'), 'wb') as f:
        pickle.dump(prompt_analysis, f)
    
    # Results storage
    results = {
        'prompt_model': {},
        'ensemble_model': {}  # Add this for storing ensemble results
    }
    
    # Evaluate prompt-based model on all slices
    print("\n=== Evaluating Prompt-Based Model on All Slices ===")
    for i in range(args.num_slices):
        slice_data = time_data.prepare_slice(i, include_previous=False)
        if check_dataset_validity(slice_data):
            ndcg, hr = evaluate(prompt_model, slice_data, args, device)
            results['prompt_model'][f'slice_{i}'] = {'ndcg': ndcg, 'hr': hr}
            print(f"Prompt-based model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}")
            log_file.write(f"Prompt-based model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}\n")
        else:
            results['prompt_model'][f'slice_{i}'] = {'ndcg': 0.0, 'hr': 0.0}
            print(f"Prompt-based model on slice {i}: No valid test users found. Skipping.")
    
    # Create replay buffer
    replay_buffer = ExperienceReplay(args.buffer_size, args.replay_ratio)
    
    # Create dataloader for T1 buffer
    t1_buffer = time_data.create_replay_buffer(0, args.buffer_size, max_seq_length=1)
    replay_buffer.update_buffer(t1_buffer)
    
    # Incremental learning on subsequent slices
    for slice_idx in range(1, args.num_slices):
        print(f"\n=== Prompt-Based Incremental Learning on Slice {slice_idx} ===")
        
        # Ensure hybrid prompt freezing during incremental learning
        prompt_model.ensure_hybrid_prompt_freezing()
        
        # Get slice data
        slice_data = time_data.prepare_slice(slice_idx, include_previous=False)
        [slice_user_train, slice_user_valid, slice_user_test, _, _] = slice_data
        
        # Generate new prompts from current slice if we're past the first incremental update
        if slice_idx > 1:
            new_prompt_count = prompt_model.generate_new_prompts(slice_data, num_new_prompts=args.num_new_prompts, device=device)
            
            # Re-apply hybrid freezing after adding new prompts
            prompt_model.ensure_hybrid_prompt_freezing()
        
        # Get slice items
        slice_users, slice_items = time_data.get_slice_data(slice_idx)
        print(f"Slice {slice_idx} unique users: {len(slice_users)}, unique items: {len(slice_items)}")
        
        # Update item sets for prompt manager
        prompt_manager.update_item_sets(t1_items, slice_items)

        # Create new replay buffer for this slice
        slice_buffer = time_data.create_replay_buffer(slice_idx, args.buffer_size, max_seq_length=1)
        replay_buffer.update_buffer(slice_buffer)

        # Determine number of epochs for each phase
        inc_epochs_total = args.num_epochs // 4  # Fewer epochs for incremental learning
        phase1_epochs = inc_epochs_total // 3
        phase2_epochs = inc_epochs_total // 3
        phase3_epochs = inc_epochs_total - phase1_epochs - phase2_epochs
        
        # ====================================================================
        # PHASE 1: Train base model with minimal prompt influence
        # ====================================================================
        print(f"=== Phase 1: Training on slice {slice_idx} with minimal prompt influence ===")
        
        # Store original prompt_mix_ratio
        original_mix_ratio = prompt_model.prompt_mix_ratio
        
        # Set a very low mix ratio for Phase 1
        prompt_model.prompt_mix_ratio = 0.05
        
        # Create sampler for this slice
        slice_sampler = WarpSampler(slice_user_train, usernum, itemnum,
                                  batch_size=args.batch_size, maxlen=args.maxlen,
                                  threshold_user=args.threshold_user,
                                  threshold_item=args.threshold_item,
                                  n_workers=3, device=device)
        
        # Create optimizer for all parameters
        phase1_lr = args.lr * 0.1  # Lower learning rate for incremental learning
        phase1_optimizer = torch.optim.Adam(prompt_model.parameters(), lr=phase1_lr, betas=(0.9, 0.98))
        
        # Determine number of batches
        num_batch = max(len(slice_user_train) // args.batch_size, 1)
        t0 = time.time()
        
        for epoch in range(1, phase1_epochs + 1):
            for step in range(num_batch):
                # Get batch from current slice
                u, seq, pos, neg = slice_sampler.next_batch()
                
                # Optionally mix with replay samples
                if random.random() < args.replay_ratio and replay_buffer.buffer:
                    replay_batch = replay_buffer.sample_batch(args.batch_size)
                    if replay_batch:
                        r_users, r_seqs, r_poss, r_negs = replay_batch
                        
                        # Convert all to numpy arrays first
                        r_users_array = np.array(r_users)
                        r_poss_array = np.array(r_poss)
                        r_negs_array = np.array(r_negs)
                        
                        # Pad sequences properly
                        padded_seqs = np.zeros((len(r_seqs), args.maxlen), dtype=np.int64)
                        for i, seq_item in enumerate(r_seqs):
                            # Make sure we handle it as a sequence
                            if not isinstance(seq_item, (list, np.ndarray)):
                                seq_item = [seq_item]
                            
                            # Place at the end of the padded sequence
                            seq_length = min(len(seq_item), args.maxlen)
                            if seq_length > 0:
                                padded_seqs[i, args.maxlen - seq_length:] = np.array(seq_item[-seq_length:])
                        
                        # Get the shapes of the original tensors for proper reshaping
                        u_shape = u.shape
                        seq_shape = seq.shape
                        pos_shape = pos.shape
                        neg_shape = neg.shape
                        
                        # Convert to tensors, ensuring they have the right dtype and shape
                        r_users_tensor = torch.tensor(r_users_array, dtype=torch.long, device=device)
                        r_seqs_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=device)
                        
                        # Reshape pos and neg to match the original tensors
                        if len(pos_shape) > 1:  # If pos is 2D
                            # Reshape to match (batch_size, maxlen) if that's the shape
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device).reshape(-1, 1)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device).reshape(-1, 1)
                            
                            # Expand to match maxlen if needed
                            if pos_shape[1] > 1:
                                r_poss_tensor = r_poss_tensor.expand(-1, pos_shape[1])
                                r_negs_tensor = r_negs_tensor.expand(-1, neg_shape[1])
                        else:
                            # If pos is 1D, keep it 1D
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device)
                        
                        # Combine current and replay
                        u = torch.cat([u, r_users_tensor])
                        seq = torch.cat([seq, r_seqs_tensor])
                        pos = torch.cat([pos, r_poss_tensor])
                        neg = torch.cat([neg, r_negs_tensor])
                
                # Update model
                phase1_optimizer.zero_grad()
                loss, _, auc, _ = prompt_model(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase1_optimizer.step()
            
            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                
                # Evaluate on current slice
                if check_dataset_validity(slice_data):
                    t_test = evaluate(prompt_model, slice_data, args, device)
                    log_file.write(f"[Slice {slice_idx}, Phase 1, epoch {epoch}]: NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}\n")
                    log_file.flush()
                    print(f"[Slice {slice_idx}, Phase 1, epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}, Time={t1:.1f}s")
                else:
                    print(f"[Slice {slice_idx}, Phase 1, epoch {epoch}] No valid test users found. Skipping.")
                
                t0 = time.time()
        
        # ====================================================================
        # PHASE 2: Train prompts with frozen base model
        # ====================================================================
        print(f"=== Phase 2: Training prompts for slice {slice_idx} with base model frozen ===")
        
        # Restore original prompt mixing ratio
        prompt_model.prompt_mix_ratio = original_mix_ratio
        
        # Freeze all non-prompt parameters
        for name, param in prompt_model.named_parameters():
            if 'prompt_bank' not in name:
                param.requires_grad = False
        
        # Create optimizer for prompt parameters only
        phase2_lr = args.lr * 0.05  # Even lower learning rate for prompt specialization
        phase2_optimizer = torch.optim.Adam(
            prompt_model.prompt_bank.parameters(),
            lr=phase2_lr, betas=(0.9, 0.98)
        )
        
        t0 = time.time()
        
        for epoch in range(1, phase2_epochs + 1):
            for step in range(num_batch):
                # Get batch from current slice
                u, seq, pos, neg = slice_sampler.next_batch()
                
                # Optionally mix with replay samples (same code as before)
                if random.random() < args.replay_ratio and replay_buffer.buffer:
                    replay_batch = replay_buffer.sample_batch(args.batch_size)
                    if replay_batch:
                        r_users, r_seqs, r_poss, r_negs = replay_batch
                        
                        # Convert all to numpy arrays first
                        r_users_array = np.array(r_users)
                        r_poss_array = np.array(r_poss)
                        r_negs_array = np.array(r_negs)
                        
                        # Pad sequences properly
                        padded_seqs = np.zeros((len(r_seqs), args.maxlen), dtype=np.int64)
                        for i, seq_item in enumerate(r_seqs):
                            # Make sure we handle it as a sequence
                            if not isinstance(seq_item, (list, np.ndarray)):
                                seq_item = [seq_item]
                            
                            # Place at the end of the padded sequence
                            seq_length = min(len(seq_item), args.maxlen)
                            if seq_length > 0:
                                padded_seqs[i, args.maxlen - seq_length:] = np.array(seq_item[-seq_length:])
                        
                        # Get the shapes of the original tensors for proper reshaping
                        u_shape = u.shape
                        seq_shape = seq.shape
                        pos_shape = pos.shape
                        neg_shape = neg.shape
                        
                        # Convert to tensors, ensuring they have the right dtype and shape
                        r_users_tensor = torch.tensor(r_users_array, dtype=torch.long, device=device)
                        r_seqs_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=device)
                        
                        # Reshape pos and neg to match the original tensors
                        if len(pos_shape) > 1:  # If pos is 2D
                            # Reshape to match (batch_size, maxlen) if that's the shape
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device).reshape(-1, 1)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device).reshape(-1, 1)
                            
                            # Expand to match maxlen if needed
                            if pos_shape[1] > 1:
                                r_poss_tensor = r_poss_tensor.expand(-1, pos_shape[1])
                                r_negs_tensor = r_negs_tensor.expand(-1, neg_shape[1])
                        else:
                            # If pos is 1D, keep it 1D
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device)
                        
                        # Combine current and replay
                        u = torch.cat([u, r_users_tensor])
                        seq = torch.cat([seq, r_seqs_tensor])
                        pos = torch.cat([pos, r_poss_tensor])
                        neg = torch.cat([neg, r_negs_tensor])
                
                # Update model
                phase2_optimizer.zero_grad()
                loss, _, auc, _ = prompt_model(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase2_optimizer.step()
            
            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                
                # Evaluate on current slice
                if check_dataset_validity(slice_data):
                    t_test = evaluate(prompt_model, slice_data, args, device)
                    log_file.write(f"[Slice {slice_idx}, Phase 2, epoch {epoch}]: NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}\n")
                    log_file.flush()
                    print(f"[Slice {slice_idx}, Phase 2, epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}, Time={t1:.1f}s")
                else:
                    print(f"[Slice {slice_idx}, Phase 2, epoch {epoch}] No valid test users found. Skipping.")
                
                t0 = time.time()
        
        # ====================================================================
        # PHASE 3: Fine-tune entire model
        # ====================================================================
        print(f"=== Phase 3: Fine-tuning all parameters for slice {slice_idx} ===")
        
        # Get the frozen_prompt_count
        frozen_count = getattr(args, 'frozen_prompt_count', 1024)
        
        # Unfreeze all parameters except permanently frozen prompts
        for name, param in prompt_model.named_parameters():
            # Don't unfreeze parameters that should be permanently frozen
            if name.startswith('prompt_bank.prompts'):
                prompt_idx = int(name.split('.')[2]) if len(name.split('.')) > 2 else -1
                if prompt_idx >= 0 and prompt_idx < frozen_count:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True
        
        # Reapply hybrid prompt freezing to ensure correct gradient masking
        prompt_model.ensure_hybrid_prompt_freezing()
        # Create optimizer with very low learning rate
        phase3_lr = args.lr * 0.01  # Very low learning rate for fine-tuning
        phase3_optimizer = torch.optim.Adam(
            [p for p in prompt_model.parameters() if p.requires_grad],
            lr=phase3_lr, betas=(0.9, 0.98)
        )
        
        t0 = time.time()
        
        for epoch in range(1, phase3_epochs + 1):
            for step in range(num_batch):
                # Get batch from current slice
                u, seq, pos, neg = slice_sampler.next_batch()
                
                # Optionally mix with replay samples (same code as before)
                if random.random() < args.replay_ratio and replay_buffer.buffer:
                    replay_batch = replay_buffer.sample_batch(args.batch_size)
                    if replay_batch:
                        r_users, r_seqs, r_poss, r_negs = replay_batch
                        
                        # Convert all to numpy arrays first
                        r_users_array = np.array(r_users)
                        r_poss_array = np.array(r_poss)
                        r_negs_array = np.array(r_negs)
                        
                        # Pad sequences properly
                        padded_seqs = np.zeros((len(r_seqs), args.maxlen), dtype=np.int64)
                        for i, seq_item in enumerate(r_seqs):
                            # Make sure we handle it as a sequence
                            if not isinstance(seq_item, (list, np.ndarray)):
                                seq_item = [seq_item]
                            
                            # Place at the end of the padded sequence
                            seq_length = min(len(seq_item), args.maxlen)
                            if seq_length > 0:
                                padded_seqs[i, args.maxlen - seq_length:] = np.array(seq_item[-seq_length:])
                        
                        # Get the shapes of the original tensors for proper reshaping
                        u_shape = u.shape
                        seq_shape = seq.shape
                        pos_shape = pos.shape
                        neg_shape = neg.shape
                        
                        # Convert to tensors, ensuring they have the right dtype and shape
                        r_users_tensor = torch.tensor(r_users_array, dtype=torch.long, device=device)
                        r_seqs_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=device)
                        
                        # Reshape pos and neg to match the original tensors
                        if len(pos_shape) > 1:  # If pos is 2D
                            # Reshape to match (batch_size, maxlen) if that's the shape
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device).reshape(-1, 1)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device).reshape(-1, 1)
                            
                            # Expand to match maxlen if needed
                            if pos_shape[1] > 1:
                                r_poss_tensor = r_poss_tensor.expand(-1, pos_shape[1])
                                r_negs_tensor = r_negs_tensor.expand(-1, neg_shape[1])
                        else:
                            # If pos is 1D, keep it 1D
                            r_poss_tensor = torch.tensor(r_poss_array, dtype=torch.long, device=device)
                            r_negs_tensor = torch.tensor(r_negs_array, dtype=torch.long, device=device)
                        
                        # Combine current and replay
                        u = torch.cat([u, r_users_tensor])
                        seq = torch.cat([seq, r_seqs_tensor])
                        pos = torch.cat([pos, r_poss_tensor])
                        neg = torch.cat([neg, r_negs_tensor])
                
                # Update model
                phase3_optimizer.zero_grad()
                loss, _, auc, _ = prompt_model(u, seq, pos, neg, is_training=True)
                loss.backward()
                phase3_optimizer.step()
            
            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                
                # Evaluate on current slice
                if check_dataset_validity(slice_data):
                    t_test = evaluate(prompt_model, slice_data, args, device)
                    log_file.write(f"[Slice {slice_idx}, Phase 3, epoch {epoch}]: NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}\n")
                    log_file.flush()
                    print(f"[Slice {slice_idx}, Phase 3, epoch {epoch}] NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f}, Loss={loss:.4f}, Time={t1:.1f}s")
                else:
                    print(f"[Slice {slice_idx}, Phase 3, epoch {epoch}] No valid test users found. Skipping.")
                
                t0 = time.time()
        
        # Close sampler
        slice_sampler.close()
        
        # Save prompt model
        torch.save(prompt_model, os.path.join(output_dir, f'prompt_model_slice_{slice_idx}.pt'))
        
        # Analyze prompt usage after this slice
        prompt_analysis = prompt_manager.analyze_prompts(slice_data, args, device)
        with open(os.path.join(output_dir, f'prompt_analysis_t{slice_idx+1}.pkl'), 'wb') as f:
            pickle.dump(prompt_analysis, f)
        
        # Evaluate prompt model on all slices
        print(f"\n=== Evaluating Prompt Model After Slice {slice_idx} ===")
        for i in range(args.num_slices):
            eval_data = time_data.prepare_slice(i, include_previous=False)
            if check_dataset_validity(eval_data):
                ndcg, hr = evaluate(prompt_model, eval_data, args, device)
                results['prompt_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': ndcg, 'hr': hr}
                print(f"Prompt model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}")
                log_file.write(f"Prompt model (after slice {slice_idx}) on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}\n")
            else:
                results['prompt_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': 0.0, 'hr': 0.0}
                print(f"Prompt model on slice {i}: No valid test users found. Skipping.")

        # Create the ensemble model with adaptive weighting
        ensemble_model = EnsemblePromptSASRec(frozen_t1_model, prompt_model, alpha=0.2)
        
        # Save ensemble model
        torch.save(ensemble_model, os.path.join(output_dir, f'ensemble_model_slice_{slice_idx}.pt'))
        
        # Evaluate ensemble model on all slices
        print(f"\n=== Evaluating Ensemble Model After Slice {slice_idx} ===")
        for i in range(args.num_slices):
            eval_data = time_data.prepare_slice(i, include_previous=False)
            if check_dataset_validity(eval_data):
                ndcg, hr = evaluate(ensemble_model, eval_data, args, device)
                results['ensemble_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': ndcg, 'hr': hr}
                print(f"Ensemble model on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}")
                log_file.write(f"Ensemble model (after slice {slice_idx}) on slice {i}: NDCG={ndcg:.4f}, HR={hr:.4f}\n")
            else:
                results['ensemble_model'][f'after_slice_{slice_idx}_eval_on_{i}'] = {'ndcg': 0.0, 'hr': 0.0}
                print(f"Ensemble model on slice {i}: No valid test users found. Skipping.")
    
    # Save final results
    with open(os.path.join(output_dir, 'results_prompt.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== Prompt-Based Incremental Learning Experiment Completed ===")
    
    return results

def compare_results(results_incremental, results_prompt, args, output_dir, log_file):
    """
    Compare results from both models
    
    Args:
        results_incremental: Results from incremental model
        results_prompt: Results from prompt-based model
        args: Command line arguments
        output_dir: Output directory
        log_file: Log file object
    """
    print("\n=== Comparing Incremental and Prompt-Based Models ===")
    log_file.write("\n=== Comparison Results ===\n")
    
    # Find the maximum slice index for which we have results
    max_slice = max([int(k.split('_')[2]) for k in results_incremental['incremental_model'].keys() 
                   if k.startswith('after_slice_')])
    
    # Compare performance on each slice after training on final slice
    comparison_table = "Slice\tBase NDCG\tBase HR\tIncremental NDCG\tIncremental HR\tPrompt NDCG\tPrompt HR\tNDCG Diff\tHR Diff\n"
    comparison_table += "-" * 100 + "\n"
    
    avg_inc_ndcg = 0.0
    avg_prompt_ndcg = 0.0
    avg_inc_hr = 0.0
    avg_prompt_hr = 0.0
    
    for i in range(args.num_slices):
        base_ndcg = results_incremental['base_model'][f'slice_{i}']['ndcg']
        base_hr = results_incremental['base_model'][f'slice_{i}']['hr']
        
        inc_ndcg = results_incremental['incremental_model'][f'after_slice_{max_slice}_eval_on_{i}']['ndcg']
        inc_hr = results_incremental['incremental_model'][f'after_slice_{max_slice}_eval_on_{i}']['hr']
        
        prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{max_slice}_eval_on_{i}']['ndcg']
        prompt_hr = results_prompt['prompt_model'][f'after_slice_{max_slice}_eval_on_{i}']['hr']
        
        # Calculate differences
        ndcg_diff = prompt_ndcg - inc_ndcg
        hr_diff = prompt_hr - inc_hr
        
        # Add to running averages
        avg_inc_ndcg += inc_ndcg
        avg_prompt_ndcg += prompt_ndcg
        avg_inc_hr += inc_hr
        avg_prompt_hr += prompt_hr
        
        # Add row to table
        comparison_table += f"T{i+1}\t{base_ndcg:.4f}\t{base_hr:.4f}\t{inc_ndcg:.4f}\t{inc_hr:.4f}\t{prompt_ndcg:.4f}\t{prompt_hr:.4f}\t{ndcg_diff:+.4f}\t{hr_diff:+.4f}\n"
    
    # Calculate averages
    avg_inc_ndcg /= args.num_slices
    avg_prompt_ndcg /= args.num_slices
    avg_inc_hr /= args.num_slices
    avg_prompt_hr /= args.num_slices
    
    # Add averages to table
    comparison_table += "-" * 100 + "\n"
    comparison_table += f"Avg\t-\t-\t{avg_inc_ndcg:.4f}\t{avg_inc_hr:.4f}\t{avg_prompt_ndcg:.4f}\t{avg_prompt_hr:.4f}\t{avg_prompt_ndcg-avg_inc_ndcg:+.4f}\t{avg_prompt_hr-avg_inc_hr:+.4f}\n"
    
    # Add performance on first slice (knowledge retention)
    t1_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{max_slice}_eval_on_0']['ndcg']
    t1_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{max_slice}_eval_on_0']['ndcg']
    t1_base_ndcg = results_incremental['base_model']['slice_0']['ndcg']
    
    t1_inc_retention = t1_inc_ndcg / t1_base_ndcg if t1_base_ndcg > 0 else 0
    t1_prompt_retention = t1_prompt_ndcg / t1_base_ndcg if t1_base_ndcg > 0 else 0
    
    comparison_table += "\n=== Knowledge Retention (T1 Performance) ===\n"
    comparison_table += f"Incremental Model Retention: {t1_inc_retention:.2%}\n"
    comparison_table += f"Prompt Model Retention: {t1_prompt_retention:.2%}\n"
    comparison_table += f"Retention Improvement: {t1_prompt_retention - t1_inc_retention:+.2%}\n"
    
    # Add performance on latest slice (new knowledge acquisition)
    latest_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{max_slice}_eval_on_{max_slice}']['ndcg']
    latest_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{max_slice}_eval_on_{max_slice}']['ndcg']
    latest_base_ndcg = results_incremental['base_model'][f'slice_{max_slice}']['ndcg']
    
    comparison_table += f"\n=== New Knowledge Acquisition (T{max_slice+1} Performance) ===\n"
    comparison_table += f"Incremental Model NDCG: {latest_inc_ndcg:.4f}\n"
    comparison_table += f"Prompt Model NDCG: {latest_prompt_ndcg:.4f}\n"
    comparison_table += f"Improvement: {latest_prompt_ndcg - latest_inc_ndcg:+.4f}\n"
    
    # Print and log comparison table
    print(comparison_table)
    log_file.write(comparison_table)
    
    # Save comparison to file
    with open(os.path.join(output_dir, 'comparison_results.txt'), 'w') as f:
        f.write(comparison_table)
    
    # Combine results for easy reference
    combined_results = {
        'base_model': results_incremental['base_model'],
        'incremental_model': results_incremental['incremental_model'],
        'prompt_model': results_prompt['prompt_model']
    }
    
    with open(os.path.join(output_dir, 'combined_results.pkl'), 'wb') as f:
        pickle.dump(combined_results, f)
    
    print(f"Comparison saved to {os.path.join(output_dir, 'comparison_results.txt')}")

    generate_detailed_comparison(results_incremental, results_prompt, args, output_dir, log_file)


def run_comparison(args):
    """
    Run both incremental and prompt-based models and compare results
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    output_dir = args.dataset + '_' + args.train_dir + '_comparison'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Log file
    log_file = open(os.path.join(output_dir, 'comparison_log.txt'), 'w')
    log_file.write(f"Random seed: {args.seed}\n")
    
    # Load time sliced data
    time_data, time_slices = prepare_time_sliced_data(args, output_dir)
    
    # Run incremental learning
    print("\n=== Running Standard Incremental Learning ===")
    results_incremental, base_model, t1_items = run_incremental_learning(args, time_data, output_dir, log_file)
    
    # Run prompt-based incremental learning with three-phase approach
    print("\n=== Running Prompt-Based Incremental Learning with Three-Phase Approach ===")
    results_prompt = run_prompt_incremental_learning(args, time_data, base_model, t1_items, output_dir, log_file)
    
    # Compare results
    compare_results(results_incremental, results_prompt, args, output_dir, log_file)
    
    # Close log file
    log_file.close()
    
    print("\n=== Comparison Completed ===")
    print(f"Results saved to {output_dir}")

def generate_detailed_comparison(results_incremental, results_prompt, args, output_dir, log_file):
    """
    Generate a detailed comparison of model performance after each training slice
    """
    print("\n=== Detailed Slice-by-Slice Comparison ===")
    log_file.write("\n=== Detailed Slice-by-Slice Comparison ===\n")
    
    # For each training slice (except T1 which is the base)
    for train_slice in range(1, args.num_slices):
        comparison_table = f"\n=== After Training on Slice {train_slice} (T{train_slice+1}) ===\n"
        comparison_table += "Test Slice\tIncremental NDCG\tPrompt NDCG\tNDCG Diff\tIncremental HR\tPrompt HR\tHR Diff\n"
        comparison_table += "-" * 100 + "\n"
        
        # Evaluate performance on all slices
        avg_inc_ndcg = 0.0
        avg_prompt_ndcg = 0.0
        avg_inc_hr = 0.0
        avg_prompt_hr = 0.0
        
        for test_slice in range(args.num_slices):
            # Get results for this training-testing combination
            inc_ndcg = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_{test_slice}']['ndcg']
            inc_hr = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_{test_slice}']['hr']
            
            prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_{test_slice}']['ndcg']
            prompt_hr = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_{test_slice}']['hr']
            
            # Calculate differences
            ndcg_diff = prompt_ndcg - inc_ndcg
            hr_diff = prompt_hr - inc_hr
            
            # Add to averages
            avg_inc_ndcg += inc_ndcg
            avg_prompt_ndcg += prompt_ndcg
            avg_inc_hr += inc_hr
            avg_prompt_hr += prompt_hr
            
            # Add row to table
            comparison_table += f"T{test_slice+1}\t{inc_ndcg:.4f}\t{prompt_ndcg:.4f}\t{ndcg_diff:+.4f}\t{inc_hr:.4f}\t{prompt_hr:.4f}\t{hr_diff:+.4f}\n"
        
        # Calculate averages
        avg_inc_ndcg /= args.num_slices
        avg_prompt_ndcg /= args.num_slices
        avg_inc_hr /= args.num_slices
        avg_prompt_hr /= args.num_slices
        
        # Add average row
        comparison_table += "-" * 100 + "\n"
        comparison_table += f"Avg\t{avg_inc_ndcg:.4f}\t{avg_prompt_ndcg:.4f}\t{avg_prompt_ndcg-avg_inc_ndcg:+.4f}\t{avg_inc_hr:.4f}\t{avg_prompt_hr:.4f}\t{avg_prompt_hr-avg_inc_hr:+.4f}\n"
        
        # Add specific analysis
        # Knowledge retention (T1 performance)
        t1_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_0']['ndcg']
        t1_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_0']['ndcg']
        t1_base_ndcg = results_incremental['base_model']['slice_0']['ndcg']
        
        comparison_table += f"\nKnowledge Retention (T1): Incremental={t1_inc_ndcg:.4f}, Prompt={t1_prompt_ndcg:.4f}, Diff={t1_prompt_ndcg-t1_inc_ndcg:+.4f}\n"
        
        # New knowledge acquisition (performance on current slice)
        current_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_{train_slice}']['ndcg']
        current_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_{train_slice}']['ndcg']
        
        comparison_table += f"New Knowledge (T{train_slice+1}): Incremental={current_inc_ndcg:.4f}, Prompt={current_prompt_ndcg:.4f}, Diff={current_prompt_ndcg-current_inc_ndcg:+.4f}\n"
        
        # Print and log
        print(comparison_table)
        log_file.write(comparison_table)
        
        # Save to individual file
        with open(os.path.join(output_dir, f'comparison_after_slice_{train_slice}.txt'), 'w') as f:
            f.write(comparison_table)
    
    # Create a summary of retention vs acquisition
    retention_table = "\n=== Summary: Knowledge Retention vs. Acquisition ===\n"
    retention_table += "Slice\tRetention (T1)\t\t\tAcquisition (Current)\n"
    retention_table += "\tIncr.\tPrompt\tDiff\tIncr.\tPrompt\tDiff\n"
    retention_table += "-" * 70 + "\n"
    
    for train_slice in range(1, args.num_slices):
        # Get retention (T1) performance
        t1_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_0']['ndcg']
        t1_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_0']['ndcg']
        retention_diff = t1_prompt_ndcg - t1_inc_ndcg
        
        # Get acquisition (current slice) performance
        current_inc_ndcg = results_incremental['incremental_model'][f'after_slice_{train_slice}_eval_on_{train_slice}']['ndcg']
        current_prompt_ndcg = results_prompt['prompt_model'][f'after_slice_{train_slice}_eval_on_{train_slice}']['ndcg']
        acquisition_diff = current_prompt_ndcg - current_inc_ndcg
        
        # Add row
        retention_table += f"T{train_slice+1}\t{t1_inc_ndcg:.4f}\t{t1_prompt_ndcg:.4f}\t{retention_diff:+.4f}\t{current_inc_ndcg:.4f}\t{current_prompt_ndcg:.4f}\t{acquisition_diff:+.4f}\n"
    
    # Print and log
    print(retention_table)
    log_file.write(retention_table)
    
    # Save to file
    with open(os.path.join(output_dir, 'retention_vs_acquisition_summary.txt'), 'w') as f:
        f.write(retention_table)


def main():
    args = parser.parse_args()
    
    # Default to running both models for comparison
    args.run_both = True
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    if args.incremental and args.run_both:
        # Run both models and compare
        run_comparison(args)
    else:
        # Original training code
        if not os.path.isdir(args.dataset + '_' + args.train_dir):
            os.makedirs(args.dataset + '_' + args.train_dir)
        with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
            params = '\n'.join([str(k) + ',' + str(v) 
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
            print(params)
            f.write(params)

        # Set device
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Load dataset
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        num_batch = len(user_train) // args.batch_size

        # Print dataset statistics
        cc = 0.0
        max_len = 0
        for u in user_train:
            cc += len(user_train[u])
            max_len = max(max_len, len(user_train[u]))
        print(f"\nThere are {usernum} users {itemnum} items")
        print(f"Average sequence length: {cc / len(user_train)}")
        print(f"Maximum length of sequence: {max_len}\n")

        # Create log file
        f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

        # Initialize model, sampler and optimizer
        model = SASRec(usernum, itemnum, args).to(device)
        sampler = WarpSampler(user_train, usernum, itemnum, 
                            batch_size=args.batch_size, maxlen=args.maxlen,
                            threshold_user=args.threshold_user, 
                            threshold_item=args.threshold_item,
                            n_workers=3, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        # Initial evaluation
        t_test = evaluate(model, dataset, args, device)
        t_valid = evaluate_valid(model, dataset, args, device)
        print(f"[0, 0.0, {t_valid[0]}, {t_valid[1]}, {t_test[0]}, {t_test[1]}]")

        t0 = time.time()

        try:
            for epoch in range(1, args.num_epochs + 1):
                for step in range(num_batch):
                    u, seq, pos, neg = sampler.next_batch()
                    
                    optimizer.zero_grad()
                    loss, attention_weights, auc, l2_loss = model(u, seq, pos, neg, is_training=True)
                    loss.backward()
                    optimizer.step()

                if epoch % args.print_freq == 0:
                    t1 = time.time() - t0
                    T = t1
                    
                    t_test = evaluate(model, dataset, args, device)
                    t_valid = evaluate_valid(model, dataset, args, device)
                    
                    print(f"[{epoch}, {T}, {loss}, {l2_loss}, {auc}, {t_valid[0]}, {t_valid[1]}, {t_test[0]}, {t_test[1]}]")
                    f.write(f"{t_valid} {t_test}\n")
                    f.flush()
                    
                    t0 = time.time()

        except KeyboardInterrupt:
            print('Early stopping triggered')
        finally:
            f.close()
            sampler.close()
            print("Done")


if __name__ == "__main__":
    main()