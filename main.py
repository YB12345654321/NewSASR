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
from util import *

# Import incremental learning components
from incremental_model import IncrementalSASRec
from incremental_data import TimeSlicedData, ExperienceReplay
from eval_incremental import evaluate_incremental, evaluate_forgetting

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
parser.add_argument('--num_epochs', default=1000, type=int)
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
parser.add_argument('--slice_ratios', nargs='+', type=float, default=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], 
                    help='Ratios for user-based time slices')
parser.add_argument('--min_interactions', default=10, type=int, 
                    help='Minimum interactions per user for valid data')
parser.add_argument('--distill_temp', default=2.0, type=float, help='Temperature for knowledge distillation')
parser.add_argument('--distill_alpha', default=0.5, type=float, help='Weight for distillation loss')
parser.add_argument('--replay_ratio', default=0.3, type=float, help='Ratio of experience replay')
parser.add_argument('--buffer_size', default=1000, type=int, help='Size of experience replay buffer')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
parser.add_argument('--load_splits', default=None, type=str, help='Path to load saved data splits')
parser.add_argument('--save_splits', action='store_true', help='Whether to save data splits for future use')

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

def run_incremental_learning(args):
    """
    Run incremental learning experiment with only knowledge distillation (no EWC, no Joint model)
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = args.dataset + '_' + args.train_dir + '_incremental'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Log file
    log_file = open(os.path.join(output_dir, 'incremental_log.txt'), 'w')
    log_file.write(f"Random seed: {args.seed}\n")
    
    # Load time-sliced data
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
        # Train for fewer epochs
        inc_epochs = args.num_epochs // 2  # Fewer epochs for incremental learning
        t0 = time.time()

        for epoch in range(1, inc_epochs + 1):
            # Determine number of batches
            num_batch = max(len(slice_user_train) // args.batch_size, 1)
            
            for step in range(num_batch):
                # Get batch from current slice
                u, seq, pos, neg = slice_sampler.next_batch()
                
                # At this point u, seq, pos, neg should all be tensors from the WarpSampler
                
                # Optionally mix with replay samples
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
            
            if epoch % (args.print_freq // 2) == 0:
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
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Close log file
    log_file.close()
    
    print("\n=== Incremental Learning Experiment Completed ===")
    print(f"Results saved to {output_dir}")
    
    return results

args = parser.parse_args()

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Run incremental learning when specified
    if args.incremental:
        run_incremental_learning(args)
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
        T = 0.0
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
                    T += t1
                    
                    t_test = evaluate(model, dataset, args, device)
                    t_valid = evaluate_valid(model, dataset, args, device)
                    
                    print(f"[{epoch}, {T}, {loss}, {l2_loss}, {auc}, {t_valid[0]}, {t_valid[1]}, {t_test[0]}, {t_test[1]}]")
                    f.write(f"{t_valid} {t_test}\n")
                    f.flush()
                    
                    # Optional: save attention maps and embeddings
                    if False:  # Change to True if you want to save these
                        attention_map = attention_weights[0].detach().cpu().numpy()
                        with open(f"attention_map_{step}.pickle", 'wb') as fw:
                            pickle.dump(attention_map, fw)
                            
                        batch_data = [u.cpu().numpy(), seq.cpu().numpy()]
                        with open(f"batch_{step}.pickle", 'wb') as fw:
                            pickle.dump(batch_data, fw)
                            
                        user_emb = model.user_emb.embedding.weight.detach().cpu().numpy()
                        with open("user_emb.pickle", 'wb') as fw:
                            pickle.dump(user_emb, fw)
                            
                        item_emb = model.item_emb.embedding.weight.detach().cpu().numpy()
                        with open("item_emb.pickle", 'wb') as fw:
                            pickle.dump(item_emb, fw)
                            
                    t0 = time.time()

        except KeyboardInterrupt:
            print('Early stopping triggered')
        finally:
            f.close()
            sampler.close()
            print("Done")