import random
import numpy as np
from collections import defaultdict
import torch
import copy
from sampler import WarpSampler

class TimeSlicedData:
    """
    Manager for chronologically sliced user interaction data
    """
    def __init__(self, dataset_path, num_slices=6, slice_ratios=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], min_interactions=5):
        self.dataset_path = dataset_path
        self.num_slices = num_slices
        self.slice_ratios = slice_ratios
        self.min_interactions = min_interactions
        self.time_slices = [[] for _ in range(self.num_slices)]
        self.user_interactions = defaultdict(list)
        self.valid_users = set()
        self.all_users = set()
        self.all_items = set()
        self.usernum = 0
        self.itemnum = 0
        


    def load_data(self):
        """
        Load data and split into time slices based on user interaction length
        """
        # First, collect all user interactions
        with open(self.dataset_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                user = int(parts[0])
                item = int(parts[1])
                
                # If timestamp is available, use it; otherwise use line order
                timestamp = float(parts[2]) if len(parts) > 2 else len(self.user_interactions[user])
                
                self.user_interactions[user].append((item, timestamp))
                
                self.all_users.add(user)
                self.all_items.add(item)
                self.usernum = max(self.usernum, user)
                self.itemnum = max(self.itemnum, item)
        
        # Sort each user's interactions by timestamp
        for user in self.user_interactions:
            self.user_interactions[user].sort(key=lambda x: x[1])
        
        # Minimum required interactions for a user to be considered
        min_interactions = self.min_interactions
        print(f"Using slice ratios: {self.slice_ratios}")
        print(f"Minimum interactions required: {min_interactions}")
        
        # Filter out users with insufficient interactions
        for user, interactions in self.user_interactions.items():
            if len(interactions) >= min_interactions:
                self.valid_users.add(user)
        
        print(f"Total users: {len(self.user_interactions)}, Valid users: {len(self.valid_users)}")
        
        # Split each valid user's interactions into slices
        for user in self.valid_users:
            interactions = self.user_interactions[user]
            total_interactions = len(interactions)
            
            # Calculate slice boundaries
            boundaries = [0]
            current_pos = 0
            
            for i in range(self.num_slices - 1):
                current_pos += int(total_interactions * self.slice_ratios[i])
                boundaries.append(current_pos)
            boundaries.append(total_interactions)
            
            # Assign interactions to slices
            for i in range(self.num_slices):
                start_idx = boundaries[i]
                end_idx = boundaries[i+1]
                
                # Add to corresponding slice
                for idx in range(start_idx, end_idx):
                    item, timestamp = interactions[idx]
                    self.time_slices[i].append((user, item, timestamp))
        
        # Sort each slice by timestamp (for global time ordering)
        for i in range(self.num_slices):
            self.time_slices[i].sort(key=lambda x: x[2])
        
        # Print slice statistics
        for i in range(self.num_slices):
            slice_users = set([x[0] for x in self.time_slices[i]])
            print(f"Slice {i}: {len(self.time_slices[i])} interactions, {len(slice_users)} users")
            
        return self.time_slices


    def get_slice_data(self, slice_idx):
        """
        Get users and items from a specific slice
        """
        if not self.time_slices or len(self.time_slices[0]) == 0:
            self.load_data()
            
        if 0 <= slice_idx < len(self.time_slices):
            slice_users = set()
            slice_items = set()
            
            for user, item, _ in self.time_slices[slice_idx]:
                slice_users.add(user)
                slice_items.add(item)
                
            return slice_users, slice_items
        
        return set(), set()
    


    def prepare_slice(self, slice_idx, include_previous=False):
        """
        Prepare a specific time slice for training with 80/20 user split
        
        Args:
            slice_idx: Index of the time slice
            include_previous: Whether to include data from previous slices
        
        Returns:
            [user_train, user_valid, user_test, usernum, itemnum]
        """
        if not self.time_slices or len(self.time_slices[0]) == 0:
            self.load_data()
            
        # Get interactions for this slice
        interactions = []
        
        if include_previous:
            # Include all slices up to and including the target slice
            for i in range(slice_idx + 1):
                interactions.extend(self.time_slices[i])
        else:
            # Only include target slice
            interactions = self.time_slices[slice_idx]
            
        # Group by user
        user_items = defaultdict(list)
        for user, item, _ in interactions:
            user_items[user].append(item)
        
        # Split users into train (80%) and test (20%) sets
        all_users = list(user_items.keys())
        num_train_users = int(len(all_users) * 0.8)
        
        # Shuffle users and split
        random.shuffle(all_users)
        train_users = all_users[:num_train_users]
        test_users = all_users[num_train_users:]
        
        # Initialize data structures
        user_train = {}
        user_valid = {}
        user_test = {}
        
        # Process train users - use last item for validation
        for user in train_users:
            items = user_items[user]
            if len(items) < 2:
                # If too few items, just use for training
                user_train[user] = items
                user_valid[user] = []
                user_test[user] = []
            else:
                # Use all but last item for training, last item for validation
                user_train[user] = items[:-1]
                user_valid[user] = [items[-1]]
                user_test[user] = []
        
        # Process test users - use last item for testing
        for user in test_users:
            items = user_items[user]
            if len(items) < 2:
                # If too few items, skip this user
                continue
            else:
                # Use all but last item as known sequence, last item for testing
                user_train[user] = items[:-1]
                user_valid[user] = []
                user_test[user] = [items[-1]]
        
        return [user_train, user_valid, user_test, self.usernum, self.itemnum]


    def create_replay_buffer(self, slice_idx, buffer_size=1000, max_seq_length=1):
        """
        Create a replay buffer of representative samples from a slice with fixed sequence length
        
        Args:
            slice_idx: Index of the time slice
            buffer_size: Maximum size of buffer
            max_seq_length: Maximum sequence length to sample
                
        Returns:
            List of (user, sequence, positive, negative) tuples
        """
        if not self.time_slices or len(self.time_slices[0]) == 0:
            self.load_data()
            
        # Get slice data
        slice_data = self.prepare_slice(slice_idx)
        user_train = slice_data[0]
        
        # Create replay buffer
        buffer = []
        users = list(user_train.keys())
        
        # Shuffle users
        random.shuffle(users)
        
        # Select representative sequences
        for user in users:
            if len(user_train[user]) < max_seq_length + 1:
                continue
                
            # Create sequence for this user
            seq = user_train[user]
            
            # For sequences of exact length max_seq_length
            if len(seq) > max_seq_length:
                # Take the last item as positive sample
                pos = seq[-1]
                # Take the sequence before the last item
                seq_sample = seq[max(0, len(seq)-max_seq_length-1):-1]
                
                # Generate negative item
                neg = random.randint(1, self.itemnum)
                while neg in seq:
                    neg = random.randint(1, self.itemnum)
                    
                # Add to buffer
                buffer.append((user, seq_sample, pos, neg))
                    
            if len(buffer) >= buffer_size:
                return buffer
        
        return buffer

class ExperienceReplay:
    """
    Manages experience replay for incremental learning
    """
    def __init__(self, buffer_size=1000, replay_ratio=0.3):
        self.buffer = []
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        
    def update_buffer(self, buffer_data):
        """
        Update replay buffer with new data
        """
        # Add new data
        self.buffer.extend(buffer_data)
        
        # Trim if necessary
        if len(self.buffer) > self.buffer_size:
            self.buffer = random.sample(self.buffer, self.buffer_size)
                
    def sample_batch(self, batch_size):
        """
        Sample a batch from the replay buffer and ensure consistent sequence lengths
        """
        if not self.buffer:
            return None
            
        # Determine how many samples to draw
        replay_size = min(int(batch_size * self.replay_ratio), len(self.buffer))
        
        if replay_size == 0:
            return None
            
        # Sample from buffer
        samples = random.sample(self.buffer, replay_size)
        
        # Organize into batch format
        users, seqs, poss, negs = zip(*samples)
        
        # Find the maximum sequence length to pad to
        max_len = max(len(seq) for seq in seqs)
        
        # Pad sequences to uniform length
        padded_seqs = []
        for seq in seqs:
            if len(seq) < max_len:
                padded_seq = [0] * (max_len - len(seq)) + list(seq)  # Left padding
                padded_seqs.append(padded_seq)
            else:
                padded_seqs.append(seq)
        
        return users, padded_seqs, poss, negs