import random
import numpy as np
from multiprocessing import Process, Queue
import torch

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen,  
                   threshold_user, threshold_item,
                   result_queue, SEED):
    def sample():
        # Get list of valid users (those that actually exist in user_train)
        valid_users = list(user_train.keys())
        if not valid_users:
            raise ValueError("No valid users in user_train dictionary")
            
        # Sample from valid users only
        user_idx = np.random.randint(0, len(valid_users))
        user = valid_users[user_idx]
        
        while len(user_train[user]) <= 1:
            user_idx = np.random.randint(0, len(valid_users))
            user = valid_users[user_idx]

        seq = np.zeros([maxlen], dtype=np.int64)
        pos = np.zeros([maxlen], dtype=np.int64)
        neg = np.zeros([maxlen], dtype=np.int64)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        
        for i in reversed(user_train[user][:-1]):
            # SSE for item side
            if random.random() > threshold_item:
                i = np.random.randint(1, itemnum + 1)
                nxt = np.random.randint(1, itemnum + 1)
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: 
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        # SSE for user side
        if random.random() > threshold_user:
            user_idx = np.random.randint(0, len(valid_users))
            user = valid_users[user_idx]
     
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(list(zip(*one_batch)))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, 
                 threshold_user=1.0, threshold_item=1.0, n_workers=1, device='cuda'):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        self.device = device
        
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                    usernum,
                                                    itemnum,
                                                    batch_size,
                                                    maxlen,
                                                    threshold_user,
                                                    threshold_item,
                                                    self.result_queue,
                                                    np.random.randint(2e9)
                                                    )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        batch = self.result_queue.get()
        user, seq, pos, neg = batch
        
        # Convert lists of arrays to single numpy arrays
        user = np.zeros_like(np.array(user)) 
        seq = np.array(seq)
        pos = np.array(pos)
        neg = np.array(neg)
        
        return (torch.LongTensor(user).to(self.device), 
                torch.LongTensor(seq).to(self.device),
                torch.LongTensor(pos).to(self.device), 
                torch.LongTensor(neg).to(self.device))

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()