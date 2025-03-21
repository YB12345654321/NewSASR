import sys
import copy
import random
import numpy as np
import torch
from collections import defaultdict

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    # assume user/item index starting from 1
    f = open('SSEPI/data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, args, device='cuda'):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    # print(test)
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # Get the list of existing users, instead of assuming sequential IDs
    all_users = list(train.keys())
    if len(all_users) > 10000:
        users = random.sample(all_users, 10000)
    else:
        users = all_users

    with torch.no_grad():
        user_num = 0
        for u in users:
            user_num+=1
            if u not in train or u not in test or len(train[u]) < 1 or len(test[u]) < 1: 
                print(user_num, 'no training data or test data')
                if u not in train:
                    print('not in train')
                if u not in test:
                    print('not in test')
                # if len(train[u])<1:
                #     print('not train data')
                # if len(test[u])<1:
                #     print('no test data')
                continue

            seq = np.zeros([args.maxlen], dtype=np.int64)
            idx = args.maxlen - 1
            
            if u in valid and len(valid[u]) > 0:
                seq[idx] = valid[u][0]
                idx -= 1
                
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: 
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            seq = torch.LongTensor(seq).unsqueeze(0).to(device)
            u_tensor = torch.LongTensor([u]).to(device)
            item_idx = torch.LongTensor(item_idx).to(device)
            
            predictions = -model.predict(u_tensor, seq, item_idx)
            predictions = predictions[0]
            rank = (-predictions).argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.k:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 1000 == 0:
                sys.stdout.flush()

    model.train()
    
    # Avoid division by zero
    if valid_user > 0:
        return NDCG / valid_user, HT / valid_user
    else:
        return -0.01, -0.01

def evaluate_valid(model, dataset, args, device='cuda'):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    # Get the list of existing users, instead of assuming sequential IDs
    all_users = list(train.keys())
    if len(all_users) > 10000:
        users = random.sample(all_users, 10000)
    else:
        users = all_users
        
    with torch.no_grad():
        for u in users:
            if u not in train or u not in valid or len(train[u]) < 1 or len(valid[u]) < 1: 
                continue

            seq = np.zeros([args.maxlen], dtype=np.int64)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: 
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            seq = torch.LongTensor(seq).unsqueeze(0).to(device)
            u_tensor = torch.LongTensor([u]).to(device)
            item_idx = torch.LongTensor(item_idx).to(device)
            
            predictions = -model.predict(u_tensor, seq, item_idx)
            predictions = predictions[0]

            rank = (-predictions).argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.k:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                sys.stdout.flush()

    model.train()
    
    # Avoid division by zero
    if valid_user > 0:
        return NDCG / valid_user, HT / valid_user
    else:
        return 0.0, 0.0