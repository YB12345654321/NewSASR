import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def positional_encoding(dim, sentence_length, dtype=torch.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zero_pad=True, scale=True):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zero_pad = zero_pad
        self.scale = scale
        
        self.embedding = nn.Embedding(vocab_size, num_units, padding_idx=0 if zero_pad else None)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, inputs):
        # print(type(inputs))
        if inputs.dtype != torch.long:
            inputs = inputs.long()
        x = self.embedding(inputs)
        if self.scale:
            x = x * (self.num_units ** 0.5)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False):
        super(MultiHeadAttention, self).__init__()
        
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        
        self.Q_proj = nn.Linear(num_units, num_units)
        self.K_proj = nn.Linear(num_units, num_units)
        self.V_proj = nn.Linear(num_units, num_units)
        
        self.output_proj = nn.Linear(num_units, num_units)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(num_units)

    def forward(self, queries, keys, is_training=True):
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)     # (N, T_k, C)
        V = self.V_proj(keys)     # (N, T_k, C)
        
        # Split and concat
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        
        # Key Masking
        key_masks = torch.sign(torch.abs(keys.sum(dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)  # (h*N, T_q, T_k)
        
        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(key_masks == 0, paddings, outputs)  # (h*N, T_q, T_k)

        # Causality
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = tril.unsqueeze(0).repeat(outputs.size(0), 1, 1)  # (h*N, T_q, T_k)
            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(masks == 0, paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        attention = outputs

        # Query Masking
        query_masks = torch.sign(torch.abs(queries.sum(dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # (h*N, T_q, T_k)

        # Dropouts
        outputs = self.dropout(outputs) if is_training else outputs
        
        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # (h*N, T_q, C/h)
        
        # Restore shape
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual connection
        outputs += queries
        
        # Normalize
        outputs = self.layer_norm(outputs)
        
        return outputs, attention

class FeedForward(nn.Module):
    def __init__(self, in_units, num_units=[2048, 512], dropout_rate=0.2):
        super(FeedForward, self).__init__()
        
        self.conv1 = nn.Conv1d(in_units, num_units[0], 1)
        self.conv2 = nn.Conv1d(num_units[0], num_units[1], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(num_units[1])
        
    def forward(self, inputs, is_training=True):
        # Inner layer
        outputs = self.conv1(inputs.transpose(1, 2))
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs) if is_training else outputs
        
        # Readout layer
        outputs = self.conv2(outputs)
        outputs = self.dropout(outputs) if is_training else outputs
        
        # Restore shape
        outputs = outputs.transpose(1, 2)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = self.layer_norm(outputs)
        
        return outputs