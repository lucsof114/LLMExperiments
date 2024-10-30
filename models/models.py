import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np


class PresidentialAttention(nn.Module):
    def __init__(self, config):
        super(PresidentialAttention, self).__init__()
        self.n_heads = config['n_heads']
        self.head_size = config['hidden_size'] // self.n_heads

        ## todo: mqa could be used here but my model is small so it's not worth it
        self.proj_q = nn.Linear(config['hidden_size'], self.head_size * self.n_heads)
        self.proj_k = nn.Linear(config['hidden_size'], self.head_size * self.n_heads)
        self.proj_v = nn.Linear(config['hidden_size'], self.head_size * self.n_heads)
        self.proj_o = nn.Linear(self.head_size * self.n_heads, config['hidden_size'])
        self.attn_dropout = nn.Dropout(config['attn_dropout'])

    def forward(self, query, key, value, attention_mask=None):
        """
        query: [batch_size, seq_len, hidden_size]
        key: [batch_size, seq_len, hidden_size]
        value: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, query_len, seq_len] (optional)
        attention_mask is 0 if the position is not masked, 1 otherwise.
        can be used for self-attention or cross-attention. 
        In self-attention, query == key == value, in cross-attention, query != key == value.
        """
        proj_q = self.proj_q(query) # [batch_size, seq_len, hidden_size] 
        proj_k = self.proj_k(key) # [batch_size, seq_len, hidden_size] 
        proj_v = self.proj_v(value) # [batch_size, seq_len, hidden_size] 

        proj_q = proj_q.reshape(query.size(0), query.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]
        proj_k = proj_k.reshape(key.size(0), key.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]
        proj_v = proj_v.reshape(value.size(0), value.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]

        scaled_qk_matmul = torch.matmul(proj_q, proj_k.transpose(-2, -1)) / np.sqrt(self.head_size) # [batch_size, n_heads, seq_len, seq_len]

        if attention_mask is not None:
            scaled_qk_matmul += attention_mask * -np.inf 

        attn_weights = nn.functional.softmax(scaled_qk_matmul, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = self.attn_dropout(attn_weights)     
        attn_output = torch.matmul(attn_weights, proj_v) # [batch_size, n_heads, seq_len, head_size]
        attn_output = attn_output.transpose(1, 2).reshape(attn_output.size(0), attn_output.size(1), -1) # [batch_size, seq_len, n_heads * head_size]
        attn_output = self.proj_o(attn_output) # [batch_size, seq_len, hidden_size]

        return attn_output

class PresidentialMLP(nn.Module):
    def __init__(self, config):
        super(PresidentialMLP, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(config['hidden_size'], config['mlp_size']) for _ in range(config['n_mlp_layers'])])
        self.act = nn.GELU()

    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = self.act(linear_layer(x))
        return x

class PresidentialTransformer(nn.Module):
    def __init__(self, config):
        super(PresidentialTransformer, self).__init__()
        self.attn = PresidentialAttention(config)
        self.ln1 = nn.LayerNorm(config['hidden_size'])
        self.mlp = PresidentialMLP(config)
        self.ln2 = nn.LayerNorm(config['hidden_size'])

    def forward(self, x, attention_mask=None):
        x = x + self.attn(x, x, x, attention_mask)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


# todo: implement a rotary positional encoding class at some point https://arxiv.org/pdf/2104.09864
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, config):
        super(LearnablePositionalEncoding, self).__init__()
        self.seq_len = config['seq_len']
        self.embed_dim = config['embed_dim']
        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, self.embed_dim))

    def forward(self, x):
        return x + self.pos_embed
    

class PresidentialModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(PresidentialModel, self).__init__()
        self.embedder = AutoModel.from_pretrained(config['embedder'])
        self.transformers = nn.Sequential([PresidentialTransformer(config['transformer']) \
                                            for _ in range(config['n_transformers'])])
        self.positional_encoder = LearnablePositionalEncoding(config['n_heads'] * config['n_kv_heads'])
        self.lm_head = nn.Linear(config['transformer']['n_heads'] * config['transformer']['n_kv_heads'], vocab_size)

    def forward(self, x):
        x = self.embedder(x)
        x = self.transformers(x)
        return x
