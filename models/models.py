import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
import numpy as np
from models.tokenizer import TokenMapper


class PresidentialAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, attn_dropout=0.1):
        super(PresidentialAttention, self).__init__()
        self.n_heads = n_heads
        self.head_size = hidden_size // self.n_heads

        ## todo: mqa could be used here but my model is small so it's not worth it
        self.proj_q = nn.Linear(hidden_size, self.head_size * self.n_heads)
        self.proj_k = nn.Linear(hidden_size, self.head_size * self.n_heads)
        self.proj_v = nn.Linear(hidden_size, self.head_size * self.n_heads)
        self.proj_o = nn.Linear(self.head_size * self.n_heads, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)

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

        proj_q = proj_q.view(query.size(0), query.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]
        proj_k = proj_k.view(key.size(0), key.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]
        proj_v = proj_v.view(value.size(0), value.size(1), self.n_heads, self.head_size).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]

        scaled_qk_matmul = torch.matmul(proj_q, proj_k.transpose(-2, -1)) / np.sqrt(self.head_size) # [batch_size, n_heads, seq_len, seq_len]

        if attention_mask is not None:
            scaled_qk_matmul += attention_mask * -1e9 

        attn_weights = nn.functional.softmax(scaled_qk_matmul, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = self.attn_dropout(attn_weights)     
        attn_output = torch.matmul(attn_weights, proj_v).transpose(1, 2) # [batch_size, n_heads, seq_len, head_size]
        attn_output = attn_output.contiguous().view(attn_output.size(0), attn_output.size(1), -1) # [batch_size, seq_len, n_heads * head_size]
        attn_output = self.proj_o(attn_output) # [batch_size, seq_len, hidden_size]
        return attn_output

class PresidentialMLP(nn.Module):
    def __init__(self, hidden_size, n_mlp_layers):
        super(PresidentialMLP, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_mlp_layers)])
        self.act = nn.GELU()

    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = self.act(linear_layer(x))
        return x

class PresidentialTransformer(nn.Module):
    def __init__(self, hidden_size, n_heads, attn_dropout=0.1, n_mlp_layers=2):
        super(PresidentialTransformer, self).__init__()
        self.attn = PresidentialAttention(hidden_size, n_heads, attn_dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.mlp = PresidentialMLP(hidden_size, n_mlp_layers)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(x, x, x, attention_mask)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


# todo: implement a rotary positional encoding class at some point https://arxiv.org/pdf/2104.09864
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_size):
        super(LearnablePositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, self.hidden_size))
        self.mapping = nn.Linear(self.embed_dim, self.hidden_size)

    def forward(self, x):
        return self.mapping(x) + self.pos_embed

class PresidentialModel(nn.Module):
    def __init__(self, config, tokenizer):
        super(PresidentialModel, self).__init__()
        self.embedder = tokenizer.embeddings
        self.transformers = nn.ModuleList([PresidentialTransformer(config['hidden_size'], 
                                                                   config['n_heads'], 
                                                                   config['attn_dropout'],
                                                                   config['n_mlp_layers']) \
                                            for _ in range(config['n_transformers'])])
        self.positional_encoder = LearnablePositionalEncoding(config['seq_len'], 
                                                            tokenizer.embed_dim, 
                                                            config['hidden_size'])
        self.lm_head = nn.Linear(config['hidden_size'], tokenizer.vocab_size)

    def forward(self, x, attention_mask):
        x = self.embedder.model(x)
        x = self.positional_encoder(x)
        for transformer in self.transformers:
            x = transformer(x, attention_mask)
        x = self.lm_head(x)
        return x