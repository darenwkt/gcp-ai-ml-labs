import math
import torch
import torch.nn as nn

try:
    import torch.distributed as dist
except ImportError:
    dist = None

# --- GPT-2 Model Configuration (must match training/predict config) ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# --- Shared Base Layers ---
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# --- 1. Standard (Non-Parallel) Layers ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1), persistent=False)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = GELU()
        self.fc2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# --- 2. 3D Distributed Parallel Layers ---

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if (dist is not None and tp_group is not None) else 1
        
        assert out_features % self.tp_size == 0, f"out_features ({out_features}) must be divisible by TP size ({self.tp_size})"
        self.sharded_out_features = out_features // self.tp_size
        
        self.weight = nn.Parameter(torch.empty(self.sharded_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.sharded_out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if (dist is not None and tp_group is not None) else 1
        
        assert in_features % self.tp_size == 0, f"in_features ({in_features}) must be divisible by TP size ({self.tp_size})"
        self.sharded_in_features = in_features // self.tp_size
        
        self.weight = nn.Parameter(torch.empty(out_features, self.sharded_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = nn.functional.linear(x, self.weight, None)
        if self.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out

class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, tp_group=None):
        super().__init__()
        self.num_heads = num_heads
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if (dist is not None and tp_group is not None) else 1
        
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        assert num_heads % self.tp_size == 0, f"num_heads ({num_heads}) must be divisible by TP size ({self.tp_size})"
        
        self.local_num_heads = num_heads // self.tp_size
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        
        self.W_query = ColumnParallelLinear(d_in, d_out, bias=qkv_bias, tp_group=tp_group)
        self.W_key = ColumnParallelLinear(d_in, d_out, bias=qkv_bias, tp_group=tp_group)
        self.W_value = ColumnParallelLinear(d_in, d_out, bias=qkv_bias, tp_group=tp_group)
        self.out_proj = RowParallelLinear(d_out, d_out, bias=True, tp_group=tp_group)
        self.dropout_p = dropout

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.local_num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.local_num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.local_num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        context_vec = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        context_vec = context_vec.transpose(1, 2).reshape(b, num_tokens, self.local_num_heads * self.head_dim)
        context_vec = self.out_proj(context_vec)
        return context_vec

class ParallelFeedForward(nn.Module):
    def __init__(self, cfg, tp_group=None):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.fc1 = ColumnParallelLinear(emb_dim, 4 * emb_dim, bias=True, tp_group=tp_group)
        self.gelu = GELU()
        self.fc2 = RowParallelLinear(4 * emb_dim, emb_dim, bias=True, tp_group=tp_group)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class ParallelTransformerBlock(nn.Module):
    def __init__(self, cfg, tp_group=None):
        super().__init__()
        self.att = ParallelMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            tp_group=tp_group
        )
        self.ff = ParallelFeedForward(cfg, tp_group=tp_group)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# --- 3. Unified GPT-2 Model ---

class GPTModel(nn.Module):
    def __init__(self, cfg, parallel=False, checkpoint_activations=False, pp_rank=0, pp_size=1, tp_group=None):
        super().__init__()
        self.parallel = parallel
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.checkpoint_activations = checkpoint_activations
        
        if parallel:
            block_fn = lambda: ParallelTransformerBlock(cfg, tp_group=tp_group)
        else:
            block_fn = lambda: TransformerBlock(cfg)
            
        total_layers = cfg["n_layers"]
        if parallel:
            assert total_layers % pp_size == 0, f"n_layers ({total_layers}) must be divisible by pp_size ({pp_size})"
            self.layers_per_stage = total_layers // pp_size
            self.start_layer = pp_rank * self.layers_per_stage
            self.end_layer = self.start_layer + self.layers_per_stage
        else:
            self.layers_per_stage = total_layers
            self.start_layer = 0
            self.end_layer = total_layers
            
        if pp_rank == 0:
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])
            
        self.trf_blocks = nn.ModuleList(
            [block_fn() for _ in range(self.start_layer, self.end_layer)]
        )
        
        if pp_rank == pp_size - 1:
            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=True)

    def forward(self, in_idx_or_x):
        if self.pp_rank == 0:
            in_idx = in_idx_or_x
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
        else:
            x = in_idx_or_x

        for block in self.trf_blocks:
            if self.parallel and self.checkpoint_activations and self.training:
                # Use activation checkpointing
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        if self.pp_rank == self.pp_size - 1:
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
        else:
            return x
