import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['n_heads']
        self.head_dim = config['d_model'] // config['n_heads']
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config['d_model'], self.all_head_size)
        self.key   = nn.Linear(config['d_model'], self.all_head_size)
        self.value = nn.Linear(config['d_model'], self.all_head_size)

    def transpose_for_scores(self, x):
        # [batch, seq_len, all_head_size] -> [batch, num_heads, seq_len, head_dim]
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query = self.transpose_for_scores(self.query(hidden_states))
        key   = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        # [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_scores += attention_mask  # attention_mask must be broadcastable

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, value)  # [B, H, S, D]

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:2] + (self.all_head_size,)
        context = context.view(*new_context_shape)

        return context


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['d_model'], config['dim_ff'])
        self.activation = F.gelu  # or F.relu

    def forward(self, x):
        return self.activation(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['dim_ff'], config['d_model'])
        self.LayerNorm = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = nn.Linear(config['d_model'], config['d_model'])
        self.LayerNorm = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self_attn(hidden_states, attention_mask)
        hidden_states = self.dropout(self_outputs)
        hidden_states = self.LayerNorm(hidden_states + hidden_states)  # residual
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.attn_output_dense = nn.Linear(config['d_model'], config['d_model'])
        self.attn_output_layer_norm = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

        self.intermediate = BertIntermediate(config)
        self.output_dense = nn.Linear(config['dim_ff'], config['d_model'])
        self.output_layer_norm = nn.LayerNorm(config['d_model'])

    def forward(self, hidden_states, attention_mask=None):
        # ---- Self-Attention ----
        self_output = self.attention(hidden_states, attention_mask)
        self_output = self.dropout(self.attn_output_dense(self_output))
        hidden_states = self.attn_output_layer_norm(hidden_states + self_output)

        # ---- Feedforward ----
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.output_layer_norm(hidden_states + layer_output)

        return hidden_states
