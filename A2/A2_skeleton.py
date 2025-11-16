
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model"""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture.
    The input and output are the same size. (hidden_size)"""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        self.gate_w = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)   
        self.up_w = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.combining_w = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = nn.functional.silu(self.gate_w(hidden_states))
        up = self.up_w(hidden_states)
        return self.combining_w(gate * up)


# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        return self.weight * hidden_states / rms

class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.norm_q = A2RMSNorm(self.hidden_size)
        self.norm_k = A2RMSNorm(self.hidden_size)


    def forward(self, hidden_states, rope_rotations):
        
        b, m, d = hidden_states.shape
        n_h = self.num_attention_heads
        d_h = self.head_dim

        # Compute 
        query = self.W_q(hidden_states)
        key = self.W_k(hidden_states)
        value = self.W_v(hidden_states)
        
        #normalize the query and key
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape the vectors into right dimensions for the attention
        query = query.view(b, m, n_h, d_h).transpose(1, 2)
        key = key.view(b, m, n_h, d_h).transpose(1, 2)
        value = value.view(b, m, n_h, d_h).transpose(1, 2)


        # Apply RoPE to the query and key
        query, key = apply_rotary_pos_emb(query, key, rope_rotations)

        # SDPA 
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal = True)
        
        #computation 3
        attention_output = attention_output.transpose(1, 2).reshape(b, m, d)

        return  self.W_o(attention_output)




class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.attention_layer = A2Attention(config)
        self.rms_norm_1 = A2RMSNorm(config.hidden_size)
        self.mlp_layer = A2MLP(config)
        self.rms_norm_2 = A2RMSNorm(config.hidden_size)


    def forward(self, hidden_states, rope_rotations):

        # Attention + first normalization
        attention_output = self.attention_layer(hidden_states, rope_rotations)
        normalized_attention_output = self.rms_norm_1(attention_output)

        #residual connection 1
        hidden_states = hidden_states + normalized_attention_output

        # MLP + second normalization
        mlp_output = self.mlp_layer(hidden_states)
        normalized_mlp_output = self.rms_norm_2(mlp_output)

        #residual connection 2
        hidden_states = hidden_states + normalized_mlp_output

        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        ...


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
