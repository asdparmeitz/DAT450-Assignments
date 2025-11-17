
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
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Linear layers for query, key, value, and output projections (no bias)
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Normalizers after query and key representations
        self.q_norm = A2RMSNorm(config)
        self.k_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        b, m, d = hidden_states.shape  # batch, sequence length, hidden size
        n_h = self.num_attention_heads
        d_h = self.head_dim
        
        # Compute query, key, and value representations
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)
        
        # Apply normalizers to query and key
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Reshape for multi-head attention: (b, m, n_h, d_h) then transpose to (b, n_h, m, d_h)
        q = q.view(b, m, n_h, d_h).transpose(1, 2)
        k = k.view(b, m, n_h, d_h).transpose(1, 2)
        v = v.view(b, m, n_h, d_h).transpose(1, 2)
        
        # Apply RoPE rotations to query and key
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        
        # Compute scaled dot-product attention with causal masking
        attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Combine attention heads: transpose to (b, m, n_h, d_h) then reshape to (b, m, d)
        attn_out = attn_out.transpose(1, 2).reshape(b, m, d)
        
        # Apply output projection
        output = self.W_o(attn_out)
        
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # Normalizer before attention (pre-norm)
        self.pre_attention_norm = A2RMSNorm(config)
        # Multi-head attention layer
        self.attention = A2Attention(config)
        # Normalizer before MLP (pre-norm)
        self.pre_mlp_norm = A2RMSNorm(config)
        # MLP layer
        self.mlp = A2MLP(config)

    def forward(self, hidden_states, rope_rotations):
        # Pre-norm attention with residual connection
        attn_input = self.pre_attention_norm(hidden_states)
        attn_output = self.attention(attn_input, rope_rotations)
        hidden_states = attn_output + hidden_states  # residual connection
        
        # Pre-norm MLP with residual connection
        mlp_input = self.pre_mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_input)
        hidden_states = mlp_output + hidden_states  # residual connection
        
        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        
        # Token embedding layer (vocab_size -> hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer decoder layers in a ModuleList
        self.layers = nn.ModuleList([
            A2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalizer
        self.norm = A2RMSNorm(config)
        
        # Unembedding layer (hidden_size -> vocab_size, no bias)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        self.post_init()
        # Tie weights: share embedding and output projection (common in language models)
        self.lm_head.weight = self.embed_tokens.weight
        # Tell transformers about tied weights for proper saving
        #self._tied_weights_keys = [("embed_tokens.weight", "lm_head.weight")]


    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # Apply token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through all transformer decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
        
        # Apply final normalizer
        hidden_states = self.norm(hidden_states)
        
        # Apply unembedding to get logits
        logits = self.lm_head(hidden_states)
        
        return logits


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


#### Simple test code ####
"""
if __name__ == "__main__":
    # Create a simple configuration for testing
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        rope_theta=10000.0,
        hidden_act='silu',
        max_position_embeddings=512,
        rms_norm_eps=1e-6
    )
    
"""