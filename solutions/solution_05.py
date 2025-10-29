import math
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config
from solutions.solution_02 import causal_mask

class GPT2Attention(Module):
    """Multi-head self-attention matching HuggingFace GPT-2 structure.

    Args:
        config: GPT-2 configuration.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute attention scores and apply to values.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            Attention output.
        """
        attn_weights = query @ key.transpose(-1, -2)

        # Scale attention weights
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        # Apply causal mask
        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def _split_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Split the last dimension into (num_heads, head_size).

        Args:
            tensor: Input tensor.
            num_heads: Number of attention heads.
            attn_head_size: Size of each attention head.

        Returns:
            Reshaped tensor with shape (batch, head, seq_length, head_features).
        """
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Merge attention heads back.

        Args:
            tensor: Input tensor.
            num_heads: Number of attention heads.
            attn_head_size: Size of each attention head.

        Returns:
            Merged tensor.
        """
        tensor = tensor.transpose(-3, -2)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply multi-head self-attention.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Attention output.
        """
        query, key, value = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.c_proj(attn_output)

        return attn_output
    