import math
# 1: Import the required modules from MAX
# TODO: Import functional module from max.experimental with the alias F
# https://docs.modular.com/max/api/python/experimental/functional

# TODO: Import Tensor from max.experimental.tensor
# https://docs.modular.com/max/api/python/experimental/tensor.Tensor

# TODO: Import Linear and Module from max.nn.module_v3
# https://docs.modular.com/max/api/python/nn/module_v3

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

        # 2: Create the attention projection layers
        # TODO: Create self.c_attn as a Linear layer from embed_dim to 3 * embed_dim with bias=True
        # https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear
        # Hint: This single layer projects input to Query, Key, and Value simultaneously (3x the embedding dimension)
        self.c_attn = None

        # TODO: Create self.c_proj as a Linear layer from embed_dim to embed_dim with bias=True
        # https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear
        # Hint: This projects the concatenated attention output back to the embedding dimension
        self.c_proj = None

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
        # 3: Reshape tensor to split the embedding dimension into multiple heads
        # TODO: Create new_shape by taking tensor.shape[:-1] and appending [num_heads, attn_head_size]
        # Hint: This transforms (batch, seq_len, embed_dim) to (batch, seq_len, num_heads, head_dim)
        new_shape = None

        # TODO: Reshape the tensor to new_shape
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.reshape
        # Hint: This splits the last dimension into separate attention heads
        tensor = None

        # TODO: Transpose dimensions -3 and -2 and return the result
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.transpose
        # Hint: This moves heads before sequence length: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        return None

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
        # 4: Merge the attention heads back into a single embedding dimension
        # TODO: Transpose dimensions -3 and -2
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.transpose
        # Hint: This reverses the split_heads transpose: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        tensor = None

        # TODO: Create new_shape by taking tensor.shape[:-2] and appending [num_heads * attn_head_size]
        # Hint: This combines the head dimensions back into a single embedding dimension
        new_shape = None

        # TODO: Reshape tensor to new_shape and return the result
        # https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.reshape
        # Hint: This merges heads: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, embed_dim)
        return None

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute attention scores and apply to values.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            Attention output.
        """
        # 5: Compute attention weights
        # TODO: Compute attention scores by multiplying query with transposed key
        # Hint: Use @ operator with key.transpose(-1, -2) to compute Q @ K^T
        # This computes similarity scores between all query-key pairs
        attn_weights = None

        # TODO: Scale attention weights by dividing by sqrt(head_dim)
        # Hint: Use math.sqrt(int(value.shape[-1])) to get the square root of the head dimension
        # Scaling prevents the dot products from growing too large
        attn_weights = None

        # 6: Apply causal masking
        # TODO: Get the sequence length from query.shape[-2]
        # Hint: This is the sequence dimension in the query tensor
        seq_len = None

        # TODO: Create a causal mask using the causal_mask function
        # Hint: Pass seq_len, 0 (for num_tokens), dtype=query.dtype, device=query.device
        # The mask will prevent attention to future positions
        mask = None

        # TODO: Add the mask to attn_weights
        # Hint: The mask contains -inf for future positions, which will become 0 after softmax
        attn_weights = None

        # 7: Apply softmax and compute attention output
        # TODO: Apply softmax to attn_weights using F.softmax()
        # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax
        # Hint: This converts attention scores into probabilities that sum to 1
        attn_weights = None

        # TODO: Multiply attn_weights by value and return the result
        # Hint: Use @ operator to compute the weighted sum of values
        # This aggregates value vectors based on attention probabilities
        attn_output = None

        return attn_output

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply multi-head self-attention.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Attention output.
        """
        # 8: Project input to Query, Key, Value and split them
        # TODO: Apply self.c_attn to hidden_states, then use F.split() to split into query, key, value
        # https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split
        # Hint: Split into three equal parts of size self.split_size along axis=2
        # The split_sizes list should be [self.split_size, self.split_size, self.split_size]
        query, key, value = None

        # 9: Split heads for query, key, and value
        # TODO: Apply self._split_heads to query with self.num_heads and self.head_dim
        # Hint: This reshapes query to have separate attention heads
        query = None

        # TODO: Apply self._split_heads to key with self.num_heads and self.head_dim
        # Hint: This reshapes key to have separate attention heads
        key = None

        # TODO: Apply self._split_heads to value with self.num_heads and self.head_dim
        # Hint: This reshapes value to have separate attention heads
        value = None

        # 10: Compute attention and merge heads
        # TODO: Compute attention by calling self._attn with query, key, and value
        # Hint: This computes the scaled dot-product attention for all heads in parallel
        attn_output = None

        # TODO: Merge attention heads back using self._merge_heads
        # Hint: Pass attn_output, self.num_heads, and self.head_dim
        # This concatenates the outputs from all attention heads
        attn_output = None

        # TODO: Project the merged attention output using self.c_proj and return the result
        # Hint: This final linear layer mixes information across the concatenated heads
        return None
