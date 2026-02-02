# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Step 03: Causal Masking

Implement causal attention masking that prevents tokens from attending to future positions.

Tasks:
1. Import functional module (as F) and Tensor from max.nn
2. Add @F.functional decorator to the causal_mask function
3. Create a constant tensor filled with negative infinity
4. Broadcast the mask to the correct shape (sequence_length, n)
5. Apply band_part to create the lower triangular causal structure

Run: pixi run s03
"""

# 1: Import the required modules from MAX
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim, DimLike
from max.tensor import Tensor


# 2: Add the @F.functional decorator to make this a MAX functional operation
@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
) -> Tensor:
    """Create a causal mask for autoregressive attention.

    Args:
        sequence_length: Length of the sequence.
        num_tokens: Number of tokens.
        dtype: Data type for the mask.
        device: Device to create the mask on.

    Returns:
        A causal mask tensor.

    """
    # Calculate total sequence length
    n = Dim(sequence_length) + num_tokens

    # 3: Create a constant tensor filled with negative infinity
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)

    # 4: Broadcast the mask to the correct shape
    mask = F.broadcast_to(mask, shape=(sequence_length, n))

    # 5: Apply band_part to create the causal (lower triangular) structure and return the mask
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
