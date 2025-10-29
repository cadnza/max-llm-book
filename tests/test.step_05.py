"""Tests for GPT2Attention implementation in steps/step_05.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect


def test_step_05():
    """Test that GPT2Attention class is correctly implemented."""
    print("Running tests for Step 05: Implement Multi-Head Attention...\n")
    print("Results:")

    # Test 1: Check if functional module is imported
    try:
        from steps import step_05 as attention_module

        # Check if F is imported from max.experimental.functional
        source = inspect.getsource(attention_module)
        if 'from max.experimental import functional as F' in source:
            print("✅  functional module is correctly imported as F from max.experimental")
        else:
            print("❌ functional module is not imported from max.experimental")
            print("   Hint: Add 'from max.experimental import functional as F'")
    except Exception as e:
        print(f"❌ Error importing step_05 module: {e}")
        return

    # Test 2: Check if Tensor is imported
    if 'from max.experimental.tensor import Tensor' in source:
        print("✅  Tensor is correctly imported from max.experimental.tensor")
    else:
        print("❌ Tensor is not imported from max.experimental.tensor")
        print("   Hint: Add 'from max.experimental.tensor import Tensor'")

    # Test 3: Check if Linear and Module are imported
    if 'from max.nn.module_v3 import Linear, Module' in source or 'from max.nn.module_v3 import' in source:
        print("✅  Linear and Module are imported from max.nn.module_v3")
    else:
        print("❌ Linear and Module are not imported from max.nn.module_v3")
        print("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    # Test 4: Check if GPT2Attention class exists
    if hasattr(attention_module, 'GPT2Attention'):
        print("✅  GPT2Attention class exists")
    else:
        print("❌ GPT2Attention class not found in step_05 module")
        return

    # Test 5: Check if c_attn Linear layer is created correctly
    if 'self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)' in source:
        print("✅  self.c_attn Linear layer is created correctly")
    else:
        print("❌ self.c_attn Linear layer is not created correctly")
        print("   Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)")

    # Test 6: Check if c_proj Linear layer is created correctly
    if 'self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)' in source:
        print("✅  self.c_proj Linear layer is created correctly")
    else:
        print("❌ self.c_proj Linear layer is not created correctly")
        print("   Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)")

    # Test 7: Check _split_heads implementation
    if 'tensor.shape[:-1] + [num_heads, attn_head_size]' in source:
        print("✅  _split_heads: new_shape calculation is correct")
    else:
        print("❌ _split_heads: new_shape calculation is not correct")
        print("   Hint: new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]")

    if 'tensor.reshape(new_shape)' in source:
        print("✅  _split_heads: tensor.reshape is used")
    else:
        print("❌ _split_heads: tensor.reshape is not used")
        print("   Hint: Use tensor.reshape(new_shape)")

    if 'tensor.transpose(-3, -2)' in source:
        print("✅  _split_heads: tensor.transpose(-3, -2) is used")
    else:
        print("❌ _split_heads: tensor.transpose(-3, -2) is not used")
        print("   Hint: Use tensor.transpose(-3, -2) to move heads before sequence length")

    # Test 8: Check _merge_heads implementation
    if 'tensor.shape[:-2] + [num_heads * attn_head_size]' in source:
        print("✅  _merge_heads: new_shape calculation is correct")
    else:
        print("❌ _merge_heads: new_shape calculation is not correct")
        print("   Hint: new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]")

    # Test 9: Check _attn implementation - attention score computation
    if 'query @ key.transpose(-1, -2)' in source:
        print("✅  _attn: attention scores computed with query @ key.transpose(-1, -2)")
    else:
        print("❌ _attn: attention scores not computed correctly")
        print("   Hint: Use query @ key.transpose(-1, -2)")

    # Test 10: Check _attn implementation - scaling
    if 'math.sqrt' in source and ('value.shape[-1]' in source or 'head_dim' in source):
        print("✅  _attn: attention weights are scaled")
    else:
        print("❌ _attn: attention weights are not scaled correctly")
        print("   Hint: Scale by math.sqrt(int(value.shape[-1]))")

    # Test 11: Check _attn implementation - causal masking
    if 'query.shape[-2]' in source:
        print("✅  _attn: sequence length extracted from query.shape[-2]")
    else:
        print("❌ _attn: sequence length not extracted correctly")
        print("   Hint: seq_len = query.shape[-2]")

    if 'causal_mask(' in source:
        print("✅  _attn: causal_mask function is called")
    else:
        print("❌ _attn: causal_mask function is not called")
        print("   Hint: Use causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)")

    # Test 12: Check _attn implementation - softmax
    if 'F.softmax(attn_weights)' in source or 'F.softmax(attn_weights' in source:
        print("✅  _attn: F.softmax is applied to attention weights")
    else:
        print("❌ _attn: F.softmax is not applied to attention weights")
        print("   Hint: Use F.softmax(attn_weights)")

    # Test 13: Check _attn implementation - weighted sum
    if 'attn_weights @ value' in source:
        print("✅  _attn: attention output computed with attn_weights @ value")
    else:
        print("❌ _attn: attention output not computed correctly")
        print("   Hint: Use attn_weights @ value")

    # Test 14: Check __call__ implementation - F.split
    if 'F.split(' in source and 'self.c_attn(hidden_states)' in source:
        print("✅  __call__: F.split is used on self.c_attn(hidden_states)")
    else:
        print("❌ __call__: F.split is not used correctly")
        print("   Hint: Use F.split(self.c_attn(hidden_states), ...)")

    if '[self.split_size, self.split_size, self.split_size]' in source:
        print("✅  __call__: F.split uses correct split sizes")
    else:
        print("❌ __call__: F.split does not use correct split sizes")
        print("   Hint: Split into [self.split_size, self.split_size, self.split_size]")

    # Test 15: Check __call__ implementation - split heads
    if source.count('self._split_heads(') >= 3:
        print("✅  __call__: self._split_heads is called for query, key, and value")
    else:
        print("❌ __call__: self._split_heads is not called for all query, key, and value")
        print("   Hint: Call self._split_heads for query, key, and value")

    # Test 16: Check __call__ implementation - attention and merge
    if 'self._attn(' in source:
        print("✅  __call__: self._attn is called")
    else:
        print("❌ __call__: self._attn is not called")
        print("   Hint: Call self._attn(query, key, value)")

    if 'self._merge_heads(' in source:
        print("✅  __call__: self._merge_heads is called")
    else:
        print("❌ __call__: self._merge_heads is not called")
        print("   Hint: Call self._merge_heads(attn_output, self.num_heads, self.head_dim)")

    if 'self.c_proj(' in source:
        print("✅  __call__: self.c_proj is called")
    else:
        print("❌ __call__: self.c_proj is not called")
        print("   Hint: Call self.c_proj(attn_output)")

    # Test 17: Check that None values are replaced
    lines = source.split('\n')
    none_assignments = [line for line in lines if (
        'self.c_attn = None' in line or
        'self.c_proj = None' in line or
        'new_shape = None' in line or
        'tensor = None' in line or
        'attn_weights = None' in line or
        'seq_len = None' in line or
        'mask = None' in line or
        'attn_output = None' in line or
        'query = None' in line or
        'key = None' in line or
        'value = None' in line or
        ('return None' in line and '__call__' in source[max(0, source.index(line)-500):source.index(line)])
    )]

    if none_assignments:
        print("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_assignments[:5]:  # Show first 5
            print(f"   {line.strip()}")
        if len(none_assignments) > 5:
            print(f"   ... and {len(none_assignments) - 5} more")
        print("   Hint: Replace all 'None' values with the actual implementation")
    else:
        print("✅  All placeholder 'None' values have been replaced")

    # Test 18: Try to instantiate the GPT2Attention class
    try:
        from solutions.solution_01 import GPT2Config

        config = GPT2Config()
        attention = attention_module.GPT2Attention(config=config)
        print("✅  GPT2Attention class can be instantiated")

        # Check if c_attn and c_proj are initialized
        if hasattr(attention, 'c_attn') and attention.c_attn is not None:
            print("✅  GPT2Attention.c_attn is initialized")
        else:
            print("❌ GPT2Attention.c_attn is not initialized")

        if hasattr(attention, 'c_proj') and attention.c_proj is not None:
            print("✅  GPT2Attention.c_proj is initialized")
        else:
            print("❌ GPT2Attention.c_proj is not initialized")

        # Check attributes
        if hasattr(attention, 'embed_dim') and attention.embed_dim == 768:
            print("✅  GPT2Attention.embed_dim is correct: 768")
        else:
            print("❌ GPT2Attention.embed_dim is not correct")

        if hasattr(attention, 'num_heads') and attention.num_heads == 12:
            print("✅  GPT2Attention.num_heads is correct: 12")
        else:
            print("❌ GPT2Attention.num_heads is not correct")

        if hasattr(attention, 'head_dim') and attention.head_dim == 64:
            print("✅  GPT2Attention.head_dim is correct: 64")
        else:
            print("❌ GPT2Attention.head_dim is not correct")

    except Exception as e:
        print(f"❌ GPT2Attention class instantiation failed: {e}")
        print("   This usually means some TODO items are not completed")

    # Test 19: Try to run the forward pass (if imports are available)
    try:
        from max.experimental.tensor import Tensor

        # Create a dummy input tensor
        # Shape: (batch_size=1, sequence_length=4, embedding_dim=768)
        hidden_states = Tensor.ones([1, 4, 768])

        result = attention(hidden_states)
        print("✅  GPT2Attention forward pass executes without errors")

        # Check output shape
        if hasattr(result, 'shape'):
            expected_shape = (1, 4, 768)
            actual_shape = tuple(result.shape)
            if actual_shape == expected_shape:
                print(f"✅  Output shape is correct: {actual_shape}")
            else:
                print(f"❌ Output shape is incorrect: expected {expected_shape}, got {actual_shape}")

    except ImportError:
        print("�  Cannot test forward pass execution (MAX not available in this environment)")
    except AttributeError as e:
        print(f"❌ Forward pass execution failed: {e}")
        print("   This usually means some TODO items are not completed")
    except Exception as e:
        print(f"❌ Forward pass execution failed with error: {e}")

    # Final summary
    print("\n" + "="*60)
    if all([
        'from max.experimental import functional as F' in source,
        'from max.experimental.tensor import Tensor' in source,
        'from max.nn.module_v3 import' in source,
        'self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)' in source,
        'self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)' in source,
        'tensor.shape[:-1] + [num_heads, attn_head_size]' in source,
        'tensor.shape[:-2] + [num_heads * attn_head_size]' in source,
        'query @ key.transpose(-1, -2)' in source,
        'math.sqrt' in source,
        'causal_mask(' in source,
        'F.softmax' in source,
        'attn_weights @ value' in source,
        'F.split(' in source,
        'self._split_heads(' in source,
        'self._merge_heads(' in source,
        'self.c_proj(' in source,
        not none_assignments
    ]):
        print("🎉 All checks passed! Your implementation matches the solution.")
        print("="*60)
    else:
        print("⚠️  Some checks failed. Review the hints above and try again.")
        print("="*60)


if __name__ == "__main__":
    test_step_05()
