# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.platforms.interface import Platform


class _Backend:
    @classmethod
    def customize_spec(cls, spec):
        # Triton packs each INT4 K/V element into half a byte and appends one
        # FP32 scale per head to each of K and V. Keep the backend contract
        # explicit so this test isolates platform-side layer selection.
        if spec.dtype == torch.uint8:
            return SimpleNamespace(
                page_size_bytes=(
                    spec.num_kv_heads * spec.head_size
                    + spec.num_kv_heads * 2 * torch.float32.itemsize
                )
            )
        return spec

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return [16]


def test_gemma4_mixed_kv_alignment_uses_per_layer_geometry():
    layer_types = [
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
        for index in range(30)
    ]
    architectures = [
        SimpleNamespace(
            head_size=512 if layer_type == "full_attention" else 256,
            num_kv_heads=2 if layer_type == "full_attention" else 8,
        )
        for layer_type in layer_types
    ]
    model_config = SimpleNamespace(
        model_arch_config=architectures,
        hf_text_config=SimpleNamespace(layer_types=layer_types),
        dtype=torch.bfloat16,
        get_total_num_hidden_layers=lambda: len(layer_types),
        get_num_kv_heads=lambda _parallel, arch=None: (
            arch.num_kv_heads if arch is not None else 8
        ),
        get_head_size=lambda: 512,
    )
    cache_config = SimpleNamespace(
        cache_dtype="int4_per_token_head",
        kv_cache_dtype_skip_layers=["sliding_window"],
        block_size=16,
        mamba_page_size_padded=None,
        skip_page_size_padded=None,
    )
    config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
    )

    Platform._align_heterogeneous_kv_block_size(config, _Backend)

    # Global INT4 layers are 2 heads x 512 dimensions plus inline scales:
    # 1,040 bytes/token. Sliding BF16 layers are 8 x 256 x K/V x 2 bytes:
    # 8,192 bytes/token. A 128-token primary block is the smallest multiple
    # of 16 large enough to cover one 16-token BF16 page.
    assert cache_config.block_size == 128
    assert cache_config.skip_page_size_padded == 128 * 1040
