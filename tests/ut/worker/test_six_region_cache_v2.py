# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.six_region_kv_cache_layout import build_six_region_kv_cache_layout
from vllm_ascend.patch.platform.patch_kv_cache_utils import _get_qwen4_exp_six_region_kv_cache_config
from vllm_ascend.worker.v2 import attn_utils


def _groups():
    sources = ["model.layers.11.self_attn", "model.layers.47.self_attn"]
    return [
        KVCacheGroupSpec(
            [f"{source}.attn" for source in sources],
            FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=64, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.raw_key_cache" for source in sources],
            CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=44, head_size_v=0, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in sources],
            MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, tokens_per_state=4),
        ),
        KVCacheGroupSpec(
            ["model.layers.0.linear_attn"],
            MambaSpec(block_size=256, shapes=((64, 4), (128, 128)), dtypes=(torch.bfloat16, torch.bfloat16)),
        ),
        KVCacheGroupSpec(
            ["model.layers.1.linear_attn"],
            MambaSpec(block_size=256, shapes=((64, 4), (128, 128)), dtypes=(torch.bfloat16, torch.bfloat16)),
        ),
        KVCacheGroupSpec(
            ["model.ple"],
            MambaSpec(block_size=256, shapes=((128, 5),), dtypes=(torch.bfloat16,), tp_replicated=True),
        ),
    ]


@pytest.mark.parametrize("connector", [None, "AscendStoreConnector"])
@pytest.mark.parametrize("blocks", [1, 3])
def test_native_six_region_allocation_and_views(monkeypatch, connector, blocks):
    vllm_config = NS(
        kv_transfer_config=NS(kv_connector=connector, kv_connector_extra_config={}) if connector else None,
        cache_config=NS(num_gpu_blocks_override=blocks, prefix_cache_retention_interval=None),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda config: False)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda config: NS(size=1))
    monkeypatch.setattr(attn_utils, "is_c8_mxfp_kv_quant", lambda config: False)
    config = _get_qwen4_exp_six_region_kv_cache_config(vllm_config, _groups(), 10**9)
    layout = build_six_region_kv_cache_layout(config.kv_cache_groups, blocks)
    raw = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    shared = layout.owners
    backing = raw[shared[0].layer_name]
    assert all(raw[owner.layer_name] is backing for owner in shared)
    assert backing.numel() == layout.slot_count * layout.slot_backing_size
    if connector:
        assert backing.data_ptr() % (2 * 1024 * 1024) == 0
    backing.fill_(-91)
    backend = NS(full_cls_name=lambda: "test.SixRegionBackend")
    layers = {owner.layer_name: NS(get_attn_backend=lambda: backend, num_heads=1) for owner in layout.owners}
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args: layers)
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", lambda *args, **kwargs: raw)
    vllm_config.cache_config.cache_dtype = "auto"
    caches = attn_utils.allocate_kv_cache_main(
        config,
        torch.device("cpu"),
        None,
        [group.kv_cache_spec.block_size for group in config.kv_cache_groups],
    )
    regions_by_role = {
        "qsa_main": ("r2", "r3"),
        "qsa_raw": ("r4",),
        "qsa_compressed": ("r5",),
        "gdn": ("r1", "r2"),
        "ple": ("r6",),
    }
    for owner in shared:
        views = caches[owner.layer_name]
        views = views if isinstance(views, (list, tuple)) else [views]
        for view, region_name in zip(views, regions_by_role[owner.role], strict=True):
            region = layout.region(region_name)
            assert view.shape[0] == blocks
            assert view.stride(0) * view.element_size() == region.page_size_bytes
            assert view.data_ptr() - backing.data_ptr() == owner.slot * layout.slot_backing_size + region.offset
            assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            end = view.data_ptr() - backing.data_ptr() + (blocks - 1) * region.page_size_bytes
            end += view[0].numel() * view.element_size()
            assert end <= owner.slot * layout.slot_backing_size + region.end
    key, value = caches["model.layers.11.self_attn.attn"]
    assert key.shape == value.shape == (blocks, 256, 1, 64)
    assert key.data_ptr() == caches["model.layers.0.linear_attn"][1].data_ptr()
    assert caches["model.layers.11.self_attn.indexer.raw_key_cache"].shape == (blocks, 1, 8, 44)
    assert caches["model.layers.11.self_attn.indexer.compressed_key_cache"].shape == (blocks, 1, 64, 32)
