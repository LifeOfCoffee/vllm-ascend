# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.worker.utils import make_page_strided_cache_view


def test_v2_views_relocate_from_external_bytes_preserving_every_other_byte():
    source = torch.full((1024,), -91, dtype=torch.int8, device="cpu")
    destination = source.clone()
    src_views = [
        ("attention", 0, (4, 3, 1, 2), torch.bfloat16, 16, 32),
        ("attention", 1, (4, 3, 1, 2), torch.bfloat16, 160, 32),
        ("indexer", 0, (4, 1, 2, 2), torch.bfloat16, 320, 24),
    ]
    dst_views = [
        ("attention", 0, (4, 3, 1, 2), torch.bfloat16, 48, 48),
        ("attention", 1, (4, 3, 1, 2), torch.bfloat16, 272, 48),
        ("indexer", 0, (4, 1, 2, 2), torch.bfloat16, 544, 32),
    ]
    src = [
        make_page_strided_cache_view(source, shape, dtype, stride, offset)
        for _, _, shape, dtype, offset, stride in src_views
    ]
    dst = [
        make_page_strided_cache_view(destination, shape, dtype, stride, offset)
        for _, _, shape, dtype, offset, stride in dst_views
    ]
    for index, tensor in enumerate(src):
        tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape) + index * 100)

    def database(views):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.num_blocks = 4
        db = ChunkedTokenDatabase([], [16, 16], None)
        groups = {0: views[:2], 1: views[2:]}
        geometry = {
            group: [worker._get_cache_block_metadata(tensor) for tensor in tensors]
            for group, tensors in groups.items()
        }
        db.set_group_buffers(
            {group: [tensor.data_ptr() for tensor in tensors] for group, tensors in groups.items()},
            {group: [item[0] for item in items] for group, items in geometry.items()},
            {group: [item[1] for item in items] for group, items in geometry.items()},
        )
        return db

    writer, reader = database(src), database(dst)
    objects = {}
    expected = [[tensor[block].clone() for tensor in src] for block in (2, 0)]
    for slot, block in enumerate((2, 0)):
        for group in (0, 1):
            addresses, sizes, selected = writer.prepare_value(slot * 16, (slot + 1) * 16, [2, 0], group)
            assert selected == block
            objects[group, slot] = [ctypes.string_at(address, size) for address, size in zip(addresses, sizes)]

    source.fill_(-25)  # Only the independent byte objects retain the saved payload.
    touched = set()
    for slot, block in enumerate((1, 3)):
        for group in (0, 1):
            addresses, sizes, selected = reader.prepare_value(slot * 16, (slot + 1) * 16, [1, 3], group)
            assert selected == block
            payloads = objects[group, slot]
            assert len(payloads) == len(addresses)
            for address, size, payload in zip(addresses, sizes, payloads):
                assert len(payload) == size
                ctypes.memmove(address, payload, size)
                touched.update(range(address - destination.data_ptr(), address - destination.data_ptr() + size))
    for block, tensors in zip((1, 3), expected):
        for tensor, saved in zip(dst, tensors):
            assert torch.equal(tensor[block], saved)
    actual = ctypes.string_at(destination.data_ptr(), destination.numel())
    assert all(byte == 0xA5 for offset, byte in enumerate(actual) if offset not in touched)
