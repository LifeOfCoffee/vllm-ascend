# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real cross-process Mooncake transfers for shared, strided cache regions.

Run with two visible NPUs and the deployment's Ascend transport configuration.
The master and both clients use the same Python environment. No model is loaded.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

PAYLOAD_BYTES = (32, 48, 64, 16, 80, 96)
SOURCE_BLOCKS = (4, 1, 3)
TARGET_BLOCKS = (0, 2, 4)
ALIGNMENT = 2 * 1024 * 1024


def _layout(padding):
    cursor = 64
    regions = []
    for payload in PAYLOAD_BYTES:
        stride = payload + padding
        regions.append((cursor, stride, payload))
        cursor += stride * 5 + 64
    return regions


def _descriptors(pointer, regions, blocks, tokens):
    addresses = [[pointer + offset + block * stride for offset, stride, _ in regions] for block in blocks]
    sizes = [[payload * tokens // 16 for _, _, payload in regions] for _ in blocks]
    return addresses, sizes


def _worker(args):
    # Each subprocess imports the native runtime after selecting its role.
    import torch
    import torch_npu  # noqa: F401
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    directory = Path(args.directory)
    torch.npu.set_device(args.device)
    store = MooncakeDistributedStore()
    host = os.environ.get("VLLM_HOST_IP") or socket.gethostbyname(socket.gethostname())
    result = store.setup(
        local_hostname=host,
        metadata_server=f"http://127.0.0.1:{args.metadata_port}/metadata",
        global_segment_size=128 * 1024 * 1024 if args.role == "owner" else 0,
        local_buffer_size=0,
        protocol="ascend",
        rdma_devices="",
        master_server_addr=f"127.0.0.1:{args.master_port}",
    )
    assert result == 0, result
    allocation = torch.empty(2 * ALIGNMENT, dtype=torch.uint8, device="npu")
    raw = allocation.narrow(0, (-allocation.data_ptr()) % ALIGNMENT, ALIGNMENT)
    assert store.register_buffer(raw.data_ptr(), raw.numel()) == 0

    def put(keys, addresses, sizes):
        if not args.layerwise:
            return store.batch_put_from_multi_buffers(keys, addresses, sizes, ReplicateConfig())
        assert store.batch_put_session_start(keys, [sum(row) for row in sizes], ReplicateConfig()) == [0] * len(keys)
        offsets = [0] * len(keys)
        for region in range(len(sizes[0])):
            lengths = [[row[region]] for row in sizes]
            copied = store.batch_put_from_multi_buffer_ranges(
                keys,
                [[row[region]] for row in addresses],
                lengths,
                [[offset] for offset in offsets],
            )
            assert len(copied) == len(keys) and all(value >= 0 for value in copied), copied
            offsets = [offset + row[0] for offset, row in zip(offsets, lengths, strict=True)]
        return store.batch_put_session_end(keys)

    def get(keys, addresses, sizes):
        if not args.layerwise:
            return store.batch_get_into_multi_buffers(keys, addresses, sizes)
        started = store.batch_get_session_start(keys)
        if any(value < 0 for value in started):
            return started
        offsets = [0] * len(keys)
        for region in range(len(sizes[0])):
            lengths = [[row[region]] for row in sizes]
            copied = store.batch_get_into_multi_buffer_ranges(
                keys,
                [[row[region]] for row in addresses],
                lengths,
                [[offset] for offset in offsets],
            )
            assert len(copied) == len(keys) and all(value >= 0 for value in copied), copied
            offsets = [offset + row[0] for offset, row in zip(offsets, lengths, strict=True)]
        assert store.batch_get_session_end(keys) == 0
        return [0] * len(keys)

    if args.role == "owner":
        cases = []
        namespace = uuid.uuid4().hex
        for padding in (0, 64):
            regions = _layout(padding)
            raw.fill_(90)
            for region, (offset, stride, payload) in enumerate(regions):
                for block in range(5):
                    values = torch.arange(payload, dtype=torch.int32, device="npu") + region * 17 + block * 29
                    raw[offset + block * stride : offset + block * stride + payload].copy_(values.to(torch.uint8))
            torch.npu.synchronize()
            for tokens in (8, 16):
                keys = [f"qsa-buffer-{namespace}-{padding}-{tokens}-{index}" for index in range(3)]
                addresses, sizes = _descriptors(raw.data_ptr(), regions, SOURCE_BLOCKS, tokens)
                assert put(keys, addresses, sizes) == [0] * len(keys)
                assert store.batch_is_exist(keys) == [1] * len(keys)
                cases.append(dict(padding=padding, tokens=tokens, keys=keys))
            # External bytes must remain valid after all source payloads change.
            for offset, stride, payload in regions:
                for block in range(5):
                    raw[offset + block * stride : offset + block * stride + payload].fill_(165)
            torch.npu.synchronize()
        (directory / "ready.json").write_text(json.dumps(cases))
        deadline = time.monotonic() + 180
        while not (directory / "stop").exists():
            assert time.monotonic() < deadline, "Peer did not finish"
            time.sleep(0.1)
    else:
        cases = json.loads((directory / "ready.json").read_text())
        results = []
        for case in cases:
            regions, tokens = _layout(case["padding"]), case["tokens"]
            raw.fill_(90)
            expected = bytearray([90]) * raw.numel()
            addresses, sizes = _descriptors(raw.data_ptr(), regions, TARGET_BLOCKS, tokens)
            for source, target in zip(SOURCE_BLOCKS, TARGET_BLOCKS, strict=True):
                for region, (offset, stride, payload) in enumerate(regions):
                    size = payload * tokens // 16
                    start = offset + target * stride
                    expected[start : start + size] = bytes(
                        (index + region * 17 + source * 29) % 256 for index in range(size)
                    )
                    raw[start : start + size].fill_(165)
            torch.npu.synchronize()
            read = get(case["keys"], addresses, sizes)
            torch.npu.synchronize()
            assert all(value >= 0 for value in read), read
            assert raw.cpu().numpy().tobytes() == expected, case
            peer_keys = [key + "-peer-write" for key in case["keys"]]
            assert put(peer_keys, addresses, sizes) == [0] * len(peer_keys)
            for pointers, lengths in zip(addresses, sizes, strict=True):
                for pointer, length in zip(pointers, lengths, strict=True):
                    raw[pointer - raw.data_ptr() : pointer - raw.data_ptr() + length].fill_(165)
            torch.npu.synchronize()
            assert all(value >= 0 for value in get(peer_keys, addresses, sizes))
            torch.npu.synchronize()
            assert raw.cpu().numpy().tobytes() == expected
            results.append(
                dict(
                    padding=case["padding"],
                    tokens=tokens,
                    keys=len(peer_keys),
                    exact=True,
                )
            )
        before = raw.cpu().numpy().tobytes()
        missing = get(["missing-" + uuid.uuid4().hex], [[raw.data_ptr() + 64]], [[32]])
        torch.npu.synchronize()
        assert missing[0] < 0, missing
        assert raw.cpu().numpy().tobytes() == before
        (directory / "result.json").write_text(json.dumps(results, indent=2))
    store.close()


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@pytest.mark.parametrize("layerwise", [False, True])
def test_mooncake_cross_process_six_region_restore(tmp_path, layerwise):
    pytest.importorskip("mooncake")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < 2:
        pytest.skip("Requires two visible NPUs")
    master_port, metadata_port, metrics_port = _port(), _port(), _port()
    environment = dict(os.environ)
    environment.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
    master = owner = peer = None
    try:
        with (tmp_path / "master.log").open("wb") as log:
            master = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from mooncake.cli import main; main()",
                    f"--port={master_port}",
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host=127.0.0.1",
                    f"--http_metadata_server_port={metadata_port}",
                    f"--metrics_port={metrics_port}",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 60
        while True:
            assert master.poll() is None, (tmp_path / "master.log").read_text()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{metrics_port}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            assert time.monotonic() < deadline, "Master was not ready"
            time.sleep(0.2)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--directory",
            str(tmp_path),
            "--master-port",
            str(master_port),
            "--metadata-port",
            str(metadata_port),
        ]
        if layerwise:
            command.append("--layerwise")
        with (tmp_path / "owner.log").open("wb") as log:
            owner = subprocess.Popen(
                command + ["--role", "owner", "--device", "0"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 90
        while not (tmp_path / "ready.json").exists():
            assert owner.poll() is None, (tmp_path / "owner.log").read_text()
            assert time.monotonic() < deadline, "Owner was not ready"
            time.sleep(0.2)
        with (tmp_path / "peer.log").open("wb") as log:
            peer = subprocess.Popen(
                command + ["--role", "peer", "--device", "1"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        assert peer.wait(timeout=90) == 0, (tmp_path / "peer.log").read_text()
        (tmp_path / "stop").touch()
        assert owner.wait(timeout=30) == 0, (tmp_path / "owner.log").read_text()
        results = json.loads((tmp_path / "result.json").read_text())
        assert len(results) == 4 and all(row["exact"] for row in results)
    finally:
        (tmp_path / "stop").touch()
        _stop(peer)
        _stop(owner)
        _stop(master)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("owner", "peer"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--metadata-port", type=int, required=True)
    parser.add_argument("--layerwise", action="store_true")
    _worker(parser.parse_args())
