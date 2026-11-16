# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend requires a SIMT-capable CANN 9.2
toolchain on Ascend 950.

### Working Principle

HiSparse keeps recent KV rows in the NPU Resident cache and uses the NPU Hot
cache for rows refilled from Host memory. The resolver checks the selected rows
against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

### Usage Scenarios

Use HiSparse with Model Runner V2 for sparse RoPE MLA/SFA models serving long
contexts. BF16 KV and native Ascend 950 C8 (E4M3FN values with FP32 scales) are
supported, with independent Main and Indexer KV dtype selection.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | Ascend 950 with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | C4, TurboQuant and INT8-format C8 are outside the current support scope. |
| Parallelism and transport | Context parallelism is unsupported. For P/D transport, use `HiSparsePDConnector` with MemFabric on P and D and matching TP sizes. |
| Host storage | Use private Host pools per tensor-parallel rank and Core eager Host mirroring. |
| P/D Host import | MemFabric owns D's Host Source allocation; D reads main KV directly into it. Indexer is read into D NPU. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native A5 C8, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

### P/D with MemFabric Pull

Run a Prefill instance without HiSparse and a Decode instance with HiSparse.
Use the Pull P/D proxy: Prefill publishes its completed KV and Decode reads it.

This experimental integration uses vendor-provided `memfabric_hybrid`
`1.5.0.dev20261115` with a matching A5 Offload extension. Obtain the SDK through
its vendor distribution and build its Offload kernel for the target SoC before
starting the workers. The paired vLLM Core is commit
[`41c4343e3cc0502f9076477aa0dbafe4965c6da5`](https://github.com/LifeOfCoffee/vllm/commit/41c4343e3cc0502f9076477aa0dbafe4965c6da5),
based on `24420b7a341af37a9bc3186e6fc6062512e0aa58`. It supplies the shared-pool
policy and native-event hooks used below. This is a fixed experimental pairing,
not a claim of compatibility with arbitrary vLLM or SDK releases.

Prefill `--kv-transfer-config`:

```json
{"kv_connector":"HiSparsePDConnector","kv_role":"kv_producer","kv_connector_extra_config":{"memfabric_transfer_protocol":"device_urma"}}
```

Decode `--kv-transfer-config`:

```json
{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"HiSparseConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"host_pool_gib":4}},{"kv_connector":"HiSparsePDConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"memfabric_transfer_protocol":"device_urma"}}]}}
```

Set distinct `kv_port` values for P and D. Use the Pull proxy described in
[disaggregated prefill](../../developer_guide/Design_Documents/disaggregated_prefill.md).
P finishes Prefill and retains its registered NPU KV until D finishes reading.
D allocates Host Source through MemFabric Offload and calls
`TransferEngine.batch_transfer_sync_read()` with local Host destinations for
main KV and local NPU destinations for Indexer. Both P source regions and D Host/Indexer destination regions are registered
with TransferEngine for `device_urma`. This differs from the SFA SDMA path,
where the D-side local destinations are not explicitly registered. MLA NoPE/RoPE components are read into
token-interleaved Host rows. There is no separate Host staging pool.

Both P and D use MemFabric `device_urma`. With single-node multiprocessing TP>1
and PP=PCP=DCP=1, D uses Offload `Scene.SHARED`: TP rank 0 contributes the Host
pool, rounded up to a GiB, and other ranks contribute zero bytes and attach to
the same pool. Only rank 0 imports main KV and writes back Decode KV; every rank
still imports its own Indexer and retains local Resident/Hot caches. Core waits
for every TP worker before exposing an imported prefix. Backend-native NPU IPC
events order shared Host writes before other ranks read them. This requires the
matching Core `can_share_host_pool`, `record_event`, and `wait_event` hooks.

For TP1 or topologies outside the shared-pool gate, the private allocator keeps
Offload LOCAL `GIANT_PAGE` pools rounded to a GiB per rank; provision HugePages
for those ranks, including alignment padding. SHARED uses the SDK's unified
address mapping, not the LOCAL GIANT_PAGE path. Register the complete physical
allocation once, rather than a Source subview, to avoid conflicting SDK ranges.
Build the SDK's A5 Offload kernel before starting the workers.

The shared allocation is retained until worker-process exit because current
TransferEngine registrations have process lifetime. Recreating engines with a
different shared-pool configuration in one worker is not supported. The TP-shared
URMA path and cross-process NPU events require hardware validation; success of
the private GIANT_PAGE path does not establish support for SHARED mappings.
BM groups and `L2GH` are not used. The connector requires matching
TP sizes and PCP=DCP=1. After all requested groups finish, Core HiSparse imports
the prefix and restores the last page to Resident. This first implementation uses
bulk Pull; layerwise overlap remains a separate adaptation.

### Verifying the Feature

Send a request and check the server's metrics endpoint for the HiSparse miss
counter. The counter can stay at zero when selected rows are already on the NPU.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:hisparse_cache_misses_total'
```

## Configuration Parameters

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Logical Host pool capacity in GiB. Private mode allocates it per TP rank; supported shared mode contributes one physical pool on rank 0. Allow for page padding and SDK allocation rounding. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |
