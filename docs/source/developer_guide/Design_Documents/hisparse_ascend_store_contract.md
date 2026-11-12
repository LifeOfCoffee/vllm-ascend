# HiSparse and AscendStore pooling contract

This document defines the handoff for the first HiSparse + AscendStore
integration on Ascend 950. It is an implementation contract, not a claim that
external pooling works in [the HiSparse foundation PR](https://github.com/vllm-project/vllm-ascend/pull/20926).
The first implementation target is non-layerwise Memcache pooling without
speculative decoding. Mooncake Store, layerwise I/O, and MTP build on the same
contract later. P/D peer-to-peer transfer is a separate integration.

## Cache groups and identity

HiSparse has four logical groups. Source is the complete MLA KV on Host;
Indexer is the device index KV. Resident and Hot are local device working
copies. Only Source and Indexer are externally transferable. The pool must
never publish a prefix based on Resident or Hot.

Use the original index in `KVCacheConfig.kv_cache_groups` as the group ID.
Do not renumber the two selected groups to 0 and 1: existing block tables,
metadata, and lookup keys index by the original group ID. Select groups using
the upstream `enable_kv_transfer` and prefix-cacheable declarations. For a
HiSparse layout, require exactly one Source and one Indexer group; reject
missing, duplicated, or unclassified roles. The first implementation also
rejects additional transferable groups until their representation is defined.
Legacy non-HiSparse group selection remains unchanged.

The scheduler and worker must agree on group IDs, block sizes, layout,
KV dtype, prefix key namespace, and the longest *complete* token boundary.
An object written with a different layout or schema version must not be
loaded under an old key.

## Scheduler to worker

Reuse the existing `AscendConnectorMetadata.requests: list[ReqMeta]` handoff.
The contract is:

| Existing field | Meaning for HiSparse pooling |
| --- | --- |
| `req_id` | Request identity; asynchronous saves also use `store_job_id` to distinguish generations of a reused request ID. |
| `kv_cache_group_ids` | Original Source and Indexer group IDs, not compacted indices. |
| `block_ids_by_group` | Outer list remains indexed by original group ID. Each selected entry contains local block IDs for that group. |
| `block_hashes` | Common prefix identity used to derive group-specific object keys. |
| `save_start_token`, `save_end_token` | Complete reusable write interval. Never include a partial tail. |
| `load_spec` | Existing local/remote hit lengths and load decision. A hit is bounded by the shorter complete Source and Indexer prefix. |

The scheduler passes logical identities and boundaries, not raw addresses,
transport directions, or device events. The worker derives Host versus device
placement, physical addresses, page bytes, strides, and layer segments from
its registered KV cache layout. Filtering groups must not change the
indexing of `block_ids_by_group`.

## Source readiness and buffer lifetime

Before submitting a Source PUT, the worker must establish that all requested
Host pages have completed the HiSparse Resident-to-Source mirror. A device
forward-completion event by itself does not establish this. The integration
needs a worker-side readiness operation with these semantics:

```python
def wait_source_ready(
    req_id: str, host_block_ids: Sequence[int], end_token: int
) -> None: ...
```

The caller acquires protection for those blocks before waiting and retains
it until the backend finishes reading them. Return means that the requested
complete Host prefix is durable. Cancellation, timeout, or invalid/reused
blocks fail the operation; partial success is not returned as readiness.
Bind this operation to HiSparse's Host write event and durable-prefix state.
The name above is a proposed adapter boundary, not an existing HiSparse API.
Indexer PUT separately waits for its device writes.

On GET, keep both destination groups valid until their transfers finish.
Source is loaded into the local Host cache, Indexer into local device cache.
Only after both groups for the same prefix succeed may AscendStore report
`finished_recving`. HiSparse then marks the Host import valid and restores
the tail page to Resident before inference continues. A failed or canceled
group must not trigger successful completion.

Memcache submits Host Source and device Indexer in separate batches with the
appropriate H-to-store/store-to-H and device-to-store/store-to-device
directions. This contract does not require an atomic two-object store
transaction: lookup must reject a prefix when either group is absent or has
an incompatible complete boundary.

## Ownership for three contributors

1. **Integration owner:** group selection, scheduler metadata, prefix lookup,
   HiSparse readiness adapter, and completion publication. Own
   `pool_scheduler.py`, `ascend_store_connector.py`, and any necessary
   `metadata.py` changes.
2. **Memcache owner:** Host/device cache registration, physical descriptors,
   split transfer batches, and per-group I/O results. Own `pool_worker.py`
   and `backend/memcache_backend.py`. Consume the metadata contract above
   without changing scheduler key semantics.
3. **Validation owner:** contract tests and cross-instance checks. Start with
   fixed Source/Indexer group IDs, a normal full prefix, and a block-boundary
   prefix. Independently verify missing one group, mismatched boundaries,
   unfinished Host mirror, canceled requests, and late asynchronous results.

The integration and Memcache owners first exchange one example table
containing group ID, role, block size, block IDs, key, memory type, page
bytes, stride, and complete token boundary. They can then implement their
separate files against that example. The integration owner connects the
readiness and completion paths; the validation owner checks the joined path.

## Acceptance boundary

The first implementation passes when one instance writes a prefix and a
fresh second instance loads Source and Indexer from Memcache, restores the
Resident tail, and matches a connector-free output reference. A one-group
hit, different prefix boundaries, failed GET, or unfinished Host mirror must
not be reported as a complete hit. Record the actual vLLM/vLLM-Ascend
commits, model and KV dtypes, TP layout, hardware, Memcache version, and
commands used. Documentation agreement and CPU mocks alone do not establish
NPU correctness.
