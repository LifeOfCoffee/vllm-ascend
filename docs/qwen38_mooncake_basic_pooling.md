# Qwen3.8 Next basic pooling with Mooncake

## Supported scope

The basic path uses Model Runner V2, no speculative decoding/MTP, and
synchronous, non-layerwise AscendStore transfers. It reuses the native six-region
allocation and reshape path, existing cache metadata, keys, publication events,
joint lookup, and aligned restoration boundary.

The model adaptation still enables the six-region path through
`worker/v2/attn_utils.py` and bounds private-state lookup in
`ascend_store/pool_scheduler.py`. The existing `pool_worker.py` transfer
flow is reused without production changes.

No model-specific transfer-layout or snapshot module is added.

## Runtime and transport

The validated combination was matching Mooncake master/client 0.3.14rc1,
CANN 9.2.0, HDK 26.1.1, and explicit device RoCE. The runtime was installed
in a separate directory; the image's global packages and vLLM version were
preserved.

A standalone two-client test reproduced the original HCCS IPC-import failure,
independently of the model and cache views. On the tested machine:

| Combination | Result |
| --- | --- |
| Installed 0.3.11.post1, default HCCS | Cross-client transfer failed; importing native modules also aborted at process exit. |
| Matching 0.3.14rc1, default HCCS | Cross-client transfer still failed. |
| Matching 0.3.14rc1, explicit device RoCE | Cross-client PUT/GET and clean client shutdown passed. |

These are results for this environment, not a claim that HCCS is unsupported
on every machine. Mixing the tested 0.3.14rc1 client with the old master failed
during segment registration, so master/client versions must match.

Configure the existing Ascend transport options before starting clients:

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device"],"store.comm_resource_config.protocol_desc":["roce:device"]}'
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

Use `protocol: "ascend"`, a reachable metadata server and matching master
in `mooncake.json`. The backend selection remains:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": false,
    "load_async": false
  }
}
```

In the installed old runtime, assigning the Fabric variable the string
`"0"` still entered Fabric mode. The validated direct-registration setup
unsets it. No new environment variable or configuration guard is introduced.

## Incomplete reads

The extra synchronous hybrid-read guard was withdrawn. `pool_worker.py`
now matches the prerequisite baseline, and its dedicated fail-closed tests
were removed. Backend status normalization and existing error handling are
reused. This minimal adaptation does not add transparent recovery for an
incomplete external hybrid read.

The earlier injected-failure result was obtained with the withdrawn guard;
it is not evidence that the current implementation rejects partial reads
before forward. Healthy restoration and failed-transfer recovery are
separate validation claims.

## Regression coverage

- `tests/ut/distributed/ascend_store/test_qsa_basic_pooling.py` exercises both
  backend adapters with independent byte objects: six disjoint regions,
  dense/padded strides, reordered source and destination blocks, half/full
  payload fragments, source/target overwrite, and whole-buffer sentinels.
  It also checks the 9,216-token boundary regression, dense fallback,
  and Mooncake byte-count/error normalization.
- `tests/ut/worker/test_attn_utils_v2.py` checks connector eligibility,
  including both backend names and unchanged layerwise/async exclusions.
- `tests/e2e/pull_request/two_card/test_qsa_mooncake_buffers.py` starts a
  matching master and two independent NPU clients. It verifies four
  stride/fragment combinations, cross-client GET and PUT, overwritten source
  bytes, reordered destination blocks, whole-allocation equality, missing
  objects, and clean process exit.

Run the NPU buffer regression with two visible cards and the transport
configuration above:

```bash
python -m pytest -q tests/e2e/pull_request/two_card/test_qsa_mooncake_buffers.py
```

The test loads no model and owns only the master/clients it creates.

## Real model validation

Validation used TP8/EP8 on eight NPUs, V2, no MTP, synchronous non-layerwise
transfer, and decode graphs.

| Prompt tokens | Restored boundary | Output tokens and logprobs |
| --- | --- | --- |
| 9,216 | 7,680 | Identical to native cached continuation |
| 10,087 | 6,144 | Identical to native cached continuation |
| 16,384 | 12,288 | Identical to native cached continuation |

Each case cleared local prefix caching, inserted an unrelated 22,000-token
request, and overwrote destination payloads immediately before external GET.
The restored request had zero local prefix hits and the expected external
hit count.

All eight ranks were checked: 320 per-rank key reads, 11,544 payload fragments
and 3,078,144,000 bytes matched their source hashes. All 1,536 sampled
gap/other-owner sentinels were preserved. Source hashes are resolved by the
external object key, including replicated objects written by another rank.

Each rank exposed 121 views sharing one storage. Individual views were
contiguous; regions and owners occupied separate ranges of the shared
backing. All view spans were in bounds and distinct ranges did not overlap.
Synthetic padded-stride cases extend coverage beyond the observed dense views.

The healthy restoration results above were collected before the extra
read guard was withdrawn. Its removal restores the original error path and
does not change successful transfer handling. No native heap error or
segfault appeared in that validated runtime. The original memcache service
was restored afterward.

## Layerwise probe after withdrawing the read guard

A real TP8/EP8 launch used physical NPUs 8-15, Model Runner V2, no MTP,
`kv_both`, `use_layerwise: true`, `load_async: false`, and eager execution.
It used the matching 0.3.14rc1 master/client and device RoCE described above.
All seven required layerwise client methods were available.

The service failed during `_allocate_kv_cache`, before any inference request
or external restoration. For `language_model.model.layers.3.self_attn.attn`,
the generic hybrid path reported:

| Descriptor field | Actual value in bytes |
| --- | ---: |
| `layer_stride` | 327127872 |
| `block_stride` | 393216 |
| `spec.page_size_bytes` | 786432 |

The generic path requires `block_stride == spec.page_size_bytes` and a
contiguous per-layer span. Those assumptions do not describe this Next
layout. The existing `_supports_six_region_cache_layout` predicate returns
false for layerwise transfers, so both allocation and reshape skip the
six-region path. The error occurs before the layerwise transfer protocol
can be validated with model data; callable backend APIs alone do not prove
that the complete model path works.

No production guard or layout predicate was relaxed for this probe. The
next adaptation must establish correct six-region owner views for the
layerwise path, then validate per-layer completion and aligned recurrent
state restoration before forward. Removing the allocation check alone
would not establish correct addresses or state consistency.

After the read-guard rollback, 511 related backend, layerwise, worker,
scheduler, metadata and QSA tests passed. Seven existing connector-eligibility
cases were also executed in isolation against the extracted production
predicate. Full `format.sh ci` could not run because the container lacks
`pre-commit`; the modified unit file passed Ruff 0.14.0 lint and format checks.
