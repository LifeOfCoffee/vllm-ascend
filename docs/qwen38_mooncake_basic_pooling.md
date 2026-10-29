# Qwen3.8 Next pooling with AscendStore

## Current layout and validation boundary

This source follows PR #16981 at
`a2d7485e79df544b387ad52c4b0a445722a4394f`. QSA K/V and packed GDN
conv/SSM owners share whole physical pages with disjoint live block IDs;
raw/compressed indexer slabs remain separate. PLE and hidden owners use
private banks. Transfer registration must retain each view's physical
stride. AscendStore is the allowed V2 shared-layout transfer connector;
other connectors keep their prior allocation path.

CPU tests copy real packed-layout views through both
backend adapters, overwrite source bytes, reorder target block IDs and
check untouched owners and padding in the complete destination backing.

The model/NPU results below describe the earlier source stack and must
not be read as validation of this new packed layout. This update has not
started a model service or rerun native/model validation.

## Supported scope

The basic path uses Model Runner V2, no speculative decoding/MTP, and
AscendStore bulk transfers, Mooncake block-key layerwise transfers, and
Memcache GVA layerwise transfers. The validation scope is described below. The adaptation reuses the native six-region
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
  including bulk and layerwise eligibility independent of backend name, and
  retained background-load and other-connector exclusions.
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
contiguous; their backing was shared across owners. All view spans were
in bounds. The unique address intervals did not partially overlap; exact
view aliases in the shared allocation are described below.
Synthetic padded-stride cases extend coverage beyond the observed dense views.

The healthy restoration results above were collected before the extra
read guard was withdrawn. Its removal restores the original error path and
does not change successful transfer handling. No native heap error or
segfault appeared in that validated runtime. The original memcache service
was restored afterward.

## Layerwise adaptation

The initial layerwise launch took the generic hybrid allocation fallback:
`block_stride=393216` did not equal the full-page size `786432`.
The adaptation keeps the existing six-region allocation and reshape path
for AscendStore bulk and layerwise transfers. Layout eligibility does not
depend on the backend name, layerwise flag, or background-loading flag.
The real-model layerwise results below cover Mooncake; the Memcache and
asynchronous-loading results are recorded separately below.

Only these existing production files need changes:

| File | Purpose |
| --- | --- |
| `worker/v2/attn_utils.py` | Admit AscendStore bulk and layerwise paths to existing six-region views, including background loading. |
| `ascend_store/pool_scheduler.py` | Select the private-state replay boundary before bulk or block-key lookup. |
| `models/qwen4_exp/qsa.py` | Wait before indexer/main KV updates, open the existing attention transfer window after those updates, and save afterward. |
| `models/qwen4_exp/ple.py` | Wait before PLE short convolution reads its state. |
| `ascend_store/ascend_store_connector.py` | Load a physical layer once when PLE and attention share it, while copying each owner's state separately. |
| `worker/v2/model_states/mamba_hybrid.py` | Derive each layer's metadata range from its registered state-copy functions, matching the existing copy context. |

GDN has two state rows per layer; PLE has one. The measured model has 72 GDN
rows followed by the PLE range `[72, 73)`. Treating every owner as a two-row
state would address the wrong rows. Group order and per-type function counts
come from the existing context instead of a separate model snapshot.

QSA updates both the indexer and main KV before attention starts. Its transfer
window must therefore open after both updates. Without that window, future
layer prefetch tasks wait on a gate that QSA never signals. PLE executes
before attention in its decoder layer, so waiting only in GDN would be too late
for PLE. Both owners reuse the same physical-layer load and then run their
own existing deferred state copies.

`pool_worker.py`, backend implementations, metadata builders, object keys,
publication, block lifetimes and session tracking remain unchanged. No new
snapshot module, transfer-layout module, runner wrapper or eager-mode
validation is introduced. Model Runner V1 and MTP source are unchanged.

### Why the private-state boundary is selected first

A difference between total and prefix-cacheable group counts identifies
state that is excluded from external storage. For the current Next layout,
the private indexer tail is rebuilt by replay from a valid common boundary.
With a 1536-token transfer granularity, a 9216-token input is queried at
7680; a 9217-token input can be queried at 9216. The lookup and subsequent
restoration must select the same checkpoint. The predicate describes cache
capabilities, not a model name.

Dense/MLA layouts whose groups are all externally cacheable keep the original
lookup path. A hybrid model can also keep that path if all of its necessary
state is stored and restored together. Conversely, another model with the
same excluded replayable state must use the aligned-boundary handling too.
A group-count difference alone is not a universal proof that an arbitrary
future model's private state can be reconstructed; that property belongs to
the model's cache contract.

### Runtime used for layerwise validation

The model probe uses physical NPUs 8-15, TP8/EP8, V2, no MTP, eager execution,
`kv_both`, `backend: "mooncake"`, `use_layerwise: true` and `load_async: false`.
No layer reuse/offload option is enabled. Matching 0.3.14rc1 master/client
and explicit device RoCE are used as above.

The test master is started with:

```bash
mooncake_master --default_kv_lease_ttl=600000 ...
```

The installed master's default is 10000 milliseconds. A real eager forward
with diagnostic copies exceeded that lease and returned `-707`. Mooncake
documents that code as [LEASE_EXPIRED](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/error-code.md).
The 600000-millisecond lease covers this diagnostic run; it is a master
configuration setting and is not hardcoded in the adaptation. A deployment
must size its read lease for the complete session's transfer/compute duration.

### Regression and hardware evidence

- 549 related backend, scheduler, worker, metadata and connector tests passed.
- Five focused Mamba tests, two PLE tests, one QSA ordering test and eight
  existing connector-eligibility cases passed: 565 unit checks in total.
- Independent native two-client tests cover both bulk and ranged/session APIs,
  dense and padded strides, half/full payloads, reordered source/destination
  blocks, overwritten source and destination bytes, and whole-backing byte
  equality. The untouched gaps and other blocks remain unchanged.
- Independent native clients sharing the model's NPUs encountered HIXL
  connection timeouts. The ranged test passed after the previous model workers
  exited. The standalone regression is run with exclusive device use; arbitrary
  shared-device process coexistence was not validated.
- Full `format.sh ci` is unavailable because the container lacks `pre-commit`.
  Ruff reports one pre-existing 121-character line at
  `worker/v2/attn_utils.py:378`; it is unchanged by this adaptation.

### Real model layerwise results

The first three cases completed with a clean external restore. Each result
was compared with a native HBM cached continuation at the same boundary.

| Prompt tokens | Restored boundary | Compared output tokens | Local hits on restore | External hits | Token/logprob comparison |
| --- | --- | --- | --- | --- | --- |
| 9,216 | 7,680 | 4 | 0 | 7,680 | Exact; maximum absolute logprob difference 0 |
| 10,087 | 6,144 | 4 | 0 | 6,144 | Exact; maximum absolute logprob difference 0 |
| 16,384 | 12,288 | 4 | 0 | 12,288 | Exact; maximum absolute logprob difference 0 |
| 9,216, fresh independent master | 7,680 | 32 | 0 | 7,680 | Exact; maximum absolute logprob difference 0 |

The probe saves an identifiable prefix, clears local prefix caching, runs an
unrelated 22,000-token request to reuse blocks, clears local prefix caching
again, and poisons each external GET destination before transfer. All eight
ranks restored 11,544 payload fragments totaling 3,078,144,000 bytes. Every
fragment's SHA-256 matched a prior successful external PUT resolved by the
complete object key, object offset and fragment length. Every fragment was
loaded at a different source/destination rank or address. These are real
native ranged/session transfers, not local prefix-cache hits.

The restored roles on every rank include main K/V, the compressed indexer,
GDN convolution state, GDN recurrent state and PLE state. The private raw
indexer tail is excluded and rebuilt by aligned replay. Sparse QSA selects
top-k entries during computation; pooling still restores the required full
main K/V and compressed-indexer pages, not just previously selected entries.

Representative measured views on rank 0 are listed below. Strides are in
elements, not bytes.

| Role | Shape | Stride | Dtype |
| --- | --- | --- | --- |
| Main K and V, separate views | [318, 768, 1, 256] | [196608, 256, 256, 1] | BF16 |
| Compressed indexer | [318, 1, 192, 128] | [24576, 24576, 128, 1] | BF16 |
| Private raw indexer | [318, 1, 4, 140] | [560, 560, 140, 1] | BF16 |
| GDN convolution | [318, 3, 1280] | [3840, 1280, 1] | BF16 |
| GDN recurrent | [318, 6, 128, 128] | [98304, 16384, 128, 1] | FP32 |
| PLE | [318, 9, 10240] | [92160, 10240, 1] | BF16 |

Each rank has 121 typed views backed by one 11,778,700,544-byte allocation.
All view spans stay inside that allocation. There are 12 exactly aliased
intervals per rank; the unique intervals do not partially overlap. For
example, a main-K BF16 view and a GDN recurrent FP32 view can cover the same
byte interval under the existing shared allocation plan. A view address
therefore does not identify its live owner by itself: the verification uses
the object group, fragment order and actual view bounds together.

The registered allocation is within every rank's reported 10.98 GiB KV
budget, including a conservative allowance for the log's two-decimal
rounding. Whole-backing gap/other-block sentinel checks come from the
independent native NPU test; the model probe hashes transferred payloads and
does not claim to poison all holes while model computation is active.

A supplementary 32-token continuation initially missed the expected joint
external hit after accumulated test objects triggered eviction in the
16 GiB pool. Its identical output was produced by recomputation and is not
counted as successful restoration. The isolated fresh-master retry then
passed with zero local hits and 7,680 external hits. All 32 generated tokens
and token logprobs matched the native cached reference exactly, including
content recalling the identifier from the beginning of the prompt.
Every rank restored 433 fragments in this retry: 3,464 fragments and
919,093,248 bytes in total, all source hashes equal and all source/destination
rank/address pairs different. No eviction occurred in the isolated run.

Across the four successful cases, 15,008 fragments totaling 3,997,237,248
bytes matched their source hashes on all eight ranks. The evidence retains
the initial capacity-miss attempt separately rather than counting its
recomputed output as a restore.

These are functional restoration comparisons, not GPQA accuracy results.


## Asynchronous transfer integration

The six-region eligibility predicate accepts AscendStore independently of
`load_async`. The asynchronous integration changes only the existing
`worker/v2/attn_utils.py` production file, plus its eligibility test cases
and this document. The shared parser, transport, scheduler, runner, and
state-copy implementation keep their existing behavior.

For bulk transfers, `load_async: true` selects the existing
`KVCacheStoreRecvingThread`. The scheduler retains the allocated blocks and
keeps the request in `WAITING_FOR_REMOTE_KVS` until receive completion is
reported. Its next scheduled forward therefore uses the original Mamba
preprocess copy after the remote bytes have landed. The synchronous bulk
copy-after-load opt-in remains disabled for this asynchronous path.

For layerwise transfers, the existing scheduler reports a synchronous
request-level match while the worker prefetches and waits per physical
layer. `load_async: true` preserves that layerwise behavior. PLE and GDN
continue copying their state after the corresponding layer wait. No second
request-level background queue is introduced.

Both modes retain the existing receive-completion, failure reporting,
block-lifetime, key-generation, and private-state boundary handling.

### Memcache asynchronous NPU validation

The validation uses physical NPUs 8-15, TP8/EP8, Model Runner V2, no MTP,
eager execution, and `backend: "memcache"`. Both bulk and layerwise runs
set `load_async: true`. No layer reuse/offload option is enabled.

Each case compares a native cached continuation at the same boundary with
an external continuation. Before the external continuation, the probe saves
the prefix, resets prefix caching, executes an unrelated 22,000-token
request to reuse local blocks, and resets again. Every GET destination is
filled with 0xA5 before the actual transfer. The first GET on each rank is
also delayed by 750 milliseconds to verify completion ordering.

| Transfer mode | Prompt tokens | Restored tokens | Recomputed prompt chunks | Compared output tokens | Result |
| --- | --- | --- | --- | --- | --- |
| Bulk async | 9,216 | 7,680 | 1,536 | 4 | Exact tokens and logprobs |
| Bulk async | 16,384 | 6,144 | 8,192 + 2,048 | 4 | Exact tokens and logprobs |
| Layerwise with load_async=true | 9,216 | 7,680 | 1,536 | 4 | Exact tokens and logprobs |
| Layerwise with load_async=true | 16,384 | 6,144 | 8,192 + 2,048 | 4 | Exact tokens and logprobs |

Every external continuation had zero local-hit tokens and the expected
external-hit tokens. All four cases started computation at the restored
boundary. Across all eight ranks, 12,704 restored fragments totaling
3,355,508,736 bytes matched saved SHA-256 values. All 784 GET calls used
poisoned destinations, and every fragment moved to another rank or address.
The matching uses external object key plus fragment position for bulk,
and global GVA plus length for layerwise. Replicated KV heads can have a
different writer rank from the reader rank.

The bulk trace used `KVCacheStoreRecvingThread` with asynchronous lookup
results. All rank-level GET completions preceded the first scheduled
forward and the native Mamba preprocess copy. The layerwise trace used
`KVCacheStoreLayerRecvingThread`, with request-level synchronous lookup
and per-layer waits. All 37 recurrent owners per rank copied after their
corresponding physical-layer wait on every forward step. PLE consistently
used the single-row range `[72, 73)`.

Both modes retained 121 typed views on one 11,778,700,544-byte backing per
rank. All spans were inside that backing. Twelve exact aliases were
preserved per rank, with no partial overlap between unique view intervals.

The focused test run passed 34 cases, including 11 eligibility cases taken
from the actual gate and test definitions and existing asynchronous
scheduler, worker, receive, completion-registration, and failure tests.
Pre-commit spelling, forbidden-import, and boolean-context checks passed.
Ruff diagnostics match the unchanged baseline's existing E501 at line 378;
unrelated formatter changes were withdrawn to keep this patch minimal.
The full CI hook setup remains blocked by the proxy certificate while
provisioning Node.js. These model runs validate eager Memcache restoration;
they do not establish graph-mode asynchronous restoration, arbitrary layer
reuse/offload settings, performance, or accuracy scores.
