# Qwen3.8 Next: AscendStore initialization validation

This change enables allocation and registration diagnostics for the QSA caches
used by Qwen3.8 Next. External prefix reuse is not supported by this change:
raw-indexer boundary snapshots and consistent hybrid-state restoration remain
unimplemented.

## Configuration

Add the following settings to the existing AscendStore memcache configuration:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": false,
    "load_async": false,
    "initialization_only": true
  }
}
```

`initialization_only` is a boolean and defaults to `false`. When enabled it
requires eager execution, non-layerwise operation, and synchronous loading
configuration. It creates the real backend and registers device buffers, but
returns no external matches and submits no external load/save tasks. The
existing deferred backend initialization still runs after a real forward.
Normal local prefix caching remains the engine's responsibility.

A cache configuration containing a QSA `.indexer.raw_key_cache` owner is
rejected unless this mode is enabled. This prevents normal external reuse from
silently proceeding without the required raw-state restore contract.

## Model runner rationale

The independent allocation path previously called the legacy
`get_kv_cache_shape` method, which the unified QSA backend does not implement.
The model runner now binds unified views from the actual cache spec, retaining
the number of states, state width, page stride, and allocation slice offset.
The descriptor path uses the same binding helper.

For a backend declaring `uses_qsa_page_major_kv_view`, a main KV page contains
one dense K plane followed by one dense V plane. The returned K and V views
share the allocation. This reuses the existing backend capability and tuple
interface; no model-name dispatch or attention-kernel changes are introduced.
Splitting a canonical `[B,H,N,K+V]` tensor instead would interleave K/V between
tokens and violate GBSA's dense-token requirement, including in eager mode.
Compressed and raw owners retain their own smaller state counts and widths.

The helper checks dtype alignment, per-page payload, and bounds relative to
the supplied allocation slice. It creates views during initialization only:
there are no new device copies, allocations, synchronization points, or
`Tensor.item()` calls in the inference hot path. Cache allocation gates and
budgeting remain unchanged. Connector-enabled shared-backing/six-region
support is deferred. Backends with the legacy shape method retain their
existing construction path.

## Validation on the image baseline

Baseline Ascend commit: `2a450bd527f8a51be456de623369f7062a107427`.
Baseline vLLM commit: `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`.

- Targeted model-runner cache tests: 56 passed, 73 deselected.
- AscendStore connector and request-lifecycle tests: 137 passed, 12 skipped.
- An unrelated prolog-weight budget test fails on the original baseline as
  well because its NPU custom operator lacks a CPU implementation; it was
  excluded from the targeted cache test selection.
- Ruff 0.14.0 reports the same seven pre-existing import-order diagnostics on
  the baseline and modified files. No new diagnostics were introduced.
- TP8/DP1, eager, MTP3, local prefix caching and Mamba align mode: all eight
  ranks completed memcache initialization and real HBM registration. Each rank
  had 282 blocks, 76 underlying storages and 125 bound/transfer tensors. All
  views were in bounds; transfer pointers, payload lengths and page strides
  matched the live tensors, and pointers stayed unchanged across requests.
- A short 16-token generation matched the baseline smoke's token IDs. Two
  runs with a 768-token prompt plus 16 generated tokens also completed with
  identical output IDs.

Tests cover independent and descriptor view construction, multiple KV-head
counts, padded pages, aligned allocation slices, K/V write isolation, raw
position bit patterns, invalid bounds, configuration rejection, disabled
external work, and real backend initialization in validation mode.

The hardware smoke does not establish full numerical parity, external byte
round trips, restored continuation, or speculative accept/reject correctness.
Baseline repeatability and hybrid-state lifecycle validation remain open.
The next stage must verify actual transfers into different destination blocks
and restore main KV, compressed/raw indexer state, GDN and PLE at a common
committed boundary before external reuse is enabled.
