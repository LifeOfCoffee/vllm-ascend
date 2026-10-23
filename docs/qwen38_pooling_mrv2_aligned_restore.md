# Qwen3.8 Next: aligned boundary restore

Updated: 2026-10-23. Scope: Model Runner V2, no MTP, non-layerwise AscendStore + Mooncake / memcache, equal producer/consumer TP, synchronous load.

## Restore contract

Use the existing lookup, allocation, transfer, wait/error handling and forward flow. `qsa_snapshot.py` and `qsa_model_snapshot.py` remain deleted. Their independent manifests, tickets and receipts are not prerequisites for this path.

A reusable prefix must contain the actual state at one common length L. Rounding a token length does not create a recurrent checkpoint. For example, a state computed through token 1700 cannot represent token 1536: replaying the suffix from that later state would apply part of the history twice.

## Attention and state roles

The current model has 48 layers, with three GDN layers followed by one QSA layer, plus PLE state. The indexer compression ratio is 4. Selecting top-k for computation does not reduce the historical main K/V that must be stored.

| Cache role | Required at L | Recovery |
| --- | --- | --- |
| Main K and V | Complete attention history through L | Restore logical payload spans for all required history blocks, not only top-k entries |
| Compressed indexer | Completed compressed groups through L | Restore corresponding compressed rows with the same logical token boundary |
| Raw indexer and packed positions | No unfinished compression group may cross L | Rebuild from replay after an aligned compression boundary |
| GDN convolution and SSM | Actual states after the first L tokens | Restore the exact handed-off physical blocks |
| PLE convolution | Actual hidden-input history after L tokens | Restore its exact handed-off physical block |
| PLE n-gram context | Token history at L, including EOS semantics | Rebuild from request token IDs |

PLE uses n-gram size 3, convolution kernel size 4 and dilation 3. Its active convolution history is `(4 - 1) * 3 = 9` hidden-input positions. These are not interchangeable with nine token IDs. The allocated PLE view width is 12 positions; allocation width and active convolution history are different quantities.

## Safe lookup

Let G be the common transfer granularity, also aligned to the indexer compression ratio. For a prompt of N tokens:

```text
lookup_limit = floor((N - 1) / G) * G
```

Apply this bound before the existing joint lookup. The coordinator then selects a reachable boundary for which all necessary cache groups and rank objects exist. Do not query a later state and round the answer down to a state that might never have been saved. Keep a forward tail for logits and raw-indexer reconstruction.

Not every theoretical boundary must be cacheable. PLE prefill writes query-end state; it does not prove that every interior boundary has a checkpoint. When a required state is absent, lookup must select an earlier complete boundary or miss.

## Save real states through the existing send path

The image's upstream scheduler supplies a scheduler-only hand-off:

```text
SchedulerOutput.kv_connector_block_state
  get_block_ids(request_id)
  boundary_state_offloads[request_id]
    = [(group_id, physical_block_id, boundary_tokens), ...]
```

The connector consumes committed, positive block IDs with an aligned boundary and an available prefix hash. It creates state-only save metadata carrying the explicit `(physical_block_id, boundary_tokens)` pairs. The regular request metadata disables inferred saves for recurrent groups. This prevents a running state or an append-only tracker column from being mislabeled as an earlier state.

The scheduler selects the exact source blocks, then reuses the same block-pinning and event-registration tail as ordinary Mamba saves. The existing send thread waits for the producer event, generates group/rank keys through the existing token-key iterator, and transfers the normal payload layout. State-only jobs bypass ordinary sequential save watermarks and loaded-prefix skipping: a newly handed-off state can correspond to an older committed prefix. Completion still uses the existing event acknowledgements. The source reference is released after all participating workers acknowledge it.

## Restore before recurrent pre-copy and forward

```text
joint lookup at common boundary L
  -> allocate current destination block tables
  -> load main K/V and compressed-indexer history
  -> load each aligned state into its actual terminal block at L
  -> finish GET and process errors
  -> V2 recurrent state pre-copy into the working slot
  -> forward the tail from L and rebuild raw indexer state
```

For aligned groups, the terminal table entry is `L / group_block_size - 1`. Its physical block ID comes from the authoritative current table, including sparse/null entries. All required aligned groups must be present, and their terminal blocks must be valid. Invalid metadata fails before GET.

V2's existing deferred Mamba pre-copy is reused: its decision kernel may run earlier, but the actual source-to-working-state copy waits until connector loading completes. A failed load must not seed forward with a partially restored state.

Next already returns `AscendQwen4ExpModelState` from the model's existing `get_model_state_cls()` hook. That class inherits both `Qwen4ExpModelState` and `AscendMambaHybridModelState`, preserving PLE inputs and Ascend's pre-copy and completion hooks. Reuse the existing factory unchanged; no dynamic state-class composition is required for basic pooling.

GET exceptions, missing/short backend status arrays and per-key failures use the existing failed-request and invalid-block reporting, covering the whole hybrid request. No independent snapshot transaction is introduced.

## Non-contiguous storage

V2 builds native role views through the six-region allocation and reshape path from PR #17249. AscendStore uses its existing per-view base address, payload size and physical block stride. It registers shared backing once, but keeps role/owner identities distinct. Transfers omit holes, padding and other owners; sharing a storage pointer does not authorize copying the entire allocation.

## Why the raw ring can be rebuilt

The checked compression kernel emits only complete compression groups. At ratio 4 and L divisible by 4, the first post-restore group contains positions L through L+3. Whether they arrive in one forward or several, the raw keys and positions read for that group were written after restore. No pre-L unfinished group is needed.

This reasoning requires correct logical positions, slot mapping and private raw-buffer allocation. MTP writes are outside the scope. A poisoned-ring, split-forward model test remains a separate validation requirement; startup alone is not evidence for that property.

## Implementation and evidence

See [stage 4 status](qwen38_pooling_mrv2_stage4.md) for current serving evidence and limitations. Focused connector unit tests cover explicit boundary saves, pinning, watermarks, sparse table selection and failures. Earlier explicit-snapshot experiments are historical and do not validate this automatic path.
