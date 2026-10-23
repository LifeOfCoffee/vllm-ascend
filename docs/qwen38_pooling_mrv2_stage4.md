# Qwen3.8 Next: stage 4 basic pooling

Updated: 2026-10-23. Target: Model Runner V2, no MTP, non-layerwise AscendStore + Mooncake / memcache.

## Connected path

- Bound lookup before probing: `floor((prompt_tokens - 1) / G) * G`.
- Consume authoritative current block tables and committed `boundary_state_offloads` from the existing scheduler.
- Save exact GDN/PLE state blocks with existing group/rank keys and payload transfer; never label an arbitrary running state as an earlier checkpoint.
- Pin handed-off blocks through the existing all-worker save acknowledgement.
- Restore every required aligned group's terminal block at the common hit boundary. Invalid or incomplete metadata fails before GET.
- Finish GET before V2 recurrent pre-copy and forward. Missing objects and backend failures follow the existing hybrid-request failure path.
- Reuse Next's existing `AscendQwen4ExpModelState`, which already preserves PLE inputs and inherits the Ascend hybrid lifecycle.
- Recompute the tail and rebuild private raw-indexer state at a compression-aligned boundary.

The independent snapshot modules remain deleted. Model Runner V1 and the upstream vLLM checkout are unchanged.

## Enable the basic path

Use the current image's QSA-capable vLLM scheduler, which supplies committed boundary-state hand-offs. No new upstream vLLM edits are required by this change. Enable `VLLM_USE_V2_MODEL_RUNNER=1`, prefix caching and `--mamba-cache-mode align`, without speculative decoding. Keep the existing memcache backend configuration and use normal connector mode:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": false,
    "load_async": false
  }
}
```

Eager mode was used for validation; this implementation adds no eager-only check. Producer and consumer use the same TP configuration in this scope.

## Validation so far

Earlier stage-4 revisions reported 300 focused AscendStore tests and 46 V2 model-state/deferred-copy tests. Those historical counts include diagnostic and generic adapter coverage that has since been removed. Current focused validation is recorded in the implementation overview; the historical counts do not validate later scope reductions.

An initial real service run on NPU 8-15 used TP8, V2, eager, no MTP, synchronous non-layerwise memcache and normal automatic pooling. It initialized 318 blocks per rank without a missing-shape or allocation error. A 10,021-token prompt produced 24 tokens. Boundary-state save event 0 received all eight worker acknowledgements before source blocks 12 and 13 were released.

A subsequent deterministic request used a 10,052-token prompt, temperature 0, seed 101 and a 128-token output limit. It asked for `CEDAR-4729` from the first sentence after a long repeated-text suffix. The first request missed externally. After resetting local prefix mappings, the repeated request reported zero local cached tokens and a 7,680-token external hit. All eight ranks executed GET: 12 objects and 433 payload spans per rank, 114,886,656 bytes per rank. The digest of every loaded span matched a saved object with the same key across the participating producer ranks. Producer head-rank sharing means comparison must include all producer ranks, not just the consumer's own PUT log.

The destination tables were authoritative current tables: aligned group 2 restored physical block 27 and group 3 restored block 28 at terminal column 9 (`7680 / 768 - 1`). Main/history groups used the ordinary transfer path. The remaining 2,372 prompt tokens were computed after restore. First computation, external restoration and a subsequent local-prefix hit all returned the same complete 91-token output, ending in `CEDAR-4729`.

These request captures came from earlier revisions. The current implementation reuses Next's existing `AscendQwen4ExpModelState` and the original model-state factory: that class already combines PLE input preparation with the Ascend lifecycle. The captures are historical serving evidence, not an NPU rerun of the later factory and connector simplifications.

The per-rank digest probe is an external diagnostic extension, not part of the implementation or a required serving path. It adds synchronization and is not performance evidence.

A final ordinary-service rerun removed the diagnostic extension and used a fresh prefix. The initial request recorded zero external hits. After local prefix reset, the second request recorded zero local hits, 7,680 external cached tokens, eight completed GET calls and 96 loaded keys in total. Both complete outputs were identical (60 completion tokens, ending in `CEDAR-4729`). This confirms that the basic loop does not depend on synchronization added by the diagnostic probe.

Evidence root: `/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2/basic-pooling`.
Unit-test log: `/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2-boundary-unit.log`.
Real request evidence: `probe2-request.json`, `probe2-{first,hit,local}-response.json`, corresponding metrics, `probe-rank{0..7}.jsonl` and `probe2-transfer-summary.json` under the evidence root. The ordinary-service rerun is recorded in `final-request.json`, `final-{first,hit}-response.json`, `final-{first,hit}-metrics.txt` and `service-final.log`.

## Validation still required

Strengthen the completed common-boundary GET/continuation check with overwritten local allocations, poisoned raw-ring split-forward cases, concurrent eviction and backend failure injection. A separate no-connector model comparison has not yet been run for this implementation. The successful single-request closed loop does not establish these broader properties.

See [restore contract](qwen38_pooling_mrv2_aligned_restore.md) for state roles and the difference between a true checkpoint and an arithmetically aligned length.


## NPU and GPQA validation update

See [the NPU and non-thinking GPQA report](qwen38_pooling_npu_gpqa_20261023.md)
for completed real restoration, matched-capacity results, and remaining
answer-equivalence limitations. Graph/thinking results are tracked separately.
