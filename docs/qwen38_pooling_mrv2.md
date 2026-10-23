# Qwen3.8 Next: Model Runner V2 pooling

Updated: 2026-10-23.

## Scope

Single-instance prefill/decode, Model Runner V2, no MTP, non-layerwise AscendStore + Mooncake / memcache. The deployed run uses eager execution on NPU 8-15. The supported scope is documented here; the connector adds no QSA-specific startup rejection for runner version, speculative decoding, layerwise transfer, asynchronous loading, or graph mode. Removing these configuration checks does not extend the adapted scope.

Persistent source: `/mnt/share/h00650310/qwen38-pooling-20261022/vllm-ascend-phase3`.
Branch: `feat/qwen38-pooling-mrv2`.
The source starts from the image checkout. Public publication excludes GBSA. Model Runner V1 is not modified.

## Stage 1: construct actual V2 views

`allocate_kv_cache_main()` follows the normal `_allocate_kv_cache()` -> `_reshape_kv_cache_v2()` path. The six-region allocation and role reconstruction follow [upstream PR #17249](https://github.com/vllm-project/vllm-ascend/pull/17249), using the existing `build_six_region_kv_cache_layout()` planner. The separate `worker/v2/cache_views.py` module and its descriptor dispatch are removed. Ordinary models keep their existing paths when the planner returns no six-region layout.

Unlike the referenced PR, this path is also enabled for synchronous, non-layerwise AscendStore, allowing registration of the actual role views. Other connectors retain their existing allocation path. Allocation and reshape geometry are reused directly from PR #17249; the basic pooling commit adds no generalized padding, indexer-shape or hidden/shared-owner allocation changes.

The existing planner supplies owner offsets, layer strides and region page strides for the following views:

| Region | Content | Compute view |
| --- | --- | --- |
| r1 | GDN convolution | `[blocks, *conv_shape]` |
| r2 | QSA K / GDN SSM | K: `[blocks, tokens, kv_heads, head_dim]`; SSM: its own spec shape/dtype |
| r3 | QSA V | `[blocks, tokens, kv_heads, head_dim]` |
| r4 | Raw indexer and packed positions | Registration: `[blocks, heads, ring_capacity, packed_width]` |
| r5 | Compressed indexer | Registration: `[blocks, heads, compressed_rows, width]` |
| r6 | PLE convolution | `[blocks, *state_shape]` |

The backing is shared; region and owner offsets remain distinct. K/SSM physical overlap is safe only because the scheduler assigns distinct physical blocks to their groups.

The existing PR #17249 constructors determine the role views; pooling does not repack them. The current full registered views have contiguous payload within each physical block, even though regions and owners occupy different offsets in shared backing. Arbitrary intra-block fragmented views are outside this basic path.

Basic pooling uses the normal connector mode with synchronous loading. Registration diagnostics add no separate connector mode.

## Stage 2: transfer logical payload

Reuse the existing AscendStore metadata flow: `_get_cache_block_metadata()` obtains the payload size and physical block stride; `_infer_cache_group_metadata()` records per-group view addresses in layer order; `ChunkedTokenDatabase.prepare_value()` produces transfer buffers. The address is `view_base + physical_block_id * block_stride`. For a complete block the transfer length is the block payload size. Shared backing registration remains deduplicated by storage, while transfer entries remain separate for each registered role view. No independent layout descriptor, wire format or transfer batch is introduced.

The focused CPU test uses these existing metadata methods to save independent byte objects, overwrite the source, and restore to different destination strides and physical block order. It compares each selected payload and checks sentinels everywhere else, including padding and unselected owners. This exercises address calculation; it is not a Mooncake/memcache or model-continuation test.

## Stages 3-4: joint boundary and serving integration

Boundary-state selection feeds the existing key iterator. Exact and ordinary state jobs share block pinning and event registration; memcache retains its original GET/PUT validation.

The existing connector flow now consumes real scheduler boundary-state hand-offs. It saves exact GDN/PLE blocks, protects their lifetime with existing save events, performs joint lookup below a safe prompt bound, loads current terminal state blocks, then runs the deferred V2 state pre-copy before forward.

Reuse the existing model-state factory without dynamic class composition. Next already returns `AscendQwen4ExpModelState`, which inherits both `Qwen4ExpModelState` and `AscendMambaHybridModelState`. This preserves PLE inputs and Ascend's state lifecycle through the existing model-specific class.

Main K/V and compressed indexer are restored through the same payload machinery. Raw indexer state is private and rebuilt by replay from a compression-aligned boundary. All necessary state must actually exist at that boundary; rounding a later state is invalid.

The independent `qsa_snapshot.py` and `qsa_model_snapshot.py` modules, their explicit worker APIs and dedicated probes/tests remain deleted. The transfer test exercises the existing AscendStore metadata path; the independent layout module and its tests are removed.

See [aligned restore contract](qwen38_pooling_mrv2_aligned_restore.md) and [current stage 4 evidence](qwen38_pooling_mrv2_stage4.md).

## Actual view geometry

Earlier initialization-only evidence used 285 physical blocks and 3-token MTP. It established V2 binding and registration, not automatic pooling. Its record contains 16 rank captures and 2,000 independently checked views. Do not apply that block count or memory budget to the current no-MTP run.

The current no-MTP run allocates 318 blocks per rank. The table below records the earlier 3-token-MTP registration capture, with B denoting its allocated block count; it must not be read as a new no-MTP rank capture:

| Registration view | Shape | Element stride | Dtype |
| --- | --- | --- | --- |
| Main K and V, separately | `[B,768,1,256]` | `[196608,256,256,1]` | BF16 |
| Raw indexer | `[B,1,8,140]` | `[1120,1120,140,1]` | BF16 packed |
| Compressed indexer | `[B,1,192,128]` | `[24576,24576,128,1]` | BF16 |
| GDN convolution | `[B,6,1280]` | `[7680,1280,1]` | BF16 |
| GDN SSM | `[B,6,128,128]` | `[98304,16384,128,1]` | FP32 |
| PLE | `[B,12,10240]` | `[122880,10240,1]` | BF16 |

Model binding transposes heads/rows for raw and compressed indexer views without copying. Their bound shapes are `[B,8,1,140]` and `[B,192,1,128]`. Transfer metadata follows the registration convention. With MTP disabled, the raw registration has four rows rather than eight: `[B,1,4,140]`, with a 1,120-byte page payload. The raw ring is private and is rebuilt by tail replay. Shared aliases must not be counted repeatedly as separate HBM allocations.

## Remaining scope

MTP, layerwise transfer, asynchronous load and producer/consumer TP mismatch are outside this basic path. Full poisoned-ring continuation, eviction races and backend failure-injection evidence must be reported separately from startup and unit tests.

## Validation scope

After removing diagnostics and reusing key generation, block pinning and event registration, the focused connector, backend, metadata, transfer, scheduler and worker suite passed 502 CPU tests. The earlier metadata, worker and relocation suite passed 204 CPU tests. The current CPU relocation test covers shared backing, separate K/V and packed indexer views, different source/destination strides, reordered blocks, source clearing and sentinels outside selected payload. Existing metadata and worker tests cover registration and boundary-state loading. Allocation and serving evidence from earlier revisions remains historical evidence, not an NPU rerun of this simplification. The service is not restarted for this change. Repository formatting requires `pre-commit`, which is absent in the image.

## Image interface compatibility

Use the image's original vLLM implementation
`c4df37dfcc7f72ca41cd43801a468526ecf99a2b` and its original version identity
`0.30.1rc1.dev447+gc4df37dfc`. Distribution metadata retains the original
`.empty` build suffix.

The dependency import restores `patch_kv_cache_utils.py` byte for byte from
the image snapshot `2a450bd527f8a51be456de623369f7062a107427`. The pool-budget
function accepts one group-list argument, matching the installed caller.
No PR #18224 version split or signature inspection is introduced by pooling.

The original V2 runner keeps its image-compatible `randomize_inputs` gate;
do not replace it with the newer main-only calling convention. Keep the
six-file PR #17249 adaptation and the model dependencies from PR #16981.
Unrelated platform/test formatting differences are also reverted.

The temporary `0.31.0` identity experiment failed because another existing
version gate passed `randomize_inputs` to a method without that parameter.
The version overlay has now been removed, original METADATA/RECORD restored,
and the reapply script disabled. No vLLM implementation code or shared
version backing file was changed.

Historical validation records retain their original manifests. A fresh
graph/thinking run under the restored image interfaces uses a separate
evidence directory so partial or failed experiments are not mixed.

The restored-image NPU restart passed startup and full decode graph smoke
checks on all eight ranks. A 10,087-token prompt restored 7,680 external
tokens with zero local hits after a filler request; cold/restored outputs
matched. The fresh thinking GPQA run is in progress. The focused planner,
V2 six-region and connector regression passed 91 tests.

## NPU and GPQA validation update

See [the NPU and non-thinking GPQA report](qwen38_pooling_npu_gpqa_20261023.md)
for completed real restoration, matched-capacity results, and remaining
answer-equivalence limitations. Graph/thinking results are tracked separately.
