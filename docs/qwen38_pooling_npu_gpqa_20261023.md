# Qwen3.8 Next basic pooling: NPU and non-thinking GPQA

## Scope and tested source

Model Runner V2, TP8/EP8, physical NPUs 8-15, eager execution, no MTP,
no layerwise transfer, synchronous AscendStore loading, Mooncake/memcache.
Model: Qwen3.8-Flash-Next-w8a8. The vLLM image version is
0.30.1rc1.dev447+gc4df37dfc.

Tested public source: a97d77f3f6a6852293b95c07598101c910765d12, tree
a977cf6b9f7db3af140157ccf62c42d3403a1345. Subsequent history reordering
does not change this source. The evidence source manifest records hashes
of all seven modified production files and image-local binary dependencies.
GBSA runtime dependencies were provisioned locally and are excluded from
the public branch.

Evidence directory:
`/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2/6c06145d-npu-gpqa`.

## Startup compatibility fix

The installed vLLM version exposes the one-argument
`_pool_bytes_per_block(kv_cache_groups)` API despite its version label.
Selecting the two-argument wrapper by version string caused startup to fail.
At the time of this recorded validation, the Ascend patch selected its wrapper
using the original function signature; both API variants passed focused tests.
That temporary fix and its tests were removed. A later version-label-only
experiment failed elsewhere and was reverted. The current dependency import
instead restores the image's original single-argument pool-budget function
and original vLLM version identity. These historical results retain their
original source/version manifest.

The successful run registered 318 blocks on every rank and logged the raw
indexer view shape `[318, 1, 4, 140]`. No missing-shape, allocation, or OOM
exception occurred. This run does not replace per-view bytewise sentinel
tests or prove every allocation address independently.

## Real external restoration

A 10,088-token prompt produced the answer `CEDAR-4729`. After the cold
request, local prefixes were reset, a separate long filler request executed,
and local prefixes were reset again. The warm request recorded:

- Local hit tokens: 0.
- External hit tokens: 7,680.
- Aggregate GET operations: 8 calls and 96 objects.
- Cold and warm output text: identical.

A no-connector request produced the same final identifier but different
reasoning text. Full cross-configuration output equivalence did not pass.
Per-rank restore DEBUG traces were unavailable at the active logging level;
aggregate external transfer metrics establish restoration, rather than an
empty trace file being interpreted as no transfer.

## GPQA Diamond: non-thinking diagnostic

All 198 local Diamond questions were evaluated once with a direct-answer
prompt, thinking disabled, temperature 0, generation seed 101, max_tokens 128,
and concurrency 4. Options were permuted with seed 42 plus the row index.
Dataset SHA256:
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.

This is a non-thinking diagnostic, not a chain-of-thought benchmark score
or a model-card reproduction.

| Configuration | Blocks | Correct | Accuracy | Errors / truncated / unparsed |
| --- | ---: | ---: | ---: | --- |
| AscendStore pooling | 318 | 104/198 | 52.53% | 0 / 0 / 0 |
| No connector, matched capacity | 318 | 99/198 | 50.00% | 0 / 0 / 0 |
| No connector, default capacity | 645 | 101/198 | 51.01% | 0 / 0 / 0 |

The matched-capacity runs disagree on six answers (indices 6, 63, 102, 119,
133, 158). Five are pool-only correct; one is wrong in both runs. This single
run shows no aggregate accuracy decrease but does not establish lossless
precision or a real quality improvement.

Standard short GPQA requests had zero local and external hits, so the above
table measures cold connector-enabled execution, not restoration accuracy.
The earlier 645-block baseline is archived separately. The image worker
budget path applies a conservative 0.4932 factor with a connector, explaining
the automatic capacity difference; no generic budget behavior was changed.

A serial repeat of the six discrepant questions on the pooled service changed
three predictions relative to its own concurrency-four run. Execution context
can affect answers, but this diagnostic does not establish the root cause.

## Separate long-prefix recovery stress

A deterministic 20-question subset used an added neutral long prefix and
serial requests. Both services had 318 blocks. Each pooled question followed
a local-prefix reset and restored 7,680 external tokens:

- Real external restorations: 20/20; local hits: 0.
- Total external hit tokens: 153,600.
- Aggregate GET operations: 160 calls and 1,920 objects.
- Pooling: 13/20 correct; no connector: 13/20 correct.
- Answer agreement: 19/20; the differing question (index 59) was wrong in both.
- No request errors, truncation, or unparsed answers.

The changed prompts and small subset make this a recovery stress test, not
a standard GPQA score. Exact answer equivalence remains unresolved.

## Reproduction artifacts and next run

Raw requests, responses, metrics, source manifest, launch scripts, per-rank
block counts and comparison JSON are retained in the evidence directory.
Focused connector/backend/metadata/transfer/scheduler/worker CPU selection
previously passed 502 tests; the API signature fix adds two passing tests.
Repository formatting could not run because pre-commit is absent.

A separate graph-enabled, thinking GPQA Diamond run is running under
`evidence/mrv2/graph-gpqa-standard-20261023`. Its results must not be inferred
from the eager non-thinking results above.

See [the graph GPQA protocol and status](qwen38_pooling_graph_gpqa_20261023.md).
