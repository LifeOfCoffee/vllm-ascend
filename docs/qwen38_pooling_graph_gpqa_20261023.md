# Graph-enabled GPQA Diamond validation

## Configuration

This run uses Model Runner V2 on physical NPUs 8-15, TP8/EP8, no MTP,
no layerwise transfer, and synchronous AscendStore/Mooncake/memcache loading.
Both pooling and no-connector services use 318 cache blocks, max_model_len
32768, max_num_seqs 4, and max_num_batched_tokens 8192.

ACL graph configuration:
`{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}`.
Prefill runs eagerly; decode uses full ACL graph replay. There is no
`--enforce-eager` flag in either launch command.

The current replacement run uses the image's original version identity
`0.30.1rc1.dev447+gc4df37dfc`, original implementation commit
`c4df37dfc`, and original single-argument pool-budget wrapper. Imported
PR #18224 interface changes and the temporary version override are removed.
QSA startup configuration gates remain removed. Source hashes are recorded
in the new run manifest.
The publication stack is PR #16981 dependency at historical revision
2e785bba69e3348b1b2cc8a7986a9a2898d41af1, PR #17249 at
4e6f56c3ea1d02b44eebd9b2ad2bfe2cb693afdb, then one basic-pooling commit.
No GBSA source is published; compatible image-local runtime dependencies
are listed in the evidence manifest.

## Evaluation protocol

- GPQA Diamond: all 198 questions, one repeat, pass@1.
- Zero-shot chain-of-thought question template and Random(0) option
  permutation from [simple-evals GPQA](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py)
  and its [multiple-choice template](https://github.com/openai/simple-evals/blob/main/common.py).
- Thinking enabled through the model chat template.
- Model generation settings: temperature 1.0, top_p 0.95, top_k 20.
- Generation seed 101, max_tokens 28672, request concurrency 4.
- Final answer extraction uses the reference Answer-letter regular expression
  after the thinking block, with unparsed and truncated outputs counted.
- Identical question order, permutations, payloads, and capacity for both services.

This is a single 198-question thinking evaluation, not the default multi-repeat
simple-evals aggregate or a reproduction of a published model-card score.
It is separate from the earlier 128-token non-thinking diagnostic.

Dataset SHA256:
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.

## Confirmed before the full run

All four concurrent smoke requests completed successfully with unequal
generation lengths, exercising batch shrink. The service logged:
`ACL graph replay is active for the V2 target model`.

The restored-image run's real external restoration probe used a 10,087-token prompt with a fresh
namespace, local-cache resets, and an intervening filler request.
The warm query recorded zero local hits and 7,680 external hit tokens.
Cold and restored outputs were identical.

## Run status and evidence

The original partial run is archived. The `0.31.0` version-only experiment
failed during startup because an existing version gate forwarded
`randomize_inputs` to an unchanged upstream method without that parameter.
That experiment is retained separately and does not contribute a GPQA score.

The original version source, METADATA and RECORD are now restored. The
pool-budget function is restored from the image snapshot without signature
inspection or a new version branch. The restored-image service started successfully on port 8091 (health 200),
all eight ranks completed graph capture, and the four-request smoke check
confirmed real ACL graph replay. The external restoration probe passed
with zero local hit tokens, 7,680 external hit tokens, and identical cold
and restored output after a filler request. The pooling GPQA arm is now
running; both arms use the original image interfaces. Final accuracy is
unavailable until both arms complete. A focused cache-planner, V2
six-region and connector test selection passed 91 tests.

Persistent evidence:
Current replacement run:
`/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2/graph-gpqa-standard-image-interfaces-20261023`.

Failed version-only experiment:
`/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2/graph-gpqa-standard-v031-20261023`.

Original partial run:
`/mnt/share/h00650310/qwen38-pooling-20261022/evidence/mrv2/graph-gpqa-standard-20261023`.

The directory contains launch scripts, `run_graph_gpqa.py`, exact configuration
and requests, source manifest, smoke and restoration responses, raw per-question
outputs, progress, metrics, and service/controller logs. Results survive client
disconnects. The controller leaves the graph-enabled pooled service on port 8091
after successful completion of both arms.
