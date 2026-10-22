# QSA and noncontiguous cache dependency baseline

This personal-fork baseline imports the QSA model integration and noncontiguous
cache support from the validated image. Phase 1 AscendStore initialization
changes are intentionally kept in a separate commit on the same review branch.

## Provenance and scope

- Public upstream baseline: `fbbc04965f459d97fe54684ed0f5cb350b4f9657`.
- Image source snapshot: `2a450bd527f8a51be456de623369f7062a107427`.
- Compatible vLLM source used for validation:
  `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`.
- 53 source/test files are imported byte for byte from the image snapshot.
- GBSA operator implementations, Python GBSA adapters, C++ sources, build
  changes, operator documentation, and private image commit history are excluded.
- No phase 1 initialization-only behavior is included in this baseline.

The old fork main is an ancestor of the public baseline above. The review branch also incorporates the existing public upstream commits up to that baseline;
it does not import the image's private merge history.

## Runtime dependency boundary

The QSA owner retains its call to `vllm_ascend.ops.gbsa_qsa`. That module and
its supporting GBSA implementation are deliberately not included here.
Running QSA therefore requires an externally provisioned compatible GBSA runtime,
including its Python integration, in addition to the compatible vLLM and CANN
environment. A fresh clone of this public repository alone is not sufficient
to build or run QSA. There is no dense-attention fallback.

Validation performed in the existing image uses that image's separately
provisioned operator/runtime components. Such validation does not demonstrate a
standalone build from this public repository.

## Cache and integration scope

The imported code includes the QSA main attention cache, compressed/indexer
cache integration, page-strided tensor views, shared storage layout helpers,
cache grouping and binding, model input adaptation, and MTP integration.
The model-runner and cache utility patches preserve the image's upstream API
integration. The long-term removal path is to adopt equivalent upstream cache
layout and model integration APIs once they support this model and version.

These dependencies do not implement external cache snapshot/restore. In
particular, main KV, indexer state, raw compression windows, and recurrent states
still need a coherent persistence and recovery contract in later pooling stages.
