# Qwen3.8 Next pooling: payload transfer on Model Runner V2

## Current scope

Basic pooling targets Model Runner V2 without MTP, with synchronous non-layerwise AscendStore + Mooncake / memcache. Model Runner V1 is unchanged. See [the implementation overview](qwen38_pooling_mrv2.md) for allocation, role views and serving integration.

Main attention transfers the complete K/V history. Top-K selection only changes which history participates in one attention calculation. Compressed indexer history is transferred with the same connector flow. The private raw indexer ring, including packed positions, is rebuilt by replay after a safe compression-aligned restore boundary.

## Reuse existing metadata and transfer

PR #17249 supplies the native six-region allocation and reshape path. K and V occupy separate region planes. Registration uses complete raw/compressed rows, retaining packed fields; model binding may transpose heads and rows without copying.

AscendStore already records each view's base address, block payload length and physical block stride. The basic pooling implementation reuses that metadata and `ChunkedTokenDatabase.prepare_value()`: addresses are `base + block_id * stride`, and complete-block lengths cover only the view's payload. Shared storage is registered once, while each role view keeps its existing group and layer ordering. Padding between blocks and unrelated owners are not transferred.

There is no separate `cache_transfer_layout.py`, layout wire format or generic transfer batch. Arbitrary intra-block fragmented views and arbitrary partial recurrent-state snapshots are outside this basic scope. Safe boundaries and exact recurrent-state block mapping are handled by the existing connector lifecycle and the boundary-state hand-off.

## Validation

The CPU relocation test calls the existing worker metadata and token-database address methods. It saves independent bytes, overwrites the source, restores into different physical blocks and strides, compares selected payload, and checks all other bytes against sentinels. It does not call an external backend.

Earlier stage-2/3 probes exercised now-removed descriptor/snapshot APIs. Their historical counts are not validation of the current path. Real serving and backend continuation evidence is tracked separately in [stage 4](qwen38_pooling_mrv2_stage4.md); this simplification does not restart the NPU service.
