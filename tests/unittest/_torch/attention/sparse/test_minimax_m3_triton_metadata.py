# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_metadata import (
    MiniMaxM3TritonSparseAttentionMetadata,
)

pytestmark = pytest.mark.cpu_only


def _prefill_metadata(prefix_lens, extend_lens) -> MiniMaxM3TritonSparseAttentionMetadata:
    cu_seqlens_q = [0]
    for extend_len in extend_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + extend_len)
    kv_lens = torch.tensor(
        [prefix + extend for prefix, extend in zip(prefix_lens, extend_lens)], dtype=torch.int32
    )
    batch = len(extend_lens)
    return MiniMaxM3TritonSparseAttentionMetadata(
        is_prefill=True,
        req_to_token=torch.zeros(batch, 1024, dtype=torch.int32),
        slot_ids=torch.arange(batch, dtype=torch.int32),
        seq_lens=kv_lens,
        seq_lens_cpu=kv_lens.clone(),
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
    )


def _mixed_batch(num_rows: int):
    generator = torch.Generator().manual_seed(0)
    prefix_lens = torch.randint(0, 900, (num_rows,), generator=generator).tolist()
    extend_lens = torch.randint(0, 4, (num_rows,), generator=generator).tolist()
    # A context chunk first, then decode-like rows, as in a mixed batch.
    return [0] + prefix_lens, [37] + extend_lens


@pytest.mark.parametrize(
    ("prefix_lens", "extend_lens"),
    [
        ([0], [7]),
        ([0, 12, 30, 5], [6, 1, 1, 1]),
        # A zero-length row contributes no query token.
        ([16, 0, 3, 9, 0], [1, 4, 0, 2, 1]),
        _mixed_batch(63),
        ([], []),
    ],
)
def test_prefill_prepare_maps_each_query_token_to_its_row_and_position(prefix_lens, extend_lens):
    metadata = _prefill_metadata(prefix_lens, extend_lens)

    metadata.prepare()

    expected_rows = [row for row, extend in enumerate(extend_lens) for _ in range(extend)]
    expected_positions = [
        prefix + offset
        for prefix, extend in zip(prefix_lens, extend_lens)
        for offset in range(extend)
    ]
    assert metadata.q_batch_row.dtype == torch.int32
    assert metadata.q_positions.dtype == torch.int32
    assert metadata.q_batch_row.tolist() == expected_rows
    assert metadata.q_positions.tolist() == expected_positions
    assert metadata.max_seqlen_q == max(extend_lens, default=1)
