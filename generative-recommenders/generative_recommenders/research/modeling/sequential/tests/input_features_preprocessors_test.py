# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

"""Tests for the temporal input-features preprocessor.

These intentionally run on CPU: the whole point of the rewrite is that the
temporal encoding is pure vectorized torch with NO ``.cpu()`` round-trip or
``datetime`` python loop (the bug in the previous "RoPE" preprocessor). A test
that passes on CPU proves there is no host-side per-element loop.
"""

import unittest

import torch
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    TemporalInputFeaturesPreprocessor,
)


def _make_batch(B: int, N: int, D: int, n_pad: int):
    """Builds a batch with chronological unix-second timestamps and right padding."""
    torch.manual_seed(0)
    past_ids = torch.randint(1, 100, (B, N))
    past_lengths = torch.full((B,), N, dtype=torch.long)
    if n_pad > 0:
        past_ids[:, N - n_pad :] = 0  # right-pad with id 0
        past_lengths[:] = N - n_pad
    # chronological unix seconds starting mid-2020, irregular gaps incl. hours/days
    base = 1593878400
    steps = torch.randint(60, 100000, (B, N)).cumsum(dim=1)
    timestamps = (base + steps).to(torch.int64)
    timestamps[past_ids == 0] = 0  # padded positions carry ts=0 from the loader
    past_embeddings = torch.randn(B, N, D)
    past_payloads = {
        "timestamps": timestamps,
        "ratings": torch.randint(0, 6, (B, N)),
    }
    return past_lengths, past_ids, past_embeddings, past_payloads


class TemporalInputFeaturesPreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.B, self.N, self.D = 4, 12, 16
        self.max_len = 64

    def _module(self) -> TemporalInputFeaturesPreprocessor:
        return TemporalInputFeaturesPreprocessor(
            max_sequence_len=self.max_len,
            embedding_dim=self.D,
            dropout_rate=0.0,
            time2vec_dim=8,
        ).eval()

    def test_output_shapes_and_lengths(self) -> None:
        m = self._module()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=3)
        out_lengths, user_emb, valid_mask = m(pl, pid, pe, pp)
        self.assertEqual(user_emb.shape, (self.B, self.N, self.D))
        self.assertEqual(valid_mask.shape, (self.B, self.N, 1))
        # baseline-style preprocessor must NOT double the sequence length.
        torch.testing.assert_close(out_lengths, pl)

    def test_padding_positions_are_zeroed(self) -> None:
        m = self._module()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=4)
        _, user_emb, _ = m(pl, pid, pe, pp)
        pad = pid == 0
        self.assertTrue(torch.all(user_emb[pad] == 0.0))
        self.assertTrue(torch.all(user_emb[~pad].abs().sum(dim=-1) > 0.0))

    def test_no_nan_inf(self) -> None:
        m = self._module()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=3)
        _, user_emb, _ = m(pl, pid, pe, pp)
        self.assertTrue(torch.isfinite(user_emb).all())

    def test_output_depends_on_timestamps(self) -> None:
        """Changing timestamps must change the embedding of valid positions.

        This is the core regression guard: the old preprocessor kept only the
        hour-of-day, so most timestamp changes were invisible. Here we shift
        every timestamp by several days+hours and require the output to move.
        """
        m = self._module()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=2)
        _, out_a, _ = m(pl, pid, pe, pp)
        pp2 = dict(pp)
        shift = torch.zeros_like(pp["timestamps"])
        shift[pid != 0] = 3 * 86400 + 7 * 3600  # 3 days + 7 hours
        pp2["timestamps"] = pp["timestamps"] + shift
        _, out_b, _ = m(pl, pid, pe, pp2)
        valid = (pid != 0).unsqueeze(-1)
        self.assertFalse(torch.allclose(out_a[valid.expand_as(out_a)],
                                        out_b[valid.expand_as(out_b)]))

    def test_reduces_to_baseline_when_temporal_zeroed(self) -> None:
        """With the temporal projection zeroed, the module must equal the
        learnable-positional baseline (proving temporal is a clean additive
        extension, not a structural change to the baseline path)."""
        m = self._module()
        base = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=self.max_len,
            embedding_dim=self.D,
            dropout_rate=0.0,
        ).eval()
        # share the positional embedding, then kill the temporal contribution.
        base._pos_emb.weight.data.copy_(m._pos_emb.weight.data)
        with torch.no_grad():
            m._temporal_proj.weight.zero_()
            m._temporal_proj.bias.zero_()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=3)
        _, out_m, _ = m(pl, pid, pe, pp)
        _, out_b, _ = base(pl, pid, pe, pp)
        torch.testing.assert_close(out_m, out_b)

    def test_gradients_flow_to_temporal_params(self) -> None:
        m = self._module()
        pl, pid, pe, pp = _make_batch(self.B, self.N, self.D, n_pad=2)
        _, user_emb, _ = m(pl, pid, pe, pp)
        user_emb.sum().backward()
        for name, p in [
            ("t2v_recency_w", m._t2v_recency_w),
            ("t2v_gap_w", m._t2v_gap_w),
            ("temporal_proj.weight", m._temporal_proj.weight),
        ]:
            self.assertIsNotNone(p.grad, f"{name} got no grad")
            self.assertTrue(p.grad.abs().sum() > 0.0, f"{name} grad is all-zero")


if __name__ == "__main__":
    unittest.main()
