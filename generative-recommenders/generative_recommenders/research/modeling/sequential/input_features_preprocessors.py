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

# pyre-unsafe

import abc
import math
from typing import Dict, Tuple

import torch

from generative_recommenders.research.modeling.initialization import truncated_normal


class InputFeaturesPreprocessorModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        #print(past_payloads.keys())
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        #print("keys in data",past_payloads.keys())
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask

class TemporalInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    """Learnable-positional preprocessor + a learnable *temporal* embedding.

    Identical to ``LearnablePositionalEmbeddingInputFeaturesPreprocessor``
    (item embeddings scaled by ``sqrt(d)`` + learnable absolute position
    embedding + dropout + padding mask), but additionally adds a temporal
    embedding derived from the interaction timestamps. Because the temporal
    term is added on top of the untouched baseline path, the two form a clean
    A/B: switching ``input_preproc_type`` between ``learnable_positional`` and
    ``temporal`` changes *only* whether time is modeled.

    This replaces the previous (mis-named) "RoPE" preprocessor, which (a) was
    an additive sinusoid, not RoPE, (b) discarded everything except the
    hour-of-day (losing recency / ordering / inter-event gaps), and (c) ran a
    per-element ``datetime`` python loop on CPU every forward step.

    Temporal features for a valid position ``i`` with unix-seconds ``t_i``:
      * ``log1p(recency) = log1p(t_last - t_i)`` -- time before the most recent
        event in the sequence (captures absolute recency);
      * ``log1p(gap)     = log1p(t_i - t_{i-1})`` -- inter-event spacing;
        both encoded with Time2Vec (one linear term + sinusoids w/ learnable
        frequency & phase), plus
      * cyclical hour-of-day and day-of-week (sin/cos), via modular arithmetic
        on the unix seconds.
    Everything is vectorized on the input device (no CPU / python loops). The
    concatenated features are projected to ``embedding_dim`` and summed in.

    NOTE on units: KuaiRec ``small_matrix.csv`` timestamps are unix *seconds*
    (the pipeline keeps them as ``int`` seconds), so ``seconds_per_day=86400``.
    """

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
        time2vec_dim: int = 16,
        seconds_per_day: int = 86400,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._seconds_per_day: int = seconds_per_day

        # Time2Vec for the two continuous features (log-recency, log-gap):
        # index 0 is a linear term, the rest are sinusoids with learnable
        # frequency (w) and phase (b).
        self._t2v_dim: int = time2vec_dim
        self._t2v_recency_w = torch.nn.Parameter(torch.empty(time2vec_dim))
        self._t2v_recency_b = torch.nn.Parameter(torch.empty(time2vec_dim))
        self._t2v_gap_w = torch.nn.Parameter(torch.empty(time2vec_dim))
        self._t2v_gap_b = torch.nn.Parameter(torch.empty(time2vec_dim))

        # Project [recency_t2v | gap_t2v | hour(sin,cos) | dow(sin,cos)] -> D.
        self._temporal_proj = torch.nn.Linear(2 * time2vec_dim + 4, embedding_dim)

        self.reset_state()

    def debug_str(self) -> str:
        return f"t2v{self._t2v_dim}_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        for w in (self._t2v_recency_w, self._t2v_gap_w):
            torch.nn.init.normal_(w, mean=0.0, std=1.0)
        for b in (self._t2v_recency_b, self._t2v_gap_b):
            torch.nn.init.uniform_(b, 0.0, 2.0 * math.pi)
        torch.nn.init.xavier_uniform_(self._temporal_proj.weight)
        torch.nn.init.zeros_(self._temporal_proj.bias)

    def _time2vec(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, N) -> (B, N, t2v_dim). Index 0 linear, the rest sinusoidal.
        v = x.unsqueeze(-1) * w + b
        return torch.cat([v[..., :1], torch.sin(v[..., 1:])], dim=-1)

    def _temporal_embedding(
        self, timestamps: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # timestamps: (B, N) int64 unix seconds; valid: (B, N) bool.
        ts = timestamps.float()
        spd = float(self._seconds_per_day)

        # recency = t_last - t_i, using the max *valid* timestamp per row.
        masked_ts = torch.where(valid, ts, torch.full_like(ts, -1.0))
        t_last = masked_ts.max(dim=1, keepdim=True).values  # (B, 1)
        log_recency = torch.log1p((t_last - ts).clamp(min=0.0))

        # gap = t_i - t_{i-1} (0 for the first position), clamped to >= 0.
        gap = torch.zeros_like(ts)
        gap[:, 1:] = (ts[:, 1:] - ts[:, :-1]).clamp(min=0.0)
        log_gap = torch.log1p(gap)

        # cyclical time-of-day and day-of-week from unix seconds (UTC).
        ang_day = 2.0 * math.pi * torch.remainder(ts, spd) / spd
        # epoch 1970-01-01 was a Thursday -> +4 makes Monday=0.
        dow = torch.remainder((ts / spd).floor() + 4.0, 7.0)
        ang_week = 2.0 * math.pi * dow / 7.0

        feats = torch.cat(
            [
                self._time2vec(log_recency, self._t2v_recency_w, self._t2v_recency_b),
                self._time2vec(log_gap, self._t2v_gap_w, self._t2v_gap_b),
                torch.sin(ang_day).unsqueeze(-1),
                torch.cos(ang_day).unsqueeze(-1),
                torch.sin(ang_week).unsqueeze(-1),
                torch.cos(ang_week).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self._temporal_proj(feats)  # (B, N, D)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        valid = past_ids != 0  # (B, N)
        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = user_embeddings + self._temporal_embedding(
            past_payloads["timestamps"], valid
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = valid.unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask