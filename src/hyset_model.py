from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _standardize(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).lower().strip("_")
    return ("get_" + text) if text and text[0].isdigit() else text


def _change_name(name: str) -> str:
    reserved = {"from", "class", "return", "false", "true", "id", "and"}
    return ("is_" + name) if name in reserved else name


def func_name_of(tool: str, api: str) -> str:
    return f"{_change_name(_standardize(api))}_for_{_standardize(tool)}"


@dataclass(frozen=True)
class RetrievalIndices:
    predicted_set: Tuple[int, ...]
    ranking: Tuple[int, ...]
    shortlist: Tuple[int, ...]
    predicted_score: float


@dataclass(frozen=True)
class RetrievalOutput:
    predicted_set: Tuple[str, ...]
    ranking: Tuple[str, ...]
    shortlist: Tuple[str, ...]
    predicted_score: float

    def as_dict(self) -> dict:
        return {
            "predicted_set": list(self.predicted_set),
            "ranking": list(self.ranking),
            "shortlist": list(self.shortlist),
            "predicted_score": self.predicted_score,
        }


class HYSETScorer(nn.Module):
    def __init__(self, num_tools: int, d_z: int, d_r: int, m_max: int = 5):
        super().__init__()
        if num_tools < 1:
            raise ValueError("num_tools must be positive")
        if d_z < 1 or d_r < 1:
            raise ValueError("embedding dimensions must be positive")
        if m_max < 1:
            raise ValueError("m_max must be positive")

        self.num_tools = num_tools
        self.d_z = d_z
        self.d_r = d_r
        self.m_max = m_max

        self.Z = nn.Embedding(num_tools, d_z)
        self.P = nn.Linear(d_z, d_r, bias=False)
        self.interactions = nn.ParameterDict(
            {str(m): nn.Parameter(torch.empty(d_z, d_z)) for m in range(2, m_max + 1)}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.Z.weight, mean=0.0, std=self.d_z ** -0.5)
        nn.init.xavier_uniform_(self.P.weight)
        for parameter in self.interactions.values():
            nn.init.xavier_uniform_(parameter)
            parameter.data.copy_(0.5 * (parameter.data + parameter.data.T))
        self.project_constraints_()

    def interaction_matrix(self, cardinality: int) -> torch.Tensor:
        if cardinality < 2 or cardinality > self.m_max:
            raise ValueError(f"cardinality must be in [2, {self.m_max}]")
        raw = self.interactions[str(cardinality)]
        return 0.5 * (raw + raw.T)

    @torch.no_grad()
    def project_constraints_(self) -> None:
        self.Z.weight.copy_(F.normalize(self.Z.weight, dim=-1))
        for parameter in self.interactions.values():
            parameter.copy_(0.5 * (parameter + parameter.T))

    def singleton_scores(self, query_embedding: torch.Tensor) -> torch.Tensor:
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.d_r:
            raise ValueError(f"query_embedding must have shape ({self.d_r},)")
        projected_tools = self.P(self.Z.weight)
        return projected_tools @ query_embedding

    def score_sets(self, query_embedding: torch.Tensor, tool_indices: torch.Tensor) -> torch.Tensor:
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.d_r:
            raise ValueError(f"query_embedding must have shape ({self.d_r},)")
        if tool_indices.ndim != 2:
            raise ValueError("tool_indices must have shape (n_sets, cardinality)")
        cardinality = tool_indices.shape[1]
        if not 1 <= cardinality <= self.m_max:
            raise ValueError(f"candidate cardinality must be in [1, {self.m_max}]")

        z = self.Z(tool_indices)
        projected = self.P(z)
        match = torch.einsum("r,nmr->nm", query_embedding, projected)
        attention = torch.softmax(match, dim=-1)
        f_align = (attention * match).sum(dim=-1)

        if cardinality == 1:
            return f_align

        matrix = self.interaction_matrix(cardinality)
        pair_scores = torch.einsum("nad,de,nbe->nab", z, matrix, z)
        pair_mask = torch.triu(
            torch.ones(cardinality, cardinality, dtype=torch.bool, device=z.device),
            diagonal=1,
        )
        f_set = pair_scores[:, pair_mask].sum(dim=-1)
        return f_set + f_align

    def score_candidate_sets(
        self,
        query_embedding: torch.Tensor,
        candidate_sets: Sequence[Sequence[int]],
        batch_size: int = 4096,
    ) -> torch.Tensor:
        if not candidate_sets:
            raise ValueError("candidate_sets must not be empty")

        grouped: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = {}
        for position, candidate in enumerate(candidate_sets):
            canonical = tuple(sorted(int(index) for index in candidate))
            if len(canonical) != len(set(canonical)):
                raise ValueError("candidate sets cannot contain duplicate tools")
            grouped.setdefault(len(canonical), []).append((position, canonical))

        values: List[torch.Tensor | None] = [None] * len(candidate_sets)
        device = query_embedding.device
        for cardinality, entries in grouped.items():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start : start + batch_size]
                indices = torch.tensor([entry[1] for entry in chunk], device=device)
                chunk_scores = self.score_sets(query_embedding, indices)
                for (position, _), score in zip(chunk, chunk_scores):
                    values[position] = score

        if any(value is None for value in values):
            raise RuntimeError("internal candidate-scoring error")
        return torch.stack([value for value in values if value is not None])

    @torch.no_grad()
    def build_shortlist(
        self,
        query_embedding: torch.Tensor,
        k1: int = 15,
        k_pool: int = 20,
    ) -> Tuple[int, ...]:
        if not self.m_max <= k_pool <= self.num_tools:
            raise ValueError("k_pool must satisfy m_max <= k_pool <= num_tools")
        if not 1 <= k1 < k_pool:
            raise ValueError("k1 must satisfy 1 <= k1 < k_pool")

        singleton = self.singleton_scores(query_embedding)
        s0 = torch.topk(singleton, k=k1).indices

        matrix = self.interaction_matrix(self.m_max)
        all_z = self.Z.weight
        s0_z = all_z[s0]
        complementarity = (all_z @ matrix) @ s0_z.T
        complementarity = complementarity.max(dim=1).values
        complementarity[s0] = -torch.inf
        expanded = torch.topk(complementarity, k=k_pool - k1).indices
        return tuple(s0.tolist() + expanded.tolist())

    @torch.no_grad()
    def predict_set(
        self,
        query_embedding: torch.Tensor,
        shortlist: Sequence[int],
        batch_size: int = 4096,
    ) -> Tuple[Tuple[int, ...], float]:
        best_set: Tuple[int, ...] = ()
        best_score = -float("inf")
        max_cardinality = min(self.m_max, len(shortlist))
        for cardinality in range(1, max_cardinality + 1):
            candidates = list(combinations(shortlist, cardinality))
            for start in range(0, len(candidates), batch_size):
                chunk = candidates[start : start + batch_size]
                indices = torch.tensor(chunk, device=query_embedding.device)
                scores = self.score_sets(query_embedding, indices)
                value, position = scores.max(dim=0)
                scalar = float(value.item())
                if scalar > best_score:
                    best_score = scalar
                    best_set = tuple(chunk[int(position.item())])
        return best_set, best_score

    @torch.no_grad()
    def greedy_ranking(
        self,
        query_embedding: torch.Tensor,
        shortlist: Sequence[int],
        length: int = 5,
    ) -> Tuple[int, ...]:
        if not 1 <= length <= self.m_max:
            raise ValueError(f"ranking length must be in [1, {self.m_max}]")
        selected: List[int] = []
        remaining = list(shortlist)
        for _ in range(min(length, len(remaining))):
            candidates = [tuple(selected + [tool]) for tool in remaining]
            scores = self.score_candidate_sets(query_embedding, candidates)
            best_position = int(scores.argmax().item())
            selected.append(remaining.pop(best_position))
        return tuple(selected)

    @torch.no_grad()
    def retrieve_from_embedding(
        self,
        query_embedding: torch.Tensor,
        k1: int = 15,
        k_pool: int = 20,
        ranking_length: int = 5,
        candidate_batch_size: int = 4096,
    ) -> RetrievalIndices:
        shortlist = self.build_shortlist(query_embedding, k1=k1, k_pool=k_pool)
        predicted_set, score = self.predict_set(
            query_embedding,
            shortlist,
            batch_size=candidate_batch_size,
        )
        ranking = self.greedy_ranking(query_embedding, shortlist, length=ranking_length)
        return RetrievalIndices(predicted_set, ranking, shortlist, score)

    def regularisation(self, weight: float) -> torch.Tensor:
        if weight < 0:
            raise ValueError("regularisation weight must be non-negative")
        total = self.Z.weight.new_zeros(())
        for cardinality in range(2, self.m_max + 1):
            total = total + self.interaction_matrix(cardinality).pow(2).sum()
        return weight * total


class HYSETModel(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        func_names: Sequence[str],
        d_z: int,
        m_max: int = 5,
        encoder_type: str = "bert",
    ):
        super().__init__()
        if encoder_type not in {"bert", "qwen"}:
            raise ValueError("encoder_type must be 'bert' or 'qwen'")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required; install requirements.txt") from exc

        self.encoder_path = encoder_path
        self.encoder_type = encoder_type
        self.func_names = list(func_names)
        self.fn_to_idx = {name: index for index, name in enumerate(self.func_names)}
        if len(self.fn_to_idx) != len(self.func_names):
            raise ValueError("func_names must be unique")

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = "auto" if encoder_type == "qwen" else None
        self.encoder_model = AutoModel.from_pretrained(encoder_path, torch_dtype=torch_dtype)
        self.d_r = int(self.encoder_model.config.hidden_size)
        if self.d_r != d_z:
            raise ValueError(
                f"paper configurations require d_z=d_r, but got d_z={d_z}, d_r={self.d_r}"
            )

        for parameter in self.encoder_model.parameters():
            parameter.requires_grad_(False)
        self.encoder_model.eval()
        self.scorer = HYSETScorer(len(self.func_names), d_z, self.d_r, m_max)

    @property
    def d_z(self) -> int:
        return self.scorer.d_z

    @property
    def m_max(self) -> int:
        return self.scorer.m_max

    @property
    def num_tools(self) -> int:
        return self.scorer.num_tools

    def train(self, mode: bool = True) -> "HYSETModel":
        super().train(mode)
        self.encoder_model.eval()
        return self

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "qwen":
            positions = attention_mask.sum(dim=1).sub(1).clamp(min=0)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[rows, positions]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_length: int = 512,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.d_r, device=next(self.parameters()).device)
        device = next(self.parameters()).device
        outputs: List[torch.Tensor] = []
        self.encoder_model.eval()
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            model_output = self.encoder_model(**encoded)
            pooled = self._pool(model_output.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.float())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def init_tool_embeddings(self, api_texts: Sequence[str], batch_size: int = 128) -> None:
        if len(api_texts) != self.num_tools:
            raise ValueError("api_texts must align one-to-one with func_names")
        embeddings = self.encode_texts(api_texts, batch_size=batch_size, max_length=128)
        if embeddings.shape[1] != self.d_z:
            raise ValueError("encoded API dimension does not match d_z")
        self.scorer.Z.weight.copy_(F.normalize(embeddings, dim=-1))

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        k1: int = 15,
        k_pool: int = 20,
        ranking_length: int = 5,
        candidate_batch_size: int = 4096,
    ) -> RetrievalOutput:
        query_embedding = self.encode_texts([query])[0]
        indices = self.scorer.retrieve_from_embedding(
            query_embedding,
            k1=k1,
            k_pool=k_pool,
            ranking_length=ranking_length,
            candidate_batch_size=candidate_batch_size,
        )
        names = self.func_names
        return RetrievalOutput(
            predicted_set=tuple(names[index] for index in indices.predicted_set),
            ranking=tuple(names[index] for index in indices.ranking),
            shortlist=tuple(names[index] for index in indices.shortlist),
            predicted_score=indices.predicted_score,
        )

    def trainable_parameters(self):
        return self.scorer.parameters()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.scorer.parameters())

    def save(self, path: str, extra: dict | None = None) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "format_version": 2,
            "scorer_state": self.scorer.state_dict(),
            "func_names": self.func_names,
            "d_z": self.d_z,
            "d_r": self.d_r,
            "m_max": self.m_max,
            "encoder_type": self.encoder_type,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, encoder_path: str, device: str = "cuda") -> "HYSETModel":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format_version") != 2:
            raise ValueError("checkpoint is not in the paper-aligned HYSET format")
        model = cls(
            encoder_path=encoder_path,
            func_names=payload["func_names"],
            d_z=int(payload["d_z"]),
            m_max=int(payload["m_max"]),
            encoder_type=payload["encoder_type"],
        )
        if model.d_r != int(payload["d_r"]):
            raise ValueError("checkpoint query dimension does not match encoder")
        model.scorer.load_state_dict(payload["scorer_state"], strict=True)
        model.scorer.project_constraints_()
        model.to(device)
        model.eval()
        return model
