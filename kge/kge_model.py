import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class TransE_filtered_negative_sampling(KGEModel):
    r"""A copy of TransE architecture (negative distance as score) with optional filtered negative sampling.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (int, optional): The margin of the ranking loss. (default: 1.0)
        p_norm (int, optional): The order embedding and distance normalization. (default: 1.0)
        sparse (bool, optional): If set to True, gradients w.r.t. embedding matrices will be sparse. (default: False)
        true_triples (Tensor, optional): Long tensor of shape [N, 3] containing all known true triples (h, r, t)
            across train/valid/test for filtered negative sampling. If None, falls back to unfiltered sampling.
        max_filter_trials (int, optional): Maximum rejection trials per corrupted triple. (default: 50)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
        true_triples: Tensor | None = None,
        max_filter_trials: int = 50,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.p_norm = p_norm
        self.margin = margin
        self.max_filter_trials = int(max_filter_trials)

        # Build a Python set for O(1) membership test if provided
        self._filtered = False
        self._true_triple_set: set[tuple[int, int, int]] = set()
        if true_triples is not None:
            tt = true_triples.detach().to('cpu').long()
            for h, r, t in tt.tolist():
                self._true_triple_set.add((int(h), int(r), int(t)))
            self._filtered = True and len(self._true_triple_set) > 0

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        # Negative TransE distance as score (higher is better)
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    def _random_sample_unfiltered(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return super().random_sample(head_index, rel_type, tail_index)

    def random_sample(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # If no filtered set, defer to default sampler
        if not self._filtered:
            return self._random_sample_unfiltered(head_index, rel_type, tail_index)

        device = head_index.device
        B = head_index.shape[0]
        neg_head = head_index.clone().detach()
        neg_tail = tail_index.clone().detach()

        # Randomly decide to corrupt head or tail per sample
        corrupt_head_mask = torch.rand(B, device=device) < 0.5

        # Rejection sampling with bounded trials
        for i in range(B):
            h = int(head_index[i].item())
            r = int(rel_type[i].item())
            t = int(tail_index[i].item())
            if corrupt_head_mask[i]:
                cand = h
                for _ in range(self.max_filter_trials):
                    cand = int(torch.randint(0, self.num_nodes, size=(1,), device=device).item())
                    if (cand, r, t) not in self._true_triple_set:
                        break
                neg_head[i] = cand
                neg_tail[i] = t
            else:
                cand = t
                for _ in range(self.max_filter_trials):
                    cand = int(torch.randint(0, self.num_nodes, size=(1,), device=device).item())
                    if (h, r, cand) not in self._true_triple_set:
                        break
                neg_head[i] = h
                neg_tail[i] = cand

        return neg_head, rel_type, neg_tail

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        ) 

# TransH: relation-specific hyperplane projection
class TransH(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 2.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.p_norm = p_norm
        self.margin = margin
        # relation-specific normal vector for hyperplane
        self.rel_norm = torch.nn.Embedding(num_relations, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_norm.weight, -bound, bound)
        F.normalize(self.rel_norm.weight.data, p=2, dim=-1, out=self.rel_norm.weight.data)

    def _proj(self, e: Tensor, w: Tensor) -> Tensor:
        # project entity embedding e onto hyperplane orthogonal to w
        # e_perp = e - <e, w> w
        return e - (e * w).sum(dim=-1, keepdim=True) * w

    def forward(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        h = self.node_emb(head_index)
        r = self.rel_emb(rel_type)
        t = self.node_emb(tail_index)
        w = F.normalize(self.rel_norm(rel_type), p=2, dim=-1)
        h = self._proj(h, w)
        t = self._proj(t, w)
        h = F.normalize(h, p=self.p_norm, dim=-1)
        t = F.normalize(t, p=self.p_norm, dim=-1)
        return -((h + r) - t).norm(p=self.p_norm, dim=-1)

    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        pos = self(head_index, rel_type, tail_index)
        neg = self(*self.random_sample(head_index, rel_type, tail_index))
        return F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos), margin=self.margin)


# TransR: relation-specific projection matrices
class TransR(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 2.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.p_norm = p_norm
        self.margin = margin
        # Each relation has a projection matrix d x d (stored flattened)
        self.rel_proj = torch.nn.Embedding(num_relations, hidden_channels * hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_proj.weight, -bound, bound)

    def _apply_proj(self, e: Tensor, M_flat: Tensor) -> Tensor:
        d = self.hidden_channels
        M = M_flat.view(-1, d, d)  # [B, d, d]
        # e: [B, d]; output: [B, d]
        return torch.bmm(M, e.unsqueeze(-1)).squeeze(-1)

    def forward(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        h = self.node_emb(head_index)
        r = self.rel_emb(rel_type)
        t = self.node_emb(tail_index)
        M = self.rel_proj(rel_type)
        h = self._apply_proj(h, M)
        t = self._apply_proj(t, M)
        h = F.normalize(h, p=self.p_norm, dim=-1)
        t = F.normalize(t, p=self.p_norm, dim=-1)
        return -((h + r) - t).norm(p=self.p_norm, dim=-1)

    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        pos = self(head_index, rel_type, tail_index)
        neg = self(*self.random_sample(head_index, rel_type, tail_index))
        return F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos), margin=self.margin)


# TransD: dynamic mapping via entity/relation projection vectors
class TransD(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 2.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.p_norm = p_norm
        self.margin = margin
        # Projection vectors
        self.ent_proj = torch.nn.Embedding(num_nodes, hidden_channels)
        self.rel_proj = torch.nn.Embedding(num_relations, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.ent_proj.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_proj.weight, -bound, bound)

    def _proj(self, e: Tensor, p_e: Tensor, p_r: Tensor) -> Tensor:
        # Dynamic mapping: M_{r,e} = I + p_r p_e^T, so M_{r,e} e = e + (e · p_e) p_r
        coef = (e * p_e).sum(dim=-1, keepdim=True)  # [B,1]
        return e + coef * p_r

    def forward(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        h = self.node_emb(head_index)
        r = self.rel_emb(rel_type)
        t = self.node_emb(tail_index)
        p_e_h = self.ent_proj(head_index)
        p_e_t = self.ent_proj(tail_index)
        p_r = self.rel_proj(rel_type)
        h = self._proj(h, p_e_h, p_r)
        t = self._proj(t, p_e_t, p_r)
        h = F.normalize(h, p=self.p_norm, dim=-1)
        t = F.normalize(t, p=self.p_norm, dim=-1)
        return -((h + r) - t).norm(p=self.p_norm, dim=-1)

    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        pos = self(head_index, rel_type, tail_index)
        neg = self(*self.random_sample(head_index, rel_type, tail_index))
        return F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos), margin=self.margin)
 