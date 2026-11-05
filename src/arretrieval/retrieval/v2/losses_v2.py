"""MNRL + in-batch listwise distillation from a frozen teacher (bge-m3).

Every example carries the teacher's embeddings of its query, positive and negatives (the `label` column, shape
(1 + 1 + n_neg) * D).  Inside a batch the teacher defines a soft ranking of ALL candidate passages (every positive and
negative of every example) for each query; the student is trained to reproduce that distribution with a KL term, next to the
usual hard-label InfoNCE term:

    loss = w_ce * CE(student logits, gold)  +  w_kd * KL( softmax(t_scale * cos_T) || softmax(scale * cos_S) )

The teacher's scores for the cross-lingual pairs (Arabic query vs English candidates) are what carries bge-m3's
cross-lingual alignment into the 118M student; the teacher also softens false negatives (it can rate a mined "negative"
as good as the gold, in which case the KL term does not push it away).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sentence_transformers.base.losses.merged_forward import embed_columns
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss


class DistillMNRL(MultipleNegativesRankingLoss):
    """MNRL + optional in-batch KL towards a frozen teacher (w_kd) + optional embedding-preservation term towards the same teacher (w_pres).

    label layout per example: [teacher q | teacher pos | teacher neg_1 ... | kd_weight | pres_weight]  (the two weights are 0/1 masks set by train_v2).
    preservation: 1 - cos(student positive embedding, teacher positive embedding), averaged over the examples whose pres mask is 1 (Arabic positives):
    it keeps the Arabic passage geometry of the frozen base model (requires student dim == teacher dim, e.g. bge-m3 -> v2-m3)."""

    def __init__(
        self, model, scale: float = 20.0, w_ce: float = 1.0, w_kd: float = 1.0, t_scale: float = 20.0, dim: int = 1024, w_pres: float = 0.0
    ):
        super().__init__(model, scale=scale)
        self.w_ce, self.w_kd, self.t_scale, self.dim, self.w_pres = w_ce, w_kd, t_scale, dim, w_pres

    def forward(self, sentence_features, labels):
        emb = embed_columns(self.model, sentence_features)
        q, docs = emb[0], torch.cat(emb[1:], dim=0)  # docs order: all positives, all negative_1, ...
        b = q.size(0)
        s = self.scale * F.normalize(q, dim=-1) @ F.normalize(docs, dim=-1).T
        loss = s.new_zeros(())
        if self.w_ce > 0:
            loss = loss + self.w_ce * F.cross_entropy(s, torch.arange(b, device=s.device))
        lab = labels.to(s.device, torch.float32)
        w_kd_vec, w_pres_vec, t = lab[:, -2], lab[:, -1], lab[:, :-2].reshape(b, -1, self.dim)
        if self.w_kd > 0:
            tq, td = t[:, 0], t[:, 1:].transpose(0, 1).reshape(-1, self.dim)  # td order matches docs
            kd_each = F.kl_div(F.log_softmax(s, dim=1), F.softmax(self.t_scale * tq @ td.T, dim=1), reduction="none").sum(1)
            loss = loss + self.w_kd * (kd_each * w_kd_vec).sum() / b
        if self.w_pres > 0:
            t_pos = t[:, 0] if t.size(1) == 1 else t[:, 1]  # preservation-only labels carry just the positive
            cos = (F.normalize(emb[1], dim=-1) * F.normalize(t_pos, dim=-1)).sum(-1)
            loss = loss + self.w_pres * ((1 - cos) * w_pres_vec).sum() / b
        return loss

    def get_config_dict(self):
        return {**super().get_config_dict(), "w_ce": self.w_ce, "w_kd": self.w_kd, "t_scale": self.t_scale, "w_pres": self.w_pres}
