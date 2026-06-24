import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 模型文件包含：FeaturePointNet、ring_adj、build_skip_edges_for_ring 等。
from Models0203 import FeaturePointNet

# ========== 旧的数据加载与工具，保持不变 ==========

from Models0123 import (
    PolygonVertexDataset,
    polygon_collate_fn
)

from DataLoading1029 import *


# ========== 1. 工具：EMA ==========
class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        import copy
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self._init_teacher(model)

    @torch.no_grad()
    def _init_teacher(self, student: nn.Module):
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data.copy_(p_s.data)
        for b_t, b_s in zip(self.teacher.buffers(), student.buffers()):
            b_t.data.copy_(b_s.data)

    @torch.no_grad()
    def update(self, student: nn.Module):
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data = p_t.data * self.decay + p_s.data * (1.0 - self.decay)
        for b_t, b_s in zip(self.teacher.buffers(), student.buffers()):
            b_t.data = b_t.data * self.decay + b_s.data * (1.0 - self.decay)


# ========== 2. 图增广（与原版一致） ==========
class GraphAugmentor:
    def __init__(
            self,
            weak_dropedge: float = 0.1,
            weak_feat_noise: float = 0.05,
            strong_dropedge: float = 0.3,
            strong_feat_noise: float = 0.15,
            coord_jitter: float = 0.01,
            node_dropout_ratio: float = 0.0,
    ):
        self.weak_dropedge = weak_dropedge
        self.weak_feat_noise = weak_feat_noise
        self.strong_dropedge = strong_dropedge
        self.strong_feat_noise = strong_feat_noise
        self.coord_jitter = coord_jitter
        self.node_dropout_ratio = node_dropout_ratio

    def _apply_dropedge_mask(self, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        B, N = mask.shape
        new_mask = mask.clone()
        for b in range(B):
            valid_idx = torch.nonzero(new_mask[b] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            drop_num = int(valid_idx.numel() * ratio)
            if drop_num > 0:
                drop_idx = valid_idx[torch.randperm(valid_idx.numel())[:drop_num]]
                new_mask[b, drop_idx] = 0.0
        return new_mask

    def _apply_node_dropout(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.node_dropout_ratio <= 0:
            return mask
        B, N = mask.shape
        new_mask = mask.clone()
        for b in range(B):
            valid_idx = torch.nonzero(new_mask[b] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            drop_num = int(valid_idx.numel() * self.node_dropout_ratio)
            if drop_num > 0:
                drop_idx = valid_idx[torch.randperm(valid_idx.numel())[:drop_num]]
                new_mask[b, drop_idx] = 0.0
        # 将drop出的节点特征置零，避免LN统计污染
        new_mask_bool = (new_mask == 0)
        feats.masked_fill_(new_mask_bool.unsqueeze(-1), 0.0)
        return new_mask

    def weak(self, feats: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor):
        new_mask = self._apply_dropedge_mask(mask, self.weak_dropedge)
        noise = torch.randn_like(feats) * self.weak_feat_noise
        w_feats = feats + noise
        c_noise = torch.randn_like(coords) * self.coord_jitter
        w_coords = coords + c_noise
        return w_feats, w_coords, new_mask

    def strong(self, feats: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor):
        new_mask = self._apply_dropedge_mask(mask, self.strong_dropedge)
        new_mask = self._apply_node_dropout(feats, new_mask)
        noise = torch.randn_like(feats) * self.strong_feat_noise
        s_feats = feats + noise
        c_noise = torch.randn_like(coords) * (self.coord_jitter * 2.0)
        s_coords = coords + c_noise
        return s_feats, s_coords, new_mask


# ========== 3. 半监督训练器：改为调用 FeaturePointNet ==========
class SemiSupTrainer:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            ema_decay: float = 0.999,
            sup_loss_pos_weight: Optional[float] = None,
            pseudo_conf_thresh: float = 0.7,
            lambda_consistency: float = 1.0,
            lambda_entropy: float = 0.0,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            grad_clip: Optional[float] = 1.0,
            augmentor: Optional[GraphAugmentor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.ema = EMAHelper(model, decay=ema_decay)
        self.teacher = self.ema.teacher.to(device).eval()
        self.pseudo_conf_thresh = pseudo_conf_thresh
        self.lambda_consistency = lambda_consistency
        self.lambda_entropy = lambda_entropy
        self.grad_clip = grad_clip
        self.augmentor = augmentor if augmentor is not None else GraphAugmentor()

        if sup_loss_pos_weight is not None:
            pw = torch.tensor([sup_loss_pos_weight], dtype=torch.float32, device=device)
            self.sup_criterion = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        else:
            self.sup_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _mask_bce(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.sup_criterion(logits, targets)
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _consistency_loss(self, stu_logits: torch.Tensor, tea_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        stu_prob = torch.sigmoid(stu_logits)
        loss = F.mse_loss(stu_prob, tea_prob, reduction="none")
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _entropy_min(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        ent = -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))
        ent = ent * mask
        denom = mask.sum().clamp_min(1.0)
        return ent.sum() / denom

    @torch.no_grad()
    def _teacher_predict_prob(self, feats_w, coords_w, mask_w) -> torch.Tensor:
        # 教师模型：直接前向，返回 keep_prob
        out = self.model(coords_w, feats_w, edge_index=None, extra_edges=None,
                         add_ring_skip=True, ring_skip_hops=(2, 3))
        logits = out["keep_logits"]
        # 将无效节点的概率置为0.5，避免影响阈值判断
        prob = torch.sigmoid(logits)
        prob = torch.where(mask_w > 0, prob, torch.full_like(prob, 0.5))
        return prob

    def train_one_epoch(
            self,
            labeled_loader: DataLoader,
            unlabeled_loader: Optional[DataLoader] = None,
            hide_label_ratio: float = 0.0,
            epoch_idx: int = 1,
            ramp_up_epochs: int = 10,
    ) -> Dict[str, float]:
        self.model.train()
        stats = {"sup": 0.0, "cons": 0.0, "ent": 0.0, "total": 0.0}
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        ru = min(1.0, epoch_idx / float(max(1, ramp_up_epochs)))
        lam_cons = self.lambda_consistency * ru
        lam_ent = self.lambda_entropy * ru

        for batch in labeled_loader:
            feats = batch["feats"].to(self.device)  # (B,MaxN,F)
            coords = batch["coords"].to(self.device)  # (B,MaxN,2)
            mask = batch["mask"].to(self.device)  # (B,MaxN)
            keep = batch["keep"].to(self.device)  # (B,MaxN)

            # 隐藏部分有标签为无标签
            if hide_label_ratio > 0:
                valid = (mask > 0)
                hide_mask = (torch.rand_like(keep) < hide_label_ratio) & valid
                sup_mask = valid & (~hide_mask)
            else:
                sup_mask = (mask > 0)

            # 弱/强增广
            feats_w, coords_w, mask_w = self.augmentor.weak(feats.clone(), coords.clone(), mask)
            feats_s, coords_s, mask_s = self.augmentor.strong(feats.clone(), coords.clone(), mask)

            # 学生监督前向（原图）
            out_stu = self.model(coords, feats, edge_index=None, extra_edges=None,
                                 add_ring_skip=True, ring_skip_hops=(2, 3))
            logits_stu = out_stu["keep_logits"]  # (B,N)
            sup_loss = self._mask_bce(logits_stu, keep, sup_mask)

            # 教师软标签（弱增广）
            with torch.no_grad():
                tea_prob = self._teacher_predict_prob(feats_w, coords_w, mask_w)  # (B,N)

            # 置信度筛选
            conf = torch.abs(tea_prob - 0.5) * 2.0  # [0,1]
            pseudo_mask = ((conf >= self.pseudo_conf_thresh).float() * mask_s).detach()

            # 学生一致性（强增广）
            stu_logits_s = self.model(coords_s, feats_s, edge_index=None, extra_edges=None,
                                      add_ring_skip=True, ring_skip_hops=(2, 3))["keep_logits"]
            cons_loss = self._consistency_loss(stu_logits_s, tea_prob.detach(), pseudo_mask)

            ent_loss = self._entropy_min(logits_stu, mask - sup_mask) if lam_ent > 0 else torch.tensor(0.0,
                                                                                                       device=self.device)

            total_loss = sup_loss + lam_cons * cons_loss + lam_ent * ent_loss

            self.opt.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.ema.update(self.model)

            stats["sup"] += float(sup_loss.item())
            stats["cons"] += float(cons_loss.item())
            stats["ent"] += float(ent_loss.item()) if lam_ent > 0 else 0.0
            stats["total"] += float(total_loss.item())
            n_steps += 1

            # 纯无标签 batch（可选）
            if unlabeled_iter is not None:
                try:
                    u_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    u_batch = next(unlabeled_iter)

                u_feats = u_batch["feats"].to(self.device)
                u_coords = u_batch["coords"].to(self.device)
                u_mask = u_batch["mask"].to(self.device)

                u_feats_w, u_coords_w, u_mask_w = self.augmentor.weak(u_feats.clone(), u_coords.clone(), u_mask)
                u_feats_s, u_coords_s, u_mask_s = self.augmentor.strong(u_feats.clone(), u_coords.clone(), u_mask)

                with torch.no_grad():
                    u_tea_prob = self._teacher_predict_prob(u_feats_w, u_coords_w, u_mask_w)

                u_conf = torch.abs(u_tea_prob - 0.5) * 2.0
                u_pseudo_mask = ((u_conf >= self.pseudo_conf_thresh).float() * u_mask_s).detach()

                u_stu_logits = self.model(u_coords_s, u_feats_s, edge_index=None, extra_edges=None,
                                          add_ring_skip=True, ring_skip_hops=(2, 3))["keep_logits"]
                u_cons_loss = self._consistency_loss(u_stu_logits, u_tea_prob.detach(), u_pseudo_mask)

                self.opt.zero_grad()
                (lam_cons * u_cons_loss).backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.ema.update(self.model)

        for k in stats:
            stats[k] = stats[k] / max(1, n_steps)
        return stats

    def train_one_epoch_new(
            self,
            unlabeled_loader: DataLoader,
            epoch_idx: int = 1,
            ramp_up_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        纯无标签训练（keep == -1），仅使用一致性 + （可选）熵最小化。
        - 教师：EMA 模型在弱增广视图产生软标签概率
        - 学生：在强增广视图与教师概率对齐（高置信筛选）
        - 无监督熵最小化：对未参与一致性的位置，促使低熵预测（可选）
        """
        self.model.train()
        stats = {"sup": 0.0, "cons": 0.0, "ent": 0.0, "total": 0.0}
        n_steps = 0

        # ramp-up 因子：逐步增强无监督项的权重
        ru = min(1.0, epoch_idx / float(max(1, ramp_up_epochs)))
        lam_cons = self.lambda_consistency * ru
        lam_ent = self.lambda_entropy * ru

        for u_batch in unlabeled_loader:
            # 无标签 batch：只用 feats/coords/mask
            u_feats = u_batch["feats"].to(self.device)  # (B, N, F)
            u_coords = u_batch["coords"].to(self.device)  # (B, N, 2)
            u_mask = u_batch["mask"].to(self.device)  # (B, N)

            # 生成弱/强增广
            u_feats_w, u_coords_w, u_mask_w = self.augmentor.weak(u_feats.clone(), u_coords.clone(), u_mask)
            u_feats_s, u_coords_s, u_mask_s = self.augmentor.strong(u_feats.clone(), u_coords.clone(), u_mask)

            # 教师在弱增广上的软标签概率
            with torch.no_grad():
                u_tea_prob = self._teacher_predict_prob(u_feats_w, u_coords_w, u_mask_w)  # (B, N) in [0,1]

            # 置信度筛选：越靠近0/1越高，靠近0.5越低
            u_conf = torch.abs(u_tea_prob - 0.5) * 2.0  # [0,1]
            u_pseudo_mask = ((u_conf >= self.pseudo_conf_thresh).float() * u_mask_s).detach()

            # 学生在强增广视图的一致性损失
            u_stu_logits = self.model(
                u_coords_s, u_feats_s,
                edge_index=None, extra_edges=None,
                add_ring_skip=True, ring_skip_hops=(2, 3)
            )["keep_logits"]  # (B, N)

            u_cons_loss = self._consistency_loss(u_stu_logits, u_tea_prob.detach(), u_pseudo_mask)

            # 可选：对未参与一致性的有效位置做熵最小化（低熵更果断）
            if lam_ent > 0:
                # 有效但未被选为伪标签的位置
                ent_mask = (u_mask_s > 0).float() * (1.0 - u_pseudo_mask)
                u_ent_loss = self._entropy_min(u_stu_logits, ent_mask)
            else:
                u_ent_loss = torch.tensor(0.0, device=self.device)

            # 总损失（纯无监督，无监督项用 ramp-up）
            total_loss = lam_cons * u_cons_loss + lam_ent * u_ent_loss

            # 梯度更新
            self.opt.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.ema.update(self.model)

            # 统计
            stats["sup"] += 0.0  # 无监督阶段无 sup
            stats["cons"] += float(u_cons_loss.item())
            stats["ent"] += float(u_ent_loss.item()) if lam_ent > 0 else 0.0
            stats["total"] += float(total_loss.item())
            n_steps += 1

        for k in stats:
            stats[k] = stats[k] / max(1, n_steps)
        return stats

    def train_one_epoch_semisuper(
            self,
            labeled_loader: DataLoader,
            unlabeled_loader: Optional[DataLoader] = None,
            hide_label_ratio: float = 0.0,
            epoch_idx: int = 1,
            ramp_up_epochs: int = 10,
    ) -> Dict[str, float]:
        '''
        半监督训练，有标签的0-599，无标签的0-7322
        :param labeled_loader:
        :param unlabeled_loader:
        :param hide_label_ratio:
        :param epoch_idx:
        :param ramp_up_epochs:
        :return:
        '''
        model = self.model
        device = self.device
        model.train()
        stats = {"sup": 0.0, "cons": 0.0, "ent": 0.0, "total": 0.0}
        n_steps = 0

        # ramp-up
        ru = min(1.0, epoch_idx / float(max(1, ramp_up_epochs)))
        lam_cons = self.lambda_consistency * ru
        lam_ent = self.lambda_entropy * ru
        aug = self.augmentor
        grad_clip_val = self.grad_clip
        pseudo_th = self.pseudo_conf_thresh
        optimizer = self.opt

        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        for batch in labeled_loader:
            feats = batch["feats"].to(device)
            coords = batch["coords"].to(device)
            mask = batch["mask"].to(device)
            keep = batch["keep"].to(device)

            # 隐藏部分标签
            if hide_label_ratio > 0:
                valid = (mask > 0)
                hide_mask = (torch.rand_like(keep, dtype=keep.dtype) < hide_label_ratio) & valid
                sup_mask = valid & (~hide_mask)
            else:
                sup_mask = (mask > 0)

            # 弱/强增强
            feats_w, coords_w, mask_w = aug.weak(feats.clone(), coords.clone(), mask.clone())
            feats_s, coords_s, mask_s = aug.strong(feats.clone(), coords.clone(), mask.clone())

            # 监督损失
            out_stu = model(coords, feats, edge_index=None, extra_edges=None,
                            add_ring_skip=True, ring_skip_hops=(2, 3))
            if "keep_logits" in out_stu:
                logits_stu = out_stu["keep_logits"]
            elif "keep_prob" in out_stu:
                p = out_stu["keep_prob"].clamp(1e-6, 1 - 1e-6)
                logits_stu = torch.log(p / (1 - p))
            else:
                raise KeyError("模型输出中既没有'keep_logits'也没有'keep_prob'")
            sup_loss = self._mask_bce(logits_stu, keep, sup_mask)

            # 教师软标签（弱增广）
            with torch.no_grad():
                tea_prob = self._teacher_predict_prob(feats_w, coords_w, mask_w)

            # FixMatch 置信筛选
            conf = torch.abs(tea_prob - 0.5) * 2.0
            pseudo_mask = ((conf >= pseudo_th).to(mask_s.dtype) * mask_s).detach()

            # 一致性损失（学生在强增广）
            out_s = model(coords=coords_s, feats=feats_s, edge_index=None, mask=mask_s)
            if "keep_logits" in out_s:
                stu_logits_s = out_s["keep_logits"]
            elif "keep_prob" in out_s:
                p_s = out_s["keep_prob"].clamp(1e-6, 1 - 1e-6)
                stu_logits_s = torch.log(p_s / (1 - p_s))
            else:
                raise KeyError("模型输出(强增广)中既没有'keep_logits'也没有'keep_prob'")
            cons_loss = self._consistency_loss(stu_logits_s, tea_prob.detach(), pseudo_mask)

            # 熵最小化（对被隐藏/无监督位置）
            if lam_ent > 0:
                unsup_mask = (mask > 0) & (~sup_mask)
                ent_loss = self._entropy_min(logits_stu, unsup_mask.to(logits_stu.dtype))
            else:
                ent_loss = torch.tensor(0.0, device=device)

            total_loss = sup_loss + lam_cons * cons_loss + lam_ent * ent_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()
            if hasattr(self, "ema") and self.ema is not None:
                self.ema.update(self.model)

            stats["sup"] += float(sup_loss.item())
            stats["cons"] += float(cons_loss.item())
            stats["ent"] += float(ent_loss.item()) if lam_ent > 0 else 0.0
            stats["total"] += float(total_loss.item())
            n_steps += 1

            # 纯无标签批次（可选）
            if unlabeled_iter is not None:
                try:
                    u_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    u_batch = next(unlabeled_iter)

                u_feats = u_batch["feats"].to(device)
                u_coords = u_batch["coords"].to(device)
                u_mask = u_batch["mask"].to(device)

                u_feats_w, u_coords_w, u_mask_w = aug.weak(u_feats.clone(), u_coords.clone(), u_mask.clone())
                u_feats_s, u_coords_s, u_mask_s = aug.strong(u_feats.clone(), u_coords.clone(), u_mask.clone())

                with torch.no_grad():
                    u_tea_prob = self._teacher_predict_prob(u_feats_w, u_coords_w, u_mask_w)

                u_conf = torch.abs(u_tea_prob - 0.5) * 2.0
                u_pseudo_mask = ((u_conf >= pseudo_th).to(u_mask_s.dtype) * u_mask_s).detach()

                u_out_s = model(coords=u_coords_s, feats=u_feats_s, mask=u_mask_s)
                if "keep_logits" in u_out_s:
                    u_stu_logits = u_out_s["keep_logits"]
                elif "keep_prob" in u_out_s:
                    p_us = u_out_s["keep_prob"].clamp(1e-6, 1 - 1e-6)
                    u_stu_logits = torch.log(p_us / (1 - p_us))
                else:
                    raise KeyError("模型输出(无标签强增广)中既没有'keep_logits'也没有'keep_prob'")

                u_cons_loss = self._consistency_loss(u_stu_logits, u_tea_prob.detach(), u_pseudo_mask)

                optimizer.zero_grad(set_to_none=True)
                (lam_cons * u_cons_loss).backward()
                if grad_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                optimizer.step()
                if hasattr(self, "ema") and self.ema is not None:
                    self.ema.update(self.model)

        for k in stats:
            stats[k] = stats[k] / max(1, n_steps)
        return stats

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        self.model.eval()
        TP = TN = FP = FN = 0
        for batch in data_loader:
            feats = batch["feats"].to(self.device)
            coords = batch["coords"].to(self.device)
            mask = batch["mask"].to(self.device)
            keep = batch["keep"].to(self.device)

            out = self.model(coords, feats, edge_index=None, extra_edges=None,
                             add_ring_skip=True, ring_skip_hops=(2, 3))
            logits = out["keep_logits"]
            prob = torch.sigmoid(logits)
            pred = (prob >= threshold).float()

            mask_bool = (mask > 0)
            TP += ((pred == 1) & (keep == 1) & mask_bool).sum().item()
            TN += ((pred == 0) & (keep == 0) & mask_bool).sum().item()
            FP += ((pred == 1) & (keep == 0) & mask_bool).sum().item()
            FN += ((pred == 0) & (keep == 1) & mask_bool).sum().item()

        eps = 1e-9
        acc = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def fix_slashes(exp_root: str) -> str:
    s = os.path.expandvars(os.path.expanduser(exp_root.strip()))
    # 把 / 和 \ 都映射成当前系统分隔符
    s = s.translate(str.maketrans({"/": os.sep, "\\": os.sep}))
    # 规范化，Windows 上会统一成 \
    s = os.path.normpath(s)
    return s


from typing import Optional, List, Dict, Any
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import torch


def generate_chats_for_building_pts(
        polys: List[Dict[str, Any]],
        out_geojson: Optional[str] = None,
        crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    根据 load_training_polygons_and_points_no_poly_0329 返回的 polys，
    输出建筑物轮廓点的特征 GeoDataFrame / GeoJSON。

    参数
    ----------
    polys : List[Dict[str, Any]]
        每个元素格式类似：
        {
            "building_id": int/str,
            "coords": Tensor (N,2),
            "feats": Tensor (N,20) 或 ndarray (N,20),
            "keep": Tensor (N,)
        }

    out_geojson : Optional[str]
        若不为 None，则保存为 GeoJSON 文件

    crs : Optional[str]
        输出 GeoDataFrame 的坐标系，例如 "EPSG:3857" 或 "EPSG:4326"

    返回
    -------
    gdf_out : geopandas.GeoDataFrame
        字段为：
        building_id, idx, keep, attri1, ..., attri20, geometry
    """
    rows = []

    for poly in polys:
        building_id = poly["building_id"]
        coords = poly["coords"]
        feats = poly["feats"]
        keep = poly["keep"]

        # 转成 numpy
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        else:
            coords = np.asarray(coords)

        if isinstance(keep, torch.Tensor):
            keep = keep.detach().cpu().numpy()
        else:
            keep = np.asarray(keep)

        if feats is None:
            raise ValueError(f"building_id={building_id} 的 feats 为 None，无法输出 attri1~attri20。")

        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        else:
            feats = np.asarray(feats)

        # 基本检查
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"building_id={building_id} 的 coords 形状应为 (N,2)，实际为 {coords.shape}")

        if feats.ndim != 2:
            raise ValueError(f"building_id={building_id} 的 feats 应为二维数组，实际为 {feats.shape}")

        n_pts = coords.shape[0]
        feat_dim = feats.shape[1]

        if feat_dim != 20:
            raise ValueError(f"building_id={building_id} 的特征维度不是20，而是 {feat_dim}")

        if len(keep) != n_pts:
            raise ValueError(f"building_id={building_id} 的 keep 长度与 coords 点数不一致")

        if feats.shape[0] != n_pts:
            raise ValueError(f"building_id={building_id} 的 feats 行数与 coords 点数不一致")

        # 逐点写入
        for i in range(n_pts):
            row = {
                "building_id": building_id,
                "idx": i,
                "keep": float(keep[i]),
                "geometry": Point(float(coords[i, 0]), float(coords[i, 1]))
            }

            for j in range(20):
                row[f"attri{j + 1}"] = float(feats[i, j])

            rows.append(row)

    gdf_out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    # 调整字段顺序
    attr_cols = [f"attri{i}" for i in range(1, 21)]
    gdf_out = gdf_out[["building_id", "idx", "keep"] + attr_cols + ["geometry"]]

    if out_geojson is not None:
        out_dir = os.path.dirname(out_geojson)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        gdf_out.to_file(out_geojson, driver="GeoJSON")

    return gdf_out


def main_train2(
        points_geojson: str,
        points_geojson2: str,
        in_dim: int = 20,
        batch_size: int = 4,
        num_workers: int = 0,
        # 半监督设置
        hide_label_ratio: float = 0.5,
        use_extra_unlabeled: bool = True,
        # 训练超参
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.999,
        pseudo_conf_thresh: float = 0.7,
        lambda_consistency: float = 1.0,
        lambda_entropy: float = 0.0,
        sup_pos_weight: Optional[float] = None,
        grad_clip: Optional[float] = 1.0,
        ramp_up_epochs: int = 10,
        # 模型结构参数
        d_model: int = 256,
        depth: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gcn: bool = True,
        use_gat: bool = True,
        use_edgeconv: bool = True,
        local_heads_ratio: float = 0.5,
        local_win: int = 8,
        local_dist: Optional[float] = None,
        # 增广强度
        weak_dropedge: float = 0.1,
        weak_feat_noise: float = 0.05,
        strong_dropedge: float = 0.3,
        strong_feat_noise: float = 0.15,
        coord_jitter: float = 0.01,
        node_dropout_ratio: float = 0.0,
        # 交叉验证参数
        k_folds: int = 5,
        shuffle: bool = True,
        random_seed: int = 42,
        sigmas=[1.0, 2.0, 3.0],
        # 新增：输出控制
        out_dir: str = None,  # 根输出目录
        exp_name: Optional[str] = None  # 实验名（可选）
):
    # ========= 新增：输出目录准备 =========
    if out_dir is None:
        # 默认放在项目根下的 runs/checkpoints
        project_root = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(project_root, "runs", "checkpoints")
    if exp_name is None:
        # 自动生成实验名：exp_YYYYmmdd_HHMMSS
        exp_name = time.strftime("exp_%Y%m%d_%H%M%S", time.localtime())

    exp_root = os.path.join(out_dir, exp_name)  # 统一的实验输出根目录
    exp_root = fix_slashes(exp_root)
    os.makedirs(exp_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[Output] experiment root: {exp_root}")

    # 1) 加载标注数据（按多边形为单位）
    polys, building_ids, norm_params_list = load_training_polygons_and_points_no_poly_0329(
        points_geojson=points_geojson, sigmas=sigmas, w_multi_scale=False
    )
    total = len(polys)
    assert total > 0, "No labeled polygons loaded."

    gdf_feats = generate_chats_for_building_pts(
        polys,
        out_geojson=r"F:\LandUseBoudary\code\data\train_building_pts_with_feats.geojson",
        crs="EPSG:32648"  # 按你的数据实际坐标系填写
    )

    # 2) 可重复打乱索引，准备K折切分
    indices = list(range(total))
    if shuffle:
        import random
        rnd = random.Random(random_seed)
        rnd.shuffle(indices)

    # 3) 准备无标注数据的 DataLoader（各折共享）
    unlabeled_loader = None
    if use_extra_unlabeled:
        polys2, building_ids2, norm_params_list2 = load_training_polygons_and_points_no_poly_0329(
            points_geojson=points_geojson2, sigmas=sigmas, w_multi_scale=False
        )
        train_ds2 = PolygonVertexDataset(polys2, norm_params_list2, building_ids2, in_dim=in_dim)
        unlabeled_loader = DataLoader(
            train_ds2, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=polygon_collate_fn
        )

    # 4) K折主循环
    fold_metrics = []
    best_overall_f1 = -1.0
    best_overall_state = None
    best_overall_tag = None

    # 按折切分索引
    fold_sizes = [total // k_folds] * k_folds
    for i in range(total % k_folds):
        fold_sizes[i] += 1
    # 生成每折的(val_start, val_end)
    current = 0
    folds = []
    for fs in fold_sizes:
        folds.append((current, current + fs))
        current += fs

    print(f"Training on {k_folds} folds begin!")
    for fold_id, (start, end) in enumerate(folds, 1):
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]

        # 切分数据
        train_polys = [polys[i] for i in train_idx]
        val_polys = [polys[i] for i in val_idx]
        train_building_ids = [building_ids[i] for i in train_idx]
        val_building_ids = [building_ids[i] for i in val_idx]
        train_norm_params = [norm_params_list[i] for i in train_idx]
        val_norm_params = [norm_params_list[i] for i in val_idx]

        # DataLoader
        train_ds = PolygonVertexDataset(train_polys, train_norm_params, train_building_ids, in_dim=in_dim)
        val_ds = PolygonVertexDataset(val_polys, val_norm_params, val_building_ids, in_dim=in_dim)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=polygon_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=polygon_collate_fn)

        # 每折新建模型与训练器
        model = FeaturePointNet(
            in_dim=in_dim,
            d_model=d_model,
            heads=n_heads,
            depth=depth,
            use_gcn=use_gcn,
            use_gat=use_gat,
            use_edgeconv=use_edgeconv,
            local_heads_ratio=local_heads_ratio,
            local_win=local_win,
            local_dist=local_dist,
            dropout=dropout
        ).to(device)

        augmentor = GraphAugmentor(
            weak_dropedge=weak_dropedge,
            weak_feat_noise=weak_feat_noise,
            strong_dropedge=strong_dropedge,
            strong_feat_noise=strong_feat_noise,
            coord_jitter=coord_jitter,
            node_dropout_ratio=node_dropout_ratio
        )

        trainer = SemiSupTrainer(
            model=model,
            device=device,
            ema_decay=ema_decay,
            sup_loss_pos_weight=sup_pos_weight,
            pseudo_conf_thresh=pseudo_conf_thresh,
            lambda_consistency=lambda_consistency,
            lambda_entropy=lambda_entropy,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            augmentor=augmentor
        )

        best_f1 = -1.0
        best_state = None

        print(f"====== Fold {fold_id}/{k_folds}: train={len(train_ds)} | val={len(val_ds)} ======")
        for ep in range(1, epochs + 1):
            tr_stats = trainer.train_one_epoch_semisuper(
                labeled_loader=train_loader,
                unlabeled_loader=unlabeled_loader
            )
            val_metrics = trainer.evaluate(val_loader, threshold=0.5)
            print(f"[Fold {fold_id}] [Epoch {ep}/{epochs}] "
                  f"sup={tr_stats['sup']:.4f} cons={tr_stats['cons']:.4f} ent={tr_stats['ent']:.4f} total={tr_stats['total']:.4f} | "
                  f"VAL acc={val_metrics['acc']:.4f} prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}")

            # 早停/模型选择：以 F1 为准
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        fold_metrics.append(best_f1)

        # ========= 修改：保存每折最优到指定输出目录 =========
        ckpt_root = os.path.join(exp_root, f"fold_{fold_id}")
        os.makedirs(ckpt_root, exist_ok=True)
        save_path = os.path.join(ckpt_root, "feature_point_net_semisup_best.pth")
        if best_state is not None:
            torch.save(best_state, save_path)
            print(f"[Fold {fold_id}] Best model saved to {save_path} with F1={best_f1:.4f}")

        # 记录全局最佳折
        if best_f1 > best_overall_f1 and best_state is not None:
            best_overall_f1 = best_f1
            best_overall_state = best_state
            best_overall_tag = f"fold_{fold_id}"

    # 5) 折间指标汇总与总体最优保存

    fold_metrics_np = np.array(fold_metrics, dtype=float)
    mean_f1 = float(fold_metrics_np.mean()) if len(fold_metrics_np) > 0 else float("nan")
    std_f1 = float(fold_metrics_np.std(ddof=0)) if len(fold_metrics_np) > 0 else float("nan")
    print(f"K-Fold results: F1 mean={mean_f1:.4f} std={std_f1:.4f}")

    # ========= 修改：保存“全局最佳折”的权重到指定输出目录 =========
    final_path = os.path.join(exp_root, "feature_point_net_semisup_best_overall.pth")
    if best_overall_state is not None:
        torch.save(best_overall_state, final_path)
        print(f"Overall best ({best_overall_tag}) model saved to {final_path} with F1={best_overall_f1:.4f}")
    else:
        print("No improvement across folds, not saving overall best.")


import argparse
import json
import os
import time

try:
    import yaml  # pip install pyyaml

    HAS_YAML = True
except Exception:
    HAS_YAML = False


def parse_bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ("1", "true", "yes", "y", "t", "on"):
        return True
    if x in ("0", "false", "no", "n", "f", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {x}")


def load_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    _, ext = os.path.splitext(path)
    with open(path, "r", encoding="utf-8") as f:
        if ext.lower() in (".yml", ".yaml"):
            if not HAS_YAML:
                raise RuntimeError("pyyaml 未安装，无法解析 YAML 配置文件。请先 pip install pyyaml。")
            return yaml.safe_load(f) or {}
        elif ext.lower() == ".json":
            return json.load(f) or {}
        else:
            raise ValueError(f"不支持的配置文件类型：{ext}（仅支持 .yml/.yaml/.json）")


def get_args():
    parser = argparse.ArgumentParser(description="FeaturePointNet 消融实验入口（单脚本 + 配置文件）")

    # 基础必需路径,去掉required=True
    parser.add_argument("--points_geojson", type=str, default=None, help="有标注训练点集（geojson）")
    parser.add_argument("--points_geojson2", type=str, default=None, help="无标注点集（geojson）用于半监督")
    #
    # 可选配置文件（YAML/JSON），命令行参数将覆盖配置文件中的同名项
    parser.add_argument("--config", type=str, default=None, help="可选：配置文件路径（.yml/.yaml/.json）")

    # 消融参数（本次重点）
    parser.add_argument("--use_gcn", type=parse_bool, default=None, help="是否启用 GCN（True/False）")
    parser.add_argument("--use_gat", type=parse_bool, default=None, help="是否启用 GAT（True/False）")
    parser.add_argument("--use_edgeconv", type=parse_bool, default=None, help="是否启用 EdgeConv（True/False）")
    parser.add_argument("--local_heads_ratio", type=float, default=None, help="局部头比例（0~1）")
    parser.add_argument("--local_win", type=int, default=None, help="局部窗口大小（顶点序号邻域）")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None,
                        help="高斯平滑/尺度列表，如 --sigmas 1.0 2.0 3.0")

    # 常用训练参数（可按需继续加；不传则走 main_train2 默认）
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--pseudo_conf_thresh", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--lambda_entropy", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--k_folds", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)

    # 运行命名与输出（可用于组织实验）
    parser.add_argument("--exp_name", type=str, default=None, help="实验名（用于日志/输出目录命名等）")

    return parser.parse_args()


def merge_cfg_with_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """
    合并配置文件 cfg 与命令行参数 args：
    - 先用 cfg 作为基础
    - 对于 args 中非 None 的值，覆盖 cfg
    - 返回用于 main_train2 的参数字典
    """
    params = dict(cfg)  # 先用 yml 填底
    for k, v in vars(args).items():
        if k == 'config':
            continue
        # 仅当命令行显式提供了值（非 None）时覆盖 yml
        if v is not None:
            params[k] = v
    return params


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.config) if args.config else {}

    # 合并配置（命令行覆盖配置文件）
    params = merge_cfg_with_cli(cfg, args)

    # 打印最终生效配置（便于复现实验）
    print("==== Effective Params ====")
    for kk in sorted(params.keys()):
        print(f"{kk}: {params[kk]}")

    # 启动训练
    start = time.perf_counter()
    main_train2(**params)
    end = time.perf_counter()

    print(f"Training took {end - start:.2f}s")
    print("end")
