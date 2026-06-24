# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 可选：PyTorch Geometric ==========
try:
    from torch_geometric.nn import GCNConv, GATConv, EdgeConv
except Exception:
    GCNConv = GATConv = EdgeConv = None


# =========================================================
# 基础几何与构图工具
# =========================================================
def ring_adj(n: int, device=None) -> torch.Tensor:
    """
    生成环形邻接边（双向）：(2, 2N)
    """
    idx = torch.arange(n, device=device)
    src = idx
    dst = (idx + 1) % n
    e1 = torch.stack([src, dst], dim=0)
    e2 = torch.stack([dst, src], dim=0)
    return torch.cat([e1, e2], dim=1)


def build_skip_edges_for_ring(N: int, hops=(2, 3), device=None) -> torch.Tensor:
    """
    在环上添加跳边（2-hop/3-hop等），构建稀疏远邻连接。
    返回 (2, E_skip)，包含双向边。
    """
    src_list = []
    dst_list = []
    for i in range(N):
        for h in hops:
            j1 = (i + h) % N
            j2 = (i - h) % N
            src_list += [i, i]
            dst_list += [j1, j2]
    src = torch.tensor(src_list, device=device, dtype=torch.long)
    dst = torch.tensor(dst_list, device=device, dtype=torch.long)
    e = torch.stack([src, dst], dim=0)
    e_rev = torch.stack([dst, src], dim=0)
    edge_index_skip = torch.cat([e, e_rev], dim=1)
    return edge_index_skip


def build_relative_geo(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    coords: (B,N,2) -> 返回 dx, dy, dist 形状 (B,N,N)
    """
    B, N, _ = coords.shape
    xi = coords.unsqueeze(2)  # (B,N,1,2)
    xj = coords.unsqueeze(1)  # (B,1,N,2)
    diff = xj - xi
    dx = diff[..., 0]
    dy = diff[..., 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
    return dx, dy, dist


def positional_s_over_L(N: int, device='cuda', feats_len_ratio: torch.Tensor = None, B: int = None) -> torch.Tensor:
    """
    相对弧长位置编码 s/L（等距近似或基于边长比率精确计算），返回 (B,N)

    参数
    - N: 顶点数
    - device: 返回张量所在设备
    - feats_len_ratio:
        可选，表示每条边（i->i+1）的长度占周长的比率
        支持形状：
          (N,)       -> 单样本，将返回 (1,N)
          (B,N)      -> 批量样本，将返回 (B,N)
        若传入 None，则使用等距近似 i/N，并需要提供 B 指定批大小
    - B: 批大小。当 feats_len_ratio 为 None 时必须提供；当 feats_len_ratio 形状为 (B,N) 时可忽略。

    约定
    - s[b, i] = sum_{k=0}^{i-1} len_ratio[b, k]，s[b, 0] = 0。
      注意不包含最后一段 (N-1->0)，因此 s[b, N-1] = sum_{k=0}^{N-2} len_ratio[b, k]。

    返回
    - s_rel: (B,N)，范围 [0, 1)
    """
    if feats_len_ratio is None:
        assert B is not None and B > 0, "feats_len_ratio 为 None 时必须提供 B（批大小）"
        idx = torch.arange(N, device=device, dtype=torch.float32) / float(N)  # (N,)
        s_rel = idx.unsqueeze(0).expand(B, N).contiguous()  # (B,N)
        # 数值保护
        s_rel = torch.clamp(s_rel, min=0.0, max=1.0 - 1e-6)
        return s_rel

    # 有提供 feats_len_ratio
    feats_len_ratio = feats_len_ratio.to(device=device, dtype=torch.float32)
    if feats_len_ratio.dim() == 1:
        # (N,) -> (1,N)
        assert feats_len_ratio.numel() == N, "feats_len_ratio 长度需为 N"
        feats_len_ratio = feats_len_ratio.unsqueeze(0)  # (1,N)
    elif feats_len_ratio.dim() == 2:
        # (B,N)
        assert feats_len_ratio.size(1) == N, "feats_len_ratio 的第二维需为 N"
    else:
        raise ValueError("feats_len_ratio 期望形状为 (N,) 或 (B,N)")

    B_eff = feats_len_ratio.size(0) if B is None else B
    if B is not None and B != feats_len_ratio.size(0):
        # 若外部传了 B，但与 feats_len_ratio 的批大小不一致，做一次检查提示
        raise ValueError(f"B={B} 与 feats_len_ratio 的批大小 {feats_len_ratio.size(0)} 不一致")

    # 计算左累积：s[b,0]=0; s[b,i]=sum_{k=0}^{i-1} len_ratio[b,k]
    s_rel = feats_len_ratio.new_zeros((feats_len_ratio.size(0), N))  # (B,N)
    if N > 1:
        s_rel[:, 1:] = torch.cumsum(feats_len_ratio[:, :-1], dim=1)

    # 数值保护到 [0, 1)
    s_rel = torch.clamp(s_rel, min=0.0, max=1.0 - 1e-6)
    return s_rel


# =========================================================
# 多尺度特征（数据阶段实现的接口占位）
# =========================================================
def build_multi_scale_feats(base_feats: torch.Tensor,
                            coords: torch.Tensor,
                            sigmas=(1.0, 2.0, 3.0)) -> torch.Tensor:
    """
    在数据预处理阶段实现：
    - 对 coords 在 σ ∈ {σ1, σ2, σ3} 做高斯平滑（环形边界）
    - 分别重算 turn_angle_σ、curvature_σ
    - 统计跨尺度稳定度（票数/强度）
    - 将这些派生特征拼接到 base_feats，得到 feats: (B,N,14 + d_multi)
    这里是占位函数，避免误用时 silent pass。
    """
    raise NotImplementedError(
        "请在数据阶段实现多尺度高斯平滑与角度/曲率/稳定度特征构造，并拼接到 base_feats 后作为模型输入。"
    )


# =========================================================
# GraphBackbone：局部图编码主干（不是整个主模型）
# =========================================================
class GraphBackbone(nn.Module):
    """
    多分支图特征提取：GCN + GAT + EdgeConv
    用于抽取 1–2 hop 的局部/近邻几何表示。
    """

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 256,
                 use_gcn=True, use_gat=True, use_edgeconv=True, heads=4, dropout=0.1):
        super().__init__()
        self.use_gcn = use_gcn and (GCNConv is not None)
        self.use_gat = use_gat and (GATConv is not None)
        self.use_edgeconv = use_edgeconv and (EdgeConv is not None)

        chs = 0
        if self.use_gcn:
            self.gcn1 = GCNConv(in_dim, hidden)
            self.gcn2 = GCNConv(hidden, hidden)
            chs += hidden
        if self.use_gat:
            self.gat1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
            chs += hidden
        if self.use_edgeconv:
            self.ec_mlp1 = nn.Sequential(
                nn.Linear(2 * in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.edgeconv1 = EdgeConv(self.ec_mlp1, aggr='max')
            self.ec_mlp2 = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.edgeconv2 = EdgeConv(self.ec_mlp2, aggr='max')
            chs += hidden

        self.out = nn.Sequential(
            nn.Linear(chs, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index):
        feats = []
        if self.use_gcn:
            h = F.relu(self.gcn1(x, edge_index))
            h = F.relu(self.gcn2(h, edge_index))
            feats.append(h)
        if self.use_gat:
            h2 = F.elu(self.gat1(x, edge_index))
            h2 = F.elu(self.gat2(h2, edge_index))
            feats.append(h2)
        if self.use_edgeconv:
            h3 = F.relu(self.edgeconv1(x, edge_index))
            h3 = F.relu(self.edgeconv2(h3, edge_index))  # 修复：传入 edge_index
            feats.append(h3)
        y = feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)
        return self.out(y)


# =========================================================
# 几何偏置注意力 + TransformerBlock
# =========================================================
# class GeoBiasMultiheadAttention(nn.Module):
# 	"""
# 	几何偏置多头自注意力：
# 	score_ij^(h) = (Q_i K_j^T)/sqrt(dk) + b_ij^(h),
# 	b_ij^(h) = MLP([Δx, Δy, dist, dir_diff, s_i - s_j])_h
# 	支持：局部头（序号窗口或距离阈值）+ 全局头混合。
# 	"""
#
# 	def __init__(self, d_model=256, nhead=8, geo_hidden=64,
# 	             local_heads_ratio=0.5, local_win: Optional[int] = 8,
# 	             local_dist: Optional[float] = None, dropout=0.1):
# 		super().__init__()
# 		assert d_model % nhead == 0
# 		self.d_model = d_model
# 		self.nhead = nhead
# 		self.dk = d_model // nhead
# 		self.local_heads = int(nhead * local_heads_ratio)
# 		self.global_heads = nhead - self.local_heads
# 		self.local_win = local_win
# 		self.local_dist = local_dist
#
# 		self.Wq = nn.Linear(d_model, d_model, bias=False)
# 		self.Wk = nn.Linear(d_model, d_model, bias=False)
# 		self.Wv = nn.Linear(d_model, d_model, bias=False)
# 		self.Wo = nn.Linear(d_model, d_model, bias=False)
#
# 		self.geo_mlp = nn.Sequential(
# 			nn.Linear(5, geo_hidden),
# 			nn.ReLU(),
# 			nn.Linear(geo_hidden, nhead)
# 		)
# 		self.drop = nn.Dropout(dropout)
#
# 	def forward(self, x, coords, s_rel=None, dir_feat=None):
# 		# x: (B,N,D), coords: (B,N,2)
# 		B, N, D = x.shape
# 		H, dk = self.nhead, self.dk
#
# 		Q = self.Wq(x).view(B, N, H, dk).transpose(1, 2)  # (B,H,N,dk)
# 		K = self.Wk(x).view(B, N, H, dk).transpose(1, 2)
# 		V = self.Wv(x).view(B, N, H, dk).transpose(1, 2)
#
# 		logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # (B,H,N,N)
#
# 		dx, dy, dist = build_relative_geo(coords)
#
# 		# 1) s_rel 确保是 (B,N)
# 		if s_rel is None:
# 			# 假定 positional_s_over_L(N) -> (N,)
# 			s_base = positional_s_over_L(N, device=coords.device)  # (N,)
# 			s_rel = s_base.unsqueeze(0).expand(B, N)  # (B,N)
# 		else:
# 			# 如果外部传入的是 (N,) 或 (1,N)，统一到 (B,N)
# 			if s_rel.dim() == 1 and s_rel.shape[0] == N:
# 				s_rel = s_rel.unsqueeze(0).expand(B, N)
# 			elif s_rel.dim() == 2:
# 				# 允许 (1,N) 或 (B,N)
# 				if s_rel.shape[0] == 1 and s_rel.shape[1] == N:
# 					s_rel = s_rel.expand(B, N)
# 				elif s_rel.shape == (B, N):
# 					pass
# 				else:
# 					raise ValueError(f"s_rel expected shape (B,N) or (1,N) or (N,), got {tuple(s_rel.shape)}")
# 			else:
# 				raise ValueError(f"s_rel unexpected shape {tuple(s_rel.shape)}")
#
# 		# 2) dir_feat 确保是 (B,N)
# 		if dir_feat is None:
# 			dir_feat = torch.zeros(B, N, device=coords.device)
# 		else:
# 			if dir_feat.dim() == 1 and dir_feat.shape[0] == N:
# 				dir_feat = dir_feat.unsqueeze(0).expand(B, N)
# 			elif dir_feat.shape != (B, N):
# 				raise ValueError(f"dir_feat expected shape (B,N) or (N,), got {tuple(dir_feat.shape)}")
#
# 		# 3) 构造 (B,N,N) 的几何项
# 		s_i = s_rel.unsqueeze(-1).expand(B, N, N)  # (B,N,N)
# 		s_j = s_rel.unsqueeze(1).expand(B, N, N)  # (B,N,N)
# 		s_ij = s_i - s_j
#
# 		d_i = dir_feat.unsqueeze(-1).expand(B, N, N)
# 		d_j = dir_feat.unsqueeze(1).expand(B, N, N)
# 		d_diff = torch.atan2(torch.sin(d_i - d_j), torch.cos(d_i - d_j))
#
# 		geo = torch.stack([dx, dy, dist, d_diff, s_ij], dim=-1)  # (B,N,N,5)
# 		bias = self.geo_mlp(geo).permute(0, 3, 1, 2).contiguous()  # (B,H,N,N)
# 		logits = logits + bias
#
# 		# 局部头掩码：限制可见性到局部窗口/距离
# 		if self.local_heads > 0 and (self.local_win is not None or self.local_dist is not None):
# 			local_mask = torch.zeros(B, N, N, dtype=torch.bool, device=x.device)
# 			if self.local_win is not None:
# 				idx = torch.arange(N, device=x.device)
# 				for i in range(N):
# 					lo = i - self.local_win
# 					win = ((idx - lo) % N) <= (2 * self.local_win)
# 					local_mask[:, i, win] = True
# 			if self.local_dist is not None:
# 				local_mask = local_mask | (dist <= self.local_dist)
# 			neg_inf = torch.finfo(logits.dtype).min
# 			loc = slice(0, self.local_heads)
# 			logits[:, loc, :, :] = torch.where(
# 				local_mask.unsqueeze(1),
# 				logits[:, loc, :, :],
# 				torch.full_like(logits[:, loc, :, :], neg_inf)
# 			)
#
# 		attn = F.softmax(logits, dim=-1)
# 		attn = self.drop(attn)
# 		out = torch.matmul(attn, V)  # (B,H,N,dk)
# 		out = out.transpose(1, 2).contiguous().view(B, N, D)
# 		return self.Wo(out), attn
#

class GeoBiasMultiheadAttention(nn.Module):
    """
    几何偏置多头自注意力：
    score_ij^(h) = (Q_i K_j^T)/sqrt(dk) + b_ij^(h),
    b_ij^(h) = MLP([Δx, Δy, dist, dir_diff, s_i - s_j])_h
    支持：局部头（序号窗口或距离阈值）+ 全局头混合 + padding/有效性 mask。
    """

    def __init__(self, d_model=256, nhead=8, geo_hidden=64,
                 local_heads_ratio=0.5, local_win: Optional[int] = 8,
                 local_dist: Optional[float] = None, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.local_heads = int(nhead * local_heads_ratio)
        self.global_heads = nhead - self.local_heads
        self.local_win = local_win
        self.local_dist = local_dist

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.geo_mlp = nn.Sequential(
            nn.Linear(5, geo_hidden),
            nn.ReLU(),
            nn.Linear(geo_hidden, nhead)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, coords, s_rel=None, dir_feat=None, mask: Optional[torch.Tensor] = None):
        """
        x:      (B,N,D)
        coords: (B,N,2)
        mask:   (B,N)  True/1 表示有效节点；若为 None 则视为全有效
        """
        B, N, D = x.shape
        H, dk = self.nhead, self.dk

        Q = self.Wq(x).view(B, N, H, dk).transpose(1, 2)  # (B,H,N,dk)
        K = self.Wk(x).view(B, N, H, dk).transpose(1, 2)
        V = self.Wv(x).view(B, N, H, dk).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # (B,H,N,N)

        dx, dy, dist = build_relative_geo(coords)  # 需返回 (B,N,N) 的 dx, dy, dist

        # 1) s_rel 统一到 (B,N)
        if s_rel is None:
            s_base = positional_s_over_L(N, device=coords.device)  # (N,)
            s_rel = s_base.unsqueeze(0).expand(B, N)  # (B,N)
        else:
            if s_rel.dim() == 1 and s_rel.shape[0] == N:
                s_rel = s_rel.unsqueeze(0).expand(B, N)
            elif s_rel.dim() == 2 and s_rel.shape in [(1, N), (B, N)]:
                s_rel = s_rel.expand(B, N) if s_rel.shape[0] == 1 else s_rel
            else:
                raise ValueError(f"s_rel expected (B,N)/(1,N)/(N,), got {tuple(s_rel.shape)}")

        # 2) dir_feat 统一到 (B,N)
        if dir_feat is None:
            dir_feat = torch.zeros(B, N, device=coords.device)
        else:
            if dir_feat.dim() == 1 and dir_feat.shape[0] == N:
                dir_feat = dir_feat.unsqueeze(0).expand(B, N)
            elif dir_feat.shape != (B, N):
                raise ValueError(f"dir_feat expected (B,N) or (N,), got {tuple(dir_feat.shape)}")

        # 3) 几何偏置
        s_i = s_rel.unsqueeze(-1).expand(B, N, N)
        s_j = s_rel.unsqueeze(1).expand(B, N, N)
        s_ij = s_i - s_j

        d_i = dir_feat.unsqueeze(-1).expand(B, N, N)
        d_j = dir_feat.unsqueeze(1).expand(B, N, N)
        d_diff = torch.atan2(torch.sin(d_i - d_j), torch.cos(d_i - d_j))

        geo = torch.stack([dx, dy, dist, d_diff, s_ij], dim=-1)  # (B,N,N,5)
        bias = self.geo_mlp(geo).permute(0, 3, 1, 2).contiguous()  # (B,H,N,N)
        logits = logits + bias

        # ============ 构造总可见性掩码 allow: (B,1,N,N) ============
        # 先初始化为全 True
        allow = torch.ones(B, 1, N, N, dtype=torch.bool, device=x.device)

        # a) padding/有效性 mask（key 侧）
        if mask is not None:
            key_mask = (mask if mask.dtype == torch.bool else (mask != 0)).view(B, 1, 1, N)
            allow = allow & key_mask  # 只允许 key 有效的位置

        # b) 局部头窗口/半径掩码（仅对前 self.local_heads 个头生效）
        if self.local_heads > 0 and (self.local_win is not None or self.local_dist is not None):
            # 基于环形序号的窗口掩码
            win_mask = torch.ones(B, N, N, dtype=torch.bool, device=x.device)
            if self.local_win is not None:
                idx = torch.arange(N, device=x.device)
                # 构造环形最短步数距离矩阵 (N,N)
                diff_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
                cyc_steps = torch.minimum(diff_idx, N - diff_idx)  # (N,N)
                win_mask = (cyc_steps <= self.local_win).unsqueeze(0).expand(B, -1, -1)
            # 距离阈值掩码
            if self.local_dist is not None:
                dist_mask = (dist <= self.local_dist)  # (B,N,N)
                win_mask = win_mask & dist_mask

            # 将局部掩码应用到局部头
            allow[:, :self.local_heads, :, :] = allow[:, :self.local_heads, :, :] & win_mask.unsqueeze(1)

        # c) 对角线保护：保证自身可见，避免整行全 False
        eye = torch.eye(N, dtype=torch.bool, device=x.device).view(1, 1, N, N)
        allow = allow | eye  # 所有头都保留对角

        # ============ 一次性掩码到 logits ============
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(allow, logits, torch.full_like(logits, neg_inf))

        # softmax + dropout
        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)  # (B,H,N,dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.Wo(out), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, mlp_ratio=4, dropout=0.1,
                 local_heads_ratio=0.5, local_win=8, local_dist=None, **geo_kwargs):
        super().__init__()
        self.attn = GeoBiasMultiheadAttention(
            d_model=d_model, nhead=nhead, geo_hidden=64,
            local_heads_ratio=local_heads_ratio, local_win=local_win,
            local_dist=local_dist, dropout=dropout, **geo_kwargs
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, coords, s_rel=None, dir_feat=None, mask: Optional[torch.Tensor] = None):
        a, attn = self.attn(x, coords, s_rel=s_rel, dir_feat=dir_feat, mask=mask)
        x = self.ln1(x + a)
        y = self.mlp(x)
        x = self.ln2(x + y)
        return x, attn


# =========================================================
# 主模型：FeaturePointNet（特征点 keep=1/0）
# =========================================================
# class FeaturePointNet(nn.Module):
# 	"""
# 	输入：无Mask的版本
# 	  - coords: (B,N,2)
# 	  - feats:  (B,N,in_dim)，建议 in_dim = 14 + d_multi
# 		其中 d_multi 为多尺度高斯平滑后重算的 turn_angle_σ、curvature_σ 与稳定度等派生维度
# 	  - edge_index: (2,E) 可为空（内部构造环）
# 	  - extra_edges: (2,E2) 可选跨段边（可见性/弦/kNN/最短路）
# 	  - add_ring_skip: 是否自动添加跳边（2-hop/3-hop）
# 	输出：
# 	  - keep_logits: (B,N) 每点特征点对数几率（BCEWithLogitsLoss 训练）
# 	  - attn_maps:   list of (B,H,N,N) 各层注意力权重
# 	  - emb:         (B,N,d_model) 最终顶点嵌入
# 	"""
#
# 	def __init__(self, in_dim=22, d_model=256, heads=8, depth=4,
# 	             use_gcn=True, use_gat=True, use_edgeconv=True,
# 	             local_heads_ratio=0.5,
# 	             local_win=8, local_dist=None, dropout=0.1):
# 		super().__init__()
# 		self.backbone = GraphBackbone(
# 			in_dim=in_dim, hidden=d_model // 2, out_dim=d_model,
# 			use_gcn=use_gcn, use_gat=use_gat, use_edgeconv=use_edgeconv,
# 			heads=max(1, heads // 2), dropout=dropout
# 		)
# 		self.blocks = nn.ModuleList([
# 			TransformerBlock(
# 				d_model=d_model, nhead=heads, mlp_ratio=4, dropout=dropout,
# 				local_heads_ratio=local_heads_ratio, local_win=local_win, local_dist=local_dist
# 			) for _ in range(depth)
# 		])
# 		self.head = nn.Sequential(
# 			nn.Linear(d_model, d_model),
# 			nn.ReLU(),
# 			nn.Linear(d_model, 1)
# 		)
#
# 	def forward(self, coords: torch.Tensor, feats: torch.Tensor,
# 	            edge_index: Optional[torch.Tensor] = None,
# 	            extra_edges: Optional[torch.Tensor] = None,
# 	            add_ring_skip: bool = False,
# 	            ring_skip_hops=(2, 3)):
# 		B, N, _ = coords.shape
# 		device = coords.device
#
# 		# 构造批环边 + 可选跳边
# 		if edge_index is None:
# 			all_e = []
# 			for b in range(B):
# 				e = ring_adj(N, device=device)
# 				if add_ring_skip:
# 					e_skip = build_skip_edges_for_ring(N, hops=ring_skip_hops, device=device)
# 					e = torch.cat([e, e_skip], dim=1)
# 				e = e + b * N
# 				all_e.append(e)
# 			edge_index = torch.cat(all_e, dim=1)
#
# 		# 追加跨段边
# 		if extra_edges is not None:
# 			edge_index = torch.cat([edge_index, extra_edges], dim=1)
#
# 		# 局部图编码
# 		x = feats.reshape(B * N, -1)
# 		h = self.backbone(x, edge_index)  # (B*N, d_model)
# 		h = h.view(B, N, -1)  # (B,N,d_model)
#
# 		# 全局几何偏置注意力
# 		len_ratio = feats[..., 7]  # (B,N)
# 		s_rel = positional_s_over_L(N, device=device, feats_len_ratio=len_ratio
# 		                            )
# 		dir_feat = feats[..., 8]  # 可从 feats 抽取某列作为方向参考（如 turn_angle）
#
# 		attn_list = []
# 		for blk in self.blocks:
# 			h, attn = blk(h, coords, s_rel=s_rel, dir_feat=dir_feat)
# 			attn_list.append(attn)
#
# 		keep_logits = self.head(h).squeeze(-1)  # (B,N)
# 		return {
# 			'keep_logits': keep_logits,
# 			'attn_maps': attn_list,
# 			'emb': h
# 		}

class FeaturePointNet(nn.Module):
    """
    输入：有Mask的版本
      - coords: (B,N,2)
      - feats:  (B,N,in_dim)
      - edge_index: (2,E) 可为空（内部构造环）
      - extra_edges: (2,E2) 可选跨段边
    输出：
      - keep_logits: (B,N)
      - attn_maps:   list[(B,H,N,N)]
      - emb:         (B,N,d_model)
    """

    def __init__(self, in_dim=14, d_model=256, heads=8, depth=4,
                 use_gcn=True, use_gat=True, use_edgeconv=True,
                 local_heads_ratio=0.5,
                 local_win=8, local_dist=None, dropout=0.1):
        super().__init__()
        self.backbone = GraphBackbone(
            in_dim=in_dim, hidden=d_model // 2, out_dim=d_model,
            use_gcn=use_gcn, use_gat=use_gat, use_edgeconv=use_edgeconv,
            heads=max(1, heads // 2), dropout=dropout
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, nhead=heads, mlp_ratio=4, dropout=dropout,
                local_heads_ratio=local_heads_ratio, local_win=local_win, local_dist=local_dist
            ) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, coords: torch.Tensor, feats: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                extra_edges: Optional[torch.Tensor] = None,
                add_ring_skip: bool = False,
                ring_skip_hops=(2, 3),
                mask: Optional[torch.Tensor] = None):
        B, N, _ = coords.shape
        device = coords.device

        # 0) 规范化 mask（True=有效）
        if mask is None:
            valid = torch.ones((B, N), dtype=torch.bool, device=device)
        else:
            valid = mask if mask.dtype == torch.bool else (mask != 0)

        # 1) 构造批环边 + 可选跳边（保持原逻辑不变）
        if edge_index is None:
            all_e = []
            for b in range(B):
                e = ring_adj(N, device=device)
                if add_ring_skip:
                    e_skip = build_skip_edges_for_ring(N, hops=ring_skip_hops, device=device)
                    e = torch.cat([e, e_skip], dim=1)
                e = e + b * N
                all_e.append(e)
            edge_index = torch.cat(all_e, dim=1)
        if extra_edges is not None:
            edge_index = torch.cat([edge_index, extra_edges], dim=1)

        # 2) 屏蔽无效位置的特征，降低污染
        feats = feats.clone()
        feats[~valid] = 0.0

        # 3) 局部图编码（展平到 (B*N, F)）
        x = feats.reshape(B * N, -1)
        h = self.backbone(x, edge_index)  # (B*N, d_model)
        h = h.view(B, N, -1)

        # 4) 全局几何偏置注意力所需辅助量（保持原逻辑）
        len_ratio = feats[..., 8]  # (B,N)
        s_rel = positional_s_over_L(N, device=device, feats_len_ratio=len_ratio)
        dir_feat = feats[..., 7]

        # 5) 堆叠的 TransformerBlock
        attn_list = []
        for blk in self.blocks:
            # 如果你的 TransformerBlock 支持 mask/shortcut_mask 参数，请在这里传递：
            # 例如：blk(h, coords, s_rel=s_rel, dir_feat=dir_feat, mask=valid)
            try:
                h, attn = blk(h, coords, s_rel=s_rel, dir_feat=dir_feat, mask=valid)
            except TypeError:
                # 若不支持 mask 参数，则退化为不传 mask；随后再屏蔽无效位置
                h, attn = blk(h, coords, s_rel=s_rel, dir_feat=dir_feat)
            # 保证无效位置不被后续传播使用（可选，但更稳健）
            h = h.masked_fill(~valid.unsqueeze(-1), 0.0)
            attn_list.append(attn)

        keep_logits = self.head(h).squeeze(-1)  # (B,N)

        return {
            'keep_logits': keep_logits,
            'attn_maps': attn_list,
            'emb': h
        }


# =========================================================
# 损失封装（可选）
# =========================================================
class FeaturePointLoss(nn.Module):
    """
    节点二分类损失：BCEWithLogitsLoss
    - 建议使用 pos_weight 处理正负不平衡
    """

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is None:
            self.crit = nn.BCEWithLogitsLoss()
        else:
            w = torch.tensor([pos_weight], dtype=torch.float32)
            self.crit = nn.BCEWithLogitsLoss(pos_weight=w)

    def forward(self, keep_logits: torch.Tensor, keep_targets: torch.Tensor):
        """
        keep_logits: (B,N), raw logits
        keep_targets: (B,N) in {0,1}
        """
        return self.crit(keep_logits, keep_targets.float())


# =========================================================
# 使用示例（注释）
# =========================================================
"""
使用示例（训练时）：
------------------------------------------------------------
# 1) 预处理阶段：生成 feats = concat(base14, multi-scale d_multi) -> (B,N,14+d_multi)
#   - 对 coords 在 σ∈{1,2,3} 上高斯平滑
#   - 计算 turn_angle_σ、curvature_σ
#   - 统计跨尺度稳定度 stable_votes / stable_strength
#   - feats = torch.cat([base14, multi], dim=-1)

coords = torch.randn(B, N, 2)
feats  = torch.randn(B, N, 14 + d_multi)  # 请用实际构造结果替换
targets = torch.randint(0, 2, (B, N))     # keep=1/0

model = FeaturePointNet(
    in_dim=14 + d_multi,
    d_model=256, heads=8, depth=4,
    use_gcn=True, use_gat=True, use_edgeconv=True,
    local_heads_ratio=0.5, local_win=8, local_dist=None, dropout=0.1
)

# 可选：构造跨段边 extra_edges 或启用跳边
out = model(coords, feats, edge_index=None,
            extra_edges=None, add_ring_skip=True, ring_skip_hops=(2,3))
keep_logits = out['keep_logits']

# 损失与优化
pos_weight = (targets.numel() - targets.sum()) / (targets.sum() + 1e-6)
criterion = FeaturePointLoss(pos_weight=float(pos_weight))
loss = criterion(keep_logits, targets)
loss.backward()
optimizer.step()

推理时：
------------------------------------------------------------
with torch.no_grad():
    out = model(coords, feats, add_ring_skip=True)
    keep_prob = torch.sigmoid(out['keep_logits'])  # (B,N)
    # 选择策略：阈值或Top-K
    keep_mask = keep_prob > 0.5
"""
