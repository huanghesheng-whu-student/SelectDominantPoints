# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

import geopandas as gpd
from shapely.geometry import Point

import torch
import torch.nn as nn

# 导入你的模型与数据装载逻辑
from Models0203 import FeaturePointNet  # 确保类名与文件一致
from DataLoading1029 import load_training_polygons_and_points_no_poly_0329

# 项目中和路径有关的参数
PATH_KEYS = {
    "ckpt_path",
    "points_geojson",
    "out_geojson"

}


# 反归一化：与 _norm_single_polygon 的 'minmax' 对应
def denorm_coords(coords_norm: np.ndarray,
                  norm_params: Dict[str, Any]) -> np.ndarray:
    mode = norm_params.get('mode', None)
    if mode is None:
        return coords_norm
    if mode == 'minmax':
        mn = np.asarray(norm_params['min'], dtype=float)
        mx = np.asarray(norm_params['max'], dtype=float)
        eps = 1e-8
        return coords_norm * (mx - mn + eps) + mn
    return coords_norm


def make_out_geojson_path(ckpt_or_cfg):
    """
    根据 cfg['ckpt_path'] 生成 out_geojson 路径。
    规则：
    - 将 ckpt_path 中的 'checkpoints' 目录替换为同层级的 'data'
    - 从 ckpt_path 提取实验名（exp）= 上上级目录名、fold = 上级目录名
    - 文件名形如：val_keep_pred_{exp}_{fold}.geojson
    例如：
    ckpt: F:\\LandUseBoudary\\code\\checkpoints\\0331_02号实验\\fold_3\\feature_point_net_semisup_best.pth
    out: F:\\LandUseBoudary\\code\\data\\val_keep_pred_0331_02号实验_fold_3.geojson
    """
    # 1) 规范化输入
    if isinstance(ckpt_or_cfg, (str, Path)):
        ckpt_path = Path(ckpt_or_cfg)
    elif isinstance(ckpt_or_cfg, dict) and 'ckpt_path' in ckpt_or_cfg:
        ckpt_path = Path(ckpt_or_cfg['ckpt_path'])
    else:
        raise TypeError("make_out_geojson_path 需要传入字符串/Path 或包含 'ckpt_path' 的字典。")

    # 2) 提取 fold 与实验名
    fold = ckpt_path.parent.name  # 例如 fold_3
    exp = ckpt_path.parent.parent.name  # 例如 0331_02号实验

    # 3) 找到 'checkpoints' 段，替换为同层级 'data'
    parts = ckpt_path.parts
    idx = next((i for i, seg in enumerate(parts) if seg.lower() == 'checkpoints'), None)
    if idx is None:
        raise ValueError(f"'checkpoints' 未在路径中找到：{ckpt_path}")

    # checkpoints 之前的根目录，例如 F:\LandUseBoudary\code
    base_root = Path(*parts[:idx])
    out_dir = base_root / 'data'
    out_name = f"val_keep_pred_{exp}_{fold}.geojson"
    out_path = out_dir / out_name

    return str(out_path)


@torch.no_grad()
def infer_keep_on_val(
        model: nn.Module,
        ckpt_path: str,
        points_geojson: str,
        target_crs: Optional[str] = None,
        in_dim: int = 20,
        use_only_coords: bool = True,
        per_polygon_norm: Optional[str] = 'minmax',
        sigmas=[1.0, 2.0, 3.0],
        device: str = 'cuda',
        out_geojson: str = r"",
        prob_threshold: float = 0.5
):
    """
    在验证/测试集上逐多边形逐点推理 keep 概率，并输出 GeoJSON（每个点一条要素）。
    """
    out_geojson = make_out_geojson_path(ckpt_path)
    # 1) 加载数据（与训练一致的参数）
    polys, building_ids, norm_params_list = load_training_polygons_and_points_no_poly_0329(
        points_geojson=points_geojson,
        sigmas=sigmas,
        w_multi_scale=False
    )

    # 2) 模型与权重
    device_t = torch.device(device if ('cuda' in device and torch.cuda.is_available()) else 'cpu')
    model.to(device_t)
    model.eval()
    assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device_t)
    state_dict = ckpt.get('state_dict', ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded checkpoint strictly: {ckpt_path}")
    except Exception as e:
        print(f"[WARN] strict load failed: {e}\n[INFO] try non-strict...")
        missing_unexp = model.load_state_dict(state_dict, strict=False)
        if isinstance(missing_unexp, tuple):
            missing, unexpected = missing_unexp
            if missing:
                print("[WARN] missing keys:", missing)
            if unexpected:
                print("[WARN] unexpected keys:", unexpected)

    # 3) 读取原始点文件，拿原始字段（如 idx、keep 等），便于合并属性
    gdf_pts_all = gpd.read_file(points_geojson)
    if target_crs is not None and gdf_pts_all.crs is not None and gdf_pts_all.crs.to_string() != target_crs:
        gdf_pts_all = gdf_pts_all.to_crs(target_crs)
    crs_out = gdf_pts_all.crs

    features = []

    # 4) 推理逐 polygon（逐点）
    for i, poly in enumerate(polys):
        bid = poly["building_id"]
        coords_norm_t: torch.Tensor = poly["coords"]  # (N,2) float32, 已归一化
        feats_t: Optional[torch.Tensor] = poly.get("feats", None)  # (N,F) or None
        keep_true_t: torch.Tensor = poly.get("keep", torch.zeros(coords_norm_t.shape[0]))  # (N,)
        N = coords_norm_t.shape[0]

        if feats_t is None:
            raise ValueError(
                "当前模型需要 feats 和 coords 两输入。请提供 feats_t 或用训练时的同一函数从 coords 构造特征。"
            )

        feats_b = feats_t.unsqueeze(0).to(device_t)  # (1,N,F)
        coords_b = coords_norm_t.unsqueeze(0).to(device_t)  # (1,N,2)

        out = model(
            feats=feats_b,
            coords=coords_b
        )

        # 兼容 keep_prob / keep_logits 两种返回
        if isinstance(out, dict):
            if "keep_prob" in out:
                keep_prob = out["keep_prob"].squeeze(0)  # (N,)
            elif "keep_logits" in out:
                keep_prob = torch.sigmoid(out["keep_logits"].squeeze(0))  # (N,)
            else:
                raise KeyError("模型输出缺少 'keep_prob' 或 'keep_logits'。")
        else:
            keep_prob = torch.sigmoid(out.squeeze(0))  # (N,)

        # 二值预测并保证至少4个点
        keep_pred = (keep_prob >= prob_threshold).to(torch.uint8)
        cur_cnt = int(keep_pred.sum().item())
        if cur_cnt < 4 and keep_prob.numel() > 0:
            K = min(4, keep_prob.numel())
            _, topk_idx = torch.topk(keep_prob, k=K, largest=True, sorted=True)
            keep_pred[topk_idx] = 1

        # 反归一化用于写回
        coords_norm = coords_norm_t.cpu().numpy().astype(np.float64)  # (N,2)
        coords_denorm = denorm_coords(coords_norm, norm_params_list[i])

        # 用原始 gdf 对齐
        sub = gdf_pts_all[gdf_pts_all['building_id'] == bid].copy()
        if 'idx' not in sub.columns or sub['idx'].isna().any():
            sub = sub.reset_index(drop=True)
            sub['idx'] = np.arange(len(sub), dtype=np.int64)
        sub = sub.sort_values('idx').reset_index(drop=True)

        if sub.shape[0] == N + 1:
            x0, y0 = sub.geometry.iloc[0].x, sub.geometry.iloc[0].y
            xn, yn = sub.geometry.iloc[-1].x, sub.geometry.iloc[-1].y
            if np.allclose([x0, y0], [xn, yn]):
                sub = sub.iloc[:-1, :].reset_index(drop=True)

        assert sub.shape[0] == N, f"ID {bid}: point count mismatch. sub={sub.shape[0]} vs N={N}"

        probs_np = keep_prob.detach().cpu().numpy().astype(float)
        preds_np = keep_pred.detach().cpu().numpy().astype(int)
        for idx_row in range(N):
            x_real, y_real = coords_denorm[idx_row]
            geom = Point(float(x_real), float(y_real))
            geom_json = json.loads(gpd.GeoSeries([geom], crs=crs_out).to_json())["features"][0]["geometry"]
            feat = {
                "type": "Feature",
                "geometry": geom_json,
                "properties": {
                    "building_id": int(bid) if isinstance(bid, (int, np.integer)) else bid,
                    "idx": int(sub.loc[idx_row, 'idx']),
                    "keep_true": float(keep_true_t[idx_row].item()) if isinstance(keep_true_t, torch.Tensor) else float(
                        keep_true_t[idx_row]),
                    "keep_prob": float(probs_np[idx_row]),
                    "keep_pred": int(preds_np[idx_row]),
                }
            }
            features.append(feat)

    # 5) 输出 GeoJSON
    fc = {
        "type": "FeatureCollection",
        "features": features
    }
    out_dir = os.path.dirname(out_geojson)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"[OK] Saved predicted keep points to: {out_geojson}")


def build_model_from_cfg(cfg: Dict[str, Any]) -> nn.Module:
    """
    根据配置构造 FeaturePointNet 模型。
    需确保与训练时超参数一致。
    """
    model = FeaturePointNet(
        in_dim=int(cfg["in_dim"]),
        d_model=int(cfg["d_model"]),
        heads=int(cfg["heads"]),
        depth=int(cfg["depth"]),
        use_gcn=bool(cfg["use_gcn"]),
        use_gat=bool(cfg["use_gat"]),
        use_edgeconv=bool(cfg["use_edgeconv"]),
        local_heads_ratio=float(cfg["local_heads_ratio"]),
        local_win=int(cfg["local_win"]),
        local_dist=(None if cfg.get("local_dist", None) in [None, "None", "null"] else float(cfg["local_dist"])),
        dropout=float(cfg["dropout"])
    )
    return model


def check_required(cfg: Dict[str, Any]):
    required = [
        "ckpt_path",
        "points_geojson",
        "out_geojson",
        "device",
        "in_dim",
        "d_model",
        "heads",
        "depth",
        "use_gcn",
        "use_gat",
        "use_edgeconv",
        "local_heads_ratio",
        "local_win",
        "local_dist",
        "dropout",
        "sigmas",
        "prob_threshold",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"YAML 缺少必要参数: {missing}")


def _normalize_windows_path(p: str) -> str:
    """
    将 Windows 路径规范化，兼容以下形式：
      - F:\\LandUseBoudary\\code\\config\\file.yml   (双反斜杠)
      - F:\LandUseBoudary\code\config\file.yml       (单反斜杠)
      - F:/LandUseBoudary/code/config/file.yml       (正斜杠)
    并清理意外的重复分隔符、尾部空白、隐藏控制字符。
    """
    if not isinstance(p, str):
        return p
    # 去除首尾空白与常见不可见控制符
    p = p.strip().strip('\u200b').strip('\ufeff')
    # 统一替换为正斜杠，避免单反斜杠被误认为转义
    p = p.replace("\\", "/")
    # 折叠可能的多重分隔符 '//' -> '/'
    p = re.sub(r"/{2,}", "/", p)
    # 交给 normpath 做平台规范化，然后返回绝对路径（可选）
    p = os.path.normpath(p)
    # 如果是 Windows 盘符路径，确保大小写/格式一致；相对路径保持不变亦可
    try:
        p = os.path.abspath(p)
    except Exception:
        # 某些场景（如 VFS、容器）下 abspath 可能异常，忽略
        pass
    return p


def _normalize_paths_in_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    遍历配置字典，把路径相关键值规范化。
    支持嵌套 dict / list 结构。
    """

    def _walk(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str) and k in PATH_KEYS:
                    out[k] = _normalize_windows_path(v)
                else:
                    out[k] = _walk(v)
            return out
        elif isinstance(obj, list):
            return [_walk(x) for x in obj]
        else:
            return obj

    return _walk(cfg)


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    # 先规范化传入的配置文件路径
    yaml_path = _normalize_windows_path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层应为字典。")
    # 对配置中的路径字段做统一规范化
    cfg = _normalize_paths_in_cfg(cfg)
    return cfg


if __name__ == "__main__":
    """
    使用方式：
      python test.py --cfg path/to/your_config.yml

    YAML 示例（ test.yml）：
    
    """

    parser = argparse.ArgumentParser(description="Inference on test dataset with YAML config.")
    parser.add_argument('--cfg', '-c', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    # 1) 读取配置并校验
    cfg = load_yaml(args.cfg)
    check_required(cfg)

    # 2) 构造设备字符串
    device_str = str(cfg.get("device", "cpu")).lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切换到 CPU")
        device_str = "cpu"

    # 3) 构造模型（确保与训练一致）
    model = build_model_from_cfg(cfg)
    print("[INFO] Model built:", model.__class__.__name__)

    # 4) 执行推理
    infer_keep_on_val(
        model=model,
        ckpt_path=str(cfg["ckpt_path"]),
        points_geojson=str(cfg["points_geojson"]),
        target_crs=None,
        in_dim=int(cfg["in_dim"]),
        use_only_coords=True,
        per_polygon_norm='minmax',
        sigmas=list(cfg["sigmas"]),
        device=device_str,
        out_geojson=str(cfg["out_geojson"]),
        prob_threshold=float(cfg["prob_threshold"])
    )

    print("[DONE] Inference finished.")
