import os
from collections import defaultdict
from typing import Dict, Any, Optional

import torch

from GeoUtils import *
from convexhullTest import distances_points_to_convex_hull_edges, graham_scan_convex_hull


# region--------------邓敏，节点重要性-------------------------------------------------------------------------
def find_support_radius(coords):
    """
    计算支撑域的半径 R
    :param coords: 二维ndarray，形状为 (n, 2)，表示多边形轮廓点坐标
    :return: 支撑域半径 R
    """
    # 计算周长 L0：相邻点距离之和 + 闭合段（最后一点到第一点）
    diffs = coords[1:] - coords[:-1]  # (V-1, 2)
    segs = np.sqrt((diffs ** 2).sum(axis=1))  # (V-1,)
    close = np.linalg.norm(coords[0] - coords[-1])
    L0 = float(segs.sum() + close)

    V = len(coords)
    R = L0 / (V + 1)
    return R


def find_intersection_points(coords, center, radius):
    """
    计算支撑域圆与曲线的交点
    :param coords: 二维ndarray，形状为 (n, 2)，表示多边形轮廓点坐标
    :param center: 圆心坐标 [x, y]
    :param radius: 圆的半径 R
    :return: 支撑域圆与曲线的交点数组
    """
    n = len(coords)
    intersections = []
    for i in range(n):
        # 获取当前线段的两个端点
        p1 = coords[i % n]
        p2 = coords[(i + 1) % n]

        # 计算线段与圆的交点
        # 圆的方程：(x - cx)^2 + (y - cy)^2 = R^2
        # 线段的参数方程：x = x1 + t * (x2 - x1), y = y1 + t * (y2 - y1)
        # 联立方程，求解 t
        dx, dy = p2 - p1
        fx, fy = p1 - center
        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = fx ** 2 + fy ** 2 - radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant >= 0:  # 存在交点
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)

            # 检查 t1 和 t2 是否在 [0, 1] 范围内
            if 0 <= t1 <= 1:
                intersection = p1 + t1 * (p2 - p1)
                intersections.append(intersection)
            if 0 <= t2 <= 1:
                intersection = p1 + t2 * (p2 - p1)
                intersections.append(intersection)

    return np.array(intersections)


def sample_segment(p1, p2, step=1.0):
    """
    在线段 p1->p2 上按给定步长进行采样，包含端点。
    step 为近似的点间距（单位与坐标一致），步数会根据长度自动调整。
    """
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        return np.array([p1])
    num = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(0.0, 1.0, num)
    return p1[None, :] + ts[:, None] * vec[None, :]


def find_boundary_indices(coords: np.ndarray, res: np.ndarray, tol: float = 1e-9):
    """
    根据 res（在圆内的顶点坐标集合）和 coords（原始多边形顶点序列），
    找到：
      - 在 res 点集合前面的一个点（即第一个 res 点在 coords 中的前一个顶点，循环）
      - 在 res 点集合后面的一个点（即最后一个 res 点在 coords 中的后一个顶点，循环）
    返回它们在 coords 中的序号索引 (prev_idx, next_idx)。

    参数:
        coords: ndarray, shape (N, 2)，多边形顶点顺序坐标（闭合或不闭合都可，函数按循环处理）
        res:    ndarray, shape (M, 2)，在圆内的顶点坐标集合（来自 coords）
        tol:    float，坐标匹配的容差，用于处理浮点误差

    返回:
        (prev_idx, next_idx): 两个整数索引，分别是：
            prev_idx = 第一个 res 点在 coords 中的前一个顶点的索引（循环）
            next_idx = 最后一个 res 点在 coords 中的后一个顶点的索引（循环）

    异常:
        ValueError: 当 res 为空、或 res 中某点无法在 coords 中匹配（超出容差），会抛出异常
    """
    if res.size == 0:
        raise ValueError("res 为空，无法确定边界点。")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords 需为形状 (N, 2) 的二维数组。")
    if res.ndim != 2 or res.shape[1] != 2:
        raise ValueError("res 需为形状 (M, 2) 的二维数组。")

    N = coords.shape[0]

    # 建立一个从坐标到索引的快速匹配（容差匹配使用 KD-like 简单线性查找，通常足够）
    def match_index(point):
        # 逐点检查匹配，考虑容差
        diffs = np.abs(coords - point)
        mask = np.all(diffs <= tol, axis=1)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            # 如果严格匹配失败，尝试距离容差匹配
            dists = np.linalg.norm(coords - point, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= tol:
                return idx
            raise ValueError(f"在 coords 中未找到与点 {point} 匹配的顶点（容差 {tol}）。")
        # 若有多个完全匹配，取第一个
        return int(idxs[0])

    # 找到 res 中每个点对应的 coords 索引
    res_indices = np.array([match_index(p) for p in res], dtype=int)

    # 为了稳健，按索引在多边形顺序上排序（防止 res 输入顺序乱）
    res_indices_sorted = np.sort(res_indices)

    # 第一个和最后一个索引（按多边形顺序）
    first_idx = int(res_indices_sorted[0])
    last_idx = int(res_indices_sorted[-1])

    # 循环前后索引
    prev_idx = (first_idx - 1) % N
    next_idx = (last_idx + 1) % N

    return prev_idx, next_idx


# def points_on_polygon_within_circle(coords, center, radius, step=1.0):
# 	"""
# 	基于交点，计算从第一个交点到第二个交点之间、且位于圆内的多边形轮廓采样点。
# 	- coords: (n,2) 不闭合（首尾不重合）
# 	- center: [cx, cy]
# 	- radius: R
# 	- step: 轮廓采样的近似间距
# 	返回：按轮廓顺序排列的点数组 (m,2)
# 	"""
# 	inters, info = find_intersection_points(coords, center, radius)
# 	if len(inters) < 2:
# 		# 无法构造区间，返回空
# 		return np.empty((0, 2))
# 	# 只取两点的典型情况；如有更多交点，可按需要排序和成对处理
# 	# 将交点按沿轮廓顺序排序（先按边索引，再按 t）
# 	info_with_pts = list(zip(inters, info))
# 	info_with_pts.sort(key=lambda it: (it[1][0], it[1][1]))  # (edge_idx, t)
# 	(pA, (edgeA, tA)), (pB, (edgeB, tB)) = info_with_pts[0], info_with_pts[1]
#
# 	cx, cy = center
# 	R2 = radius * radius
#
# 	def inside(pt):
# 		dx = pt[0] - cx
# 		dy = pt[1] - cy
# 		return dx * dx + dy * dy <= R2 + 1e-9  # 容差
#
# 	n = len(coords)
# 	result_pts = []
#
# 	# 1) 处理起始边 edgeA，从交点 tA 到该边的终点
# 	p1A = coords[edgeA]
# 	p2A = coords[(edgeA + 1) % n]
# 	# 起段：从 tA 到 1.0
# 	start_seg_start = pA
# 	start_seg_end = p2A
# 	seg_pts = sample_segment(start_seg_start, start_seg_end, step=step)
# 	seg_pts = seg_pts[[inside(pt) for pt in seg_pts]]
# 	if len(seg_pts) > 0:
# 		# 确保第一个点是交点
# 		if not np.allclose(seg_pts[0], pA):
# 			seg_pts = np.vstack([pA, seg_pts])
# 		result_pts.append(seg_pts)
#
# 	# 2) 处理中间完整边，从 edgeA+1 到 edgeB-1（循环）
# 	i = (edgeA + 1) % n
# 	while i != edgeB:
# 		q1 = coords[i]
# 		q2 = coords[(i + 1) % n]
# 		# 判断整条边是否在圆内；这里以端点都在圆内作为快速判定
# 		if inside(q1) and inside(q2):
# 			mid_pts = sample_segment(q1, q2, step=step)
# 			result_pts.append(mid_pts)
# 		else:
# 			# 边可能部分在圆内，但交点应位于 edgeA 或 edgeB；中间边通常全在圆内或全在圆外
# 			# 为稳妥起见，对该边进行采样后过滤
# 			mid_pts = sample_segment(q1, q2, step=step)
# 			mid_pts = mid_pts[[inside(pt) for pt in mid_pts]]
# 			if len(mid_pts) > 0:
# 				result_pts.append(mid_pts)
# 		i = (i + 1) % n
#
# 	# 3) 处理终止边 edgeB，从该边起点到交点 tB
# 	p1B = coords[edgeB]
# 	p2B = coords[(edgeB + 1) % n]
# 	end_seg_start = p1B
# 	end_seg_end = pB
# 	seg_pts2 = sample_segment(end_seg_start, end_seg_end, step=step)
# 	seg_pts2 = seg_pts2[[inside(pt) for pt in seg_pts2]]
# 	if len(seg_pts2) > 0:
# 		# 确保最后一个点是交点
# 		if not np.allclose(seg_pts2[-1], pB):
# 			seg_pts2 = np.vstack([seg_pts2, pB])
# 		result_pts.append(seg_pts2)
#
# 	if len(result_pts) == 0:
# 		return np.empty((0, 2))
#
# 	res = np.vstack(result_pts)
#
# 	# 去重与顺序优化：去除相邻重复点
# 	if len(res) > 1:
# 		keep = [True]
# 		for k in range(1, len(res)):
# 			keep.append(not np.allclose(res[k], res[k - 1]))
# 		res = res[np.array(keep)]
#
# 	return res

def points_on_polygon_within_circle(coords, center, radius):
    # coords: np.ndarray of shape (N, 2), 顶点顺序为环形
    # inters, info = find_intersection_points(coords, center, radius)
    # 如果没有或只有一个交点，仍可以直接筛顶点（全部在圆内或全部在圆外）
    # 为稳健，这里不依赖交点，直接按顶点过滤
    cx, cy = center
    R2 = radius * radius

    def inside(pt):
        dx = pt[0] - cx
        dy = pt[1] - cy
        return dx * dx + dy * dy <= R2 + 1e-9

    n = len(coords)
    if n == 0:
        return np.empty((0, 2))

    # 直接筛选所有在圆内的顶点
    inside_mask = np.array([inside(coords[i]) for i in range(n)])
    res = coords[inside_mask]

    # 去重（防止输入有重复顶点或数值近似重复）
    if len(res) > 1:
        keep = [True]
        for k in range(1, len(res)):
            keep.append(not np.allclose(res[k], res[k - 1]))
        res = res[np.array(keep)]

    return res


def points_on_polygon_within_circle2(coords, center, radius):
    """
    返回在圆内的顶点索引列表。
    - coords: np.ndarray of shape (N, 2)，顶点顺序为环形
    - center: (cx, cy)
    - radius: float
    """
    cx, cy = center
    R2 = radius * radius

    def inside(pt):
        dx = pt[0] - cx
        dy = pt[1] - cy
        return dx * dx + dy * dy <= R2 + 1e-9

    n = len(coords)
    if n == 0:
        return np.empty((0,), dtype=int)

    # 标记每个点是否在圆内
    inside_mask = np.array([inside(coords[i]) for i in range(n)])
    idxs = np.nonzero(inside_mask)[0]  # 在圆内的索引

    # 去重：防止输入有重复顶点或数值近似重复导致相邻索引对应的点近似相同
    if len(idxs) > 1:
        keep = [True]
        for k in range(1, len(idxs)):
            i_prev = idxs[k - 1]
            i_curr = idxs[k]
            keep.append(not np.allclose(coords[i_curr], coords[i_prev]))
        idxs = idxs[np.array(keep)]

    return idxs


def _segment_lengths(coords: np.ndarray, idxs: list[int]) -> float:
    # 连续索引的折线长度累加
    if len(idxs) < 2:
        return 0.0
    diffs = coords[np.array(idxs[1:])] - coords[np.array(idxs[:-1])]
    return float(np.linalg.norm(diffs, axis=1).sum())


def _path_indices(pre: int, i: int, nxt: int, n: int, closed: bool = False) -> list[int]:
    # 生成 pre→...→i→...→nxt 的索引序列
    # 假定原始折线为自然序，若 closed=True 则允许环绕
    seq = []
    if not closed:
        # 要求 pre <= i <= nxt 或 nxt <= i <= pre
        if pre <= i <= nxt:
            seq = list(range(pre, i + 1)) + list(range(i, nxt + 1))
        elif nxt <= i <= pre:
            seq = list(range(nxt, i + 1)) + list(range(i, pre + 1))
        else:
            # 若不在同一方向，退化为直接相邻段连接
            left = sorted([pre, i, nxt])
            seq = list(range(left[0], left[1] + 1)) + list(range(left[1], left[2] + 1))
    else:
        # 闭合情况：选择两侧路径中较短的一侧到 i，再到另一侧
        def ring_range(a, b):
            if a <= b:
                return list(range(a, b + 1))
            else:
                return list(range(a, n)) + list(range(0, b + 1))

        # 选 pre→i 与 i→nxt 的环路径
        seq = ring_range(pre, i) + ring_range(i, nxt)
    # 去重合并（i 会重复一次）
    if len(seq) >= 2:
        dedup = [seq[0]]
        for k in seq[1:]:
            if k != dedup[-1]:
                dedup.append(k)
        return dedup
    return seq


def _point_line_perp_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    w = p - a
    norm_v = np.linalg.norm(v)
    if norm_v == 0.0:
        return float(np.linalg.norm(p - a))
    # 2D 叉积标量的绝对值等于平行四边形面积
    cross = v[0] * w[1] - v[1] * w[0]
    return float(abs(cross) / norm_v)


def dis_chord_length(coords: np.ndarray):
    # 垂比弦，by邓敏
    n = len(coords)
    radius = find_support_radius(coords)
    results = []
    for i in range(n):
        # 每个点
        c = coords[i]
        pts_in_circle = points_on_polygon_within_circle(coords, c, radius)
        pre, nxt = find_boundary_indices(coords, pts_in_circle)  # pre，next是两个序号，代表coords上点的序号

        # 计算 pre 号到 i 号到 next 号这些点串在 coords 的长度之和
        idx_seq = _path_indices(pre, i, nxt, n, closed=True)
        length_sum = _segment_lengths(coords, idx_seq)

        # 计算 i 号点到 pre 号点和 next 号点的垂距（到弦的垂距），以及端点直线距离
        d_perp = _point_line_perp_dist(coords[i], coords[pre], coords[nxt])
        dis_chrod = d_perp / length_sum

        # results.append({
        # 	"index": i,
        # 	"pre": pre,
        # 	"next": nxt,
        # 	"length_sum": length_sum,
        # 	"d_perp": d_perp,
        # 	"dis_chord": dis_chrod
        # })
        results.append(dis_chrod)
    return results


def find_boundary_indices2(n, idxs):
    """
    根据给定点总数 n 和索引列表 idxs，返回 idxs 范围前后的一个索引（环绕式）。
    - 原始点序号为 0 到 n-1
    - 若 idxs 为空则抛出异常
    - 若 idxs 含一个或多个索引，视其最小值到最大值为范围，
      返回 (min_idx - 1) % n 和 (max_idx + 1) % n
    例如：
      n=6, idxs=[0] -> 返回 (5, 1)
    """
    # 基本校验
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须为正整数")
    if idxs is None or len(idxs) == 0:
        raise ValueError("idxs 为空，无法计算边界")
    # 校验每个索引合法
    for x in idxs:
        # if not isinstance(x, int):
        # 	raise ValueError("idxs 中存在非整数索引")
        if x < 0 or x >= n:
            raise ValueError(f"索引 {x} 越界，应在 [0, {n - 1}]")

    # 取范围的最小与最大
    min_idx = min(idxs)
    max_idx = max(idxs)

    # 计算前后边界（环绕）
    prev_idx = (min_idx - 1) % n
    next_idx = (max_idx + 1) % n

    return prev_idx, next_idx


def dis_chord_length2(coords: np.ndarray):
    # 垂比弦，by邓敏
    n = len(coords)
    radius = find_support_radius(coords)
    results = []
    for i in range(n):
        # 每个点
        c = coords[i]
        pts_in_circle_idx = points_on_polygon_within_circle2(coords, c, radius)
        pre, nxt = find_boundary_indices2(n, pts_in_circle_idx)  # pre，next是两个序号，代表coords上点的序号
        pre = int(pre)
        nxt = int(nxt)
        # 计算 pre 号到 i 号到 next 号这些点串在 coords 的长度之和
        idx_seq = _path_indices(pre, i, nxt, n, closed=True)
        length_sum = _segment_lengths(coords, idx_seq)

        # 计算 i 号点到 pre 号点和 next 号点的垂距（到弦的垂距），以及端点直线距离
        d_perp = _point_line_perp_dist(coords[i], coords[pre], coords[nxt])
        dis_chrod = d_perp / (length_sum + 1e-9)

        # results.append({
        # 	"index": i,
        # 	"pre": pre,
        # 	"next": nxt,
        # 	"length_sum": length_sum,
        # 	"d_perp": d_perp,
        # 	"dis_chord": dis_chrod
        # })
        results.append(dis_chrod)
    return results


def polygon_angles_at_i(coords: np.ndarray, i: int, fill_nan=True) -> float:
    """
    计算多边形轮廓中每个中间点 pi 的向量 (pi-1->pi) 与 (pi-1->pi+1) 的夹角，范围 0..180 度。

    参数：
        coords: 形状 (n, 2) 的 ndarray，每行 [x, y]，首尾不重合（不闭合）。
        fill_nan: 是否返回长度 n 的数组，前两个和最后一个填 NaN。
                  若 False，则只返回 i=1..n-2 的角度数组，长度 n-2。

    返回：
        angles: ndarray，单位为度。
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 3:
        raise ValueError("需要至少 3 个点")
    if not (0 <= i <= n - 1):
        raise ValueError("i 必须在 [1, n-2] 之间，避免越界（可自行扩展为闭合处理）")

    p_im1 = coords[(i - 1) % n]
    p_i = coords[i]
    p_ip1 = coords[(i + 1) % n]

    a = p_i - p_im1  # p_{i-1} -> p_i
    b = p_ip1 - p_im1  # p_{i-1} -> p_{i+1}

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan  # 任一边为零向量则角度不可定义

    cos_theta = np.dot(a, b) / (na * nb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # return float(np.degrees(np.arccos(cos_theta)))
    return float(np.arccos(cos_theta))


def polygon_side_angles(coords):
    n = len(coords)
    angles = []
    for i in range(n):
        angles.append(polygon_angles_at_i(coords, i))
    return angles


# endregion-------------------------------------------------------------------------------------------


def edge_axis_angles_from_obb(coords: np.ndarray):
    """
    输入:
      - coords: (N,2)，多边形顶点，建议逆时针
    返回:
      - angles: (N,) 每条边与 OBB 长轴的最小夹角，范围 [0, pi/2]
    """
    # 1) 通过 OBB 获取长轴方向 u0
    obb = OBBOject(coords.tolist())  # OBB 类
    u0 = np.asarray(obb.u0, dtype=float)
    u0 /= (np.linalg.norm(u0) + 1e-12)  # 归一化以防万一

    # 2) 计算每条边向量
    N = len(coords)
    next_idx = (np.arange(N) + 1) % N
    edges = coords[next_idx] - coords
    elen = np.linalg.norm(edges, axis=1, keepdims=True) + 1e-12
    ehat = edges / elen

    # 3) 与 u0 的夹角（最小夹角）
    dot = ehat @ u0  # (N,)
    cross = ehat[:, 0] * u0[1] - ehat[:, 1] * u0[0]
    theta = np.arctan2(np.abs(cross), dot)  # [0, pi]
    theta = np.where(theta > np.pi - theta, np.pi - theta, theta)  # 压到 [0, pi/2]
    return theta


def _signed_area(coords: np.ndarray) -> float:
    """
    计算多边形的有向面积：
    > 0 表示逆时针 (CCW)
    < 0 表示顺时针 (CW)
    == 0 可能共线或退化
    """
    # 去除重复的闭环终点（若存在）
    # if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
    # 	pts = coords[:-1]
    # else:
    # 	pts = coords

    if len(coords) < 3:
        return 0.0

    x = coords[:, 0]
    y = coords[:, 1]
    # 使用 Shoelace formula 的一条向量化形式
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def build_base_features2(coords: np.ndarray):
    """
    输入：coords: (N,2)
    返回: (N,11)
    """
    N = len(coords)
    area = _signed_area(coords)
    if area < 0:  # 顺时针，翻转
        coords = coords[::-1].copy()

    x = coords[:, 0]
    y = coords[:, 1]

    # 邻接索引（环）
    prev_idx = (np.arange(N) - 1) % N
    next_idx = (np.arange(N) + 1) % N

    prev_vec = coords - coords[prev_idx]
    next_vec = coords[next_idx] - coords

    prev_len = np.linalg.norm(prev_vec, axis=1) + 1e-9
    next_len = np.linalg.norm(next_vec, axis=1) + 1e-9

    # 周长（用 next_len 或 prev_len 求和都可以）
    perimeter = float(np.sum(next_len)) + 1e-12

    # 新特征：每点取较大边长与周长的比值
    len_divide_by_perimeter = np.maximum(prev_len, next_len) / perimeter
    # 新特征，坐标和凸包之间的距离。（规则四角矩形的时候，效果好）
    hull = graham_scan_convex_hull(coords)
    dis_to_convx = distances_points_to_convex_hull_edges(coords, hull)
    # 转角（用有符号外转角）
    # 叉积 + 点积
    cross = prev_vec[:, 0] * next_vec[:, 1] - prev_vec[:, 1] * next_vec[:, 0]
    dot = (prev_vec * next_vec).sum(axis=1)
    turn_angle = np.arctan2(cross, dot)  # [-pi,pi]

    # 从 OBB 获取长轴方向，并计算每条边与长轴的夹角
    edge_axis_angles = edge_axis_angles_from_obb(coords)  # (N,)

    eps = 1e-12
    cross_sign = np.where(cross > eps, 1.0, np.where(cross < -eps, -1.0, 0.0))

    feats = np.stack([
        x, y,
        prev_vec[:, 0], prev_vec[:, 1],
        next_vec[:, 0], next_vec[:, 1],
        prev_len,
        next_len,
        turn_angle,
        edge_axis_angles,
        cross_sign,
        dis_to_convx,
        len_divide_by_perimeter
    ], axis=1).astype(np.float32)
    return feats


def build_base_features3(coords: np.ndarray):
    """
    输入：coords: (N,2)
    返回: (N,11)
    """
    N = len(coords)
    area = _signed_area(coords)
    if area < 0:  # 顺时针，翻转
        coords = coords[::-1].copy()

    x = coords[:, 0]
    y = coords[:, 1]

    # 邻接索引（环）
    prev_idx = (np.arange(N) - 1) % N
    next_idx = (np.arange(N) + 1) % N

    prev_vec = coords - coords[prev_idx]
    next_vec = coords[next_idx] - coords

    prev_len = np.linalg.norm(prev_vec, axis=1) + 1e-9
    next_len = np.linalg.norm(next_vec, axis=1) + 1e-9

    # 周长
    perimeter = float(np.sum(next_len)) + 1e-12

    # 每点取较大边长与周长的比值
    len_divide_by_perimeter = np.maximum(prev_len, next_len) / perimeter

    # 转角（用有符号外转角）
    # 叉积 + 点积
    cross = prev_vec[:, 0] * next_vec[:, 1] - prev_vec[:, 1] * next_vec[:, 0]
    dot = (prev_vec * next_vec).sum(axis=1)
    turn_angle = np.arctan2(cross, dot)  # [-pi,pi]

    # 从 OBB 获取长轴方向，并计算每条边与长轴的夹角
    edge_axis_angles = edge_axis_angles_from_obb(coords)  # (N,)

    eps = 1e-12

    feats = np.stack([
        x, y,
        prev_vec[:, 0], prev_vec[:, 1],
        next_vec[:, 0], next_vec[:, 1],
        prev_len,
        next_len,
        turn_angle,
        edge_axis_angles,
        len_divide_by_perimeter
    ], axis=1).astype(np.float32)
    return feats


# region--------------高斯滤波多尺度特征，跳边----
import numpy as np
from scipy.ndimage import gaussian_filter1d


# —— 工具：环形索引 —— #
def prev_idx(i, N): return (i - 1) % N


def next_idx(i, N): return (i + 1) % N


# —— 环形高斯平滑 —— #
def gaussian_smooth_circular(arr: np.ndarray, sigma: float
                             ) -> np.ndarray:
    """
    对一维或二维序列做环形高斯平滑：
    - 若 arr 形状为 (N,2) 表示坐标，对两个通道分别平滑
    - mode='wrap' 确保闭合多边形的环形边界条件
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return gaussian_filter1d(arr, sigma=sigma, mode='wrap')
    elif arr.ndim == 2 and arr.shape[1] == 2:
        x = gaussian_filter1d(arr[:, 0], sigma=sigma, mode='wrap')
        y = gaussian_filter1d(arr[:, 1], sigma=sigma, mode='wrap')
        return np.stack([x, y], axis=1)
    else:
        raise ValueError("arr shape must be (N,) or (N,2).")


# —— 基于三点的转角与（有符号/无符号）曲率近似 —— #
def recompute_turn_angle_and_curvature(coords_sigma: np.ndarray,
                                       signed_curv: bool = True,
                                       eps: float = 1e-12):
    """
    输入：平滑后的坐标 (N,2)
    输出：
      - turn_angle: (N,) ∈ [-pi, pi]，外转角（与您 base 特征一致的定义）
      - kappa:      (N,) 有符号或无符号曲率近似
    曲率近似：用三点法的外接圆半径 R ≈ |AB|·|BC|·|CA| / (4·|Δ|)，kappa=1/R；
             符号用 cross(prev_vec, next_vec) 的符号。
    """
    N = len(coords_sigma)
    idx = np.arange(N)
    i_prev = (idx - 1) % N
    i_next = (idx + 1) % N

    prev_vec = coords_sigma - coords_sigma[i_prev]  # v_{i-1->i}
    next_vec = coords_sigma[i_next] - coords_sigma  # v_{i->i+1}

    # 转角：atan2(cross, dot)
    cross = prev_vec[:, 0] * next_vec[:, 1] - prev_vec[:, 1] * next_vec[:, 0]
    dot = (prev_vec * next_vec).sum(axis=1)
    turn_angle = np.arctan2(cross, dot)  # [-pi, pi]

    # 曲率：用三点确定的三边长度与有向面积计算外接圆半径
    a = np.linalg.norm(prev_vec, axis=1) + eps
    b = np.linalg.norm(next_vec, axis=1) + eps
    # 第三边 c = |p_{i+1} - p_{i-1}|
    c = np.linalg.norm(coords_sigma[i_next] - coords_sigma[i_prev], axis=1) + eps
    # 三角形面积 2Δ = |cross(p_i - p_{i-1}, p_{i+1} - p_{i-1})|
    cross_big = (coords_sigma - coords_sigma[i_prev])[:, 0] * \
                (coords_sigma[i_next] - coords_sigma[i_prev])[:, 1] - \
                (coords_sigma - coords_sigma[i_prev])[:, 1] * \
                (coords_sigma[i_next] - coords_sigma[i_prev])[:, 0]
    area2 = np.abs(cross_big) + eps  # 2 * |Δ|

    R = (a * b * c) / (2.0 * area2 + eps)  # R ≈ abc / (4Δ)；这里用 area2=2Δ，因此分母 2*area2
    kappa = 1.0 / (R + eps)

    if signed_curv:
        sign = np.sign(cross)  # 与 turn_angle 符号一致
        kappa = kappa * sign
    return turn_angle.astype(np.float32), kappa.astype(np.float32)


# —— 多尺度派生特征构造 —— #
def build_multi_scale_addons(coords: np.ndarray,
                             sigmas=(1.0, 2.0, 3.0),
                             signed_curv: bool = True) -> np.ndarray:
    """
    输入：coords (N,2)
    输出：multi_feats (N, d_multi)
      例如：每个 σ 输出 [turn_angle_σ, curvature_σ] 共 2*len(sigmas) 维；
            再加稳定度票数 stable_votes 与稳定强度 stable_strength，共 2 维。
      d_multi = 2*len(sigmas) + 2
    """
    N = len(coords)
    per_sigma_feats = []
    curv_stack = []

    for s in sigmas:
        coords_s = gaussian_smooth_circular(coords, sigma=s)
        ta_s, kappa_s = recompute_turn_angle_and_curvature(coords_s, signed_curv=signed_curv)
        per_sigma_feats.append(np.stack([ta_s, kappa_s], axis=1))  # (N,2)
        curv_stack.append(kappa_s[:, None])

    # 叠成 (N, num_scales)
    curv_mat = np.concatenate(curv_stack, axis=1)
    # 跨尺度“极值稳定度”（简单版）：该点在所有尺度上都是绝对曲率的局部极值 → 记1，否则0
    stable_votes = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        is_extreme_all = True
        for s_i in range(curv_mat.shape[1]):
            left = curv_mat[(i - 1) % N, s_i]
            mid = curv_mat[i, s_i]
            right = curv_mat[(i + 1) % N, s_i]
            if not (abs(mid) >= abs(left) and abs(mid) >= abs(right)):
                is_extreme_all = False
                break
        stable_votes[i] = 1.0 if is_extreme_all else 0.0

    # 稳定强度：三尺度绝对曲率的平均
    stable_strength = np.mean(np.abs(curv_mat), axis=1).astype(np.float32)

    per_sigma_feats = np.concatenate(per_sigma_feats, axis=1)  # (N, 2*len(sigmas))
    multi_feats = np.concatenate([
        per_sigma_feats,
        stable_votes[:, None],
        stable_strength[:, None]
    ], axis=1).astype(np.float32)
    return multi_feats


# —— 将多尺度特征拼到已有的 feats14 —— #
def build_polygon_features_with_chacts3_multiscale(coords: np.ndarray,
                                                   sigmas=(0.5, 1.0, 2.0),
                                                   signed_curv: bool = True) -> np.ndarray:
    """
    在原始的 14 维特征基础上，追加多尺度角度/曲率/稳定度：
      feats14: (N,14) <-  build_polygon_features_with_chacts3
      addons : (N, d_multi) <- 2*len(sigmas) + 2
      feats  : (N, 14 + d_multi)
    """
    feats14 = build_polygon_features_with_chacts3(coords)
    multi_addons = build_multi_scale_addons(coords, sigmas=sigmas, signed_curv=signed_curv)
    feats = np.concatenate([feats14.astype(np.float32), multi_addons], axis=1)
    return feats


def build_polygon_features_with_no_multiscale(coords: np.ndarray,
                                              sigmas=(1.0, 2.0, 3.0),
                                              signed_curv: bool = True) -> np.ndarray:
    """
    不要多尺度特征叠加
      feats14: (N,14) <-  build_polygon_features_with_chacts3
      addons : (N, d_multi) <- 2*len(sigmas) + 2
      feats  : (N, 14 + d_multi)
    """
    feats14 = build_polygon_features_with_chacts3(coords)

    return feats14


# endregion------------------------------------------------


# region------------------跳边-----------


def cat_batched_edges(edge_index_np_list, B: int, N: int, device=None):
    """
    将每个样本的 edge_index（numpy 数组，形状 (2, E_i)）加偏移后拼接为全局 edge_index（torch.Tensor）。
    假设每个样本的节点数相同，均为 N；样本数为 B。

    参数
    - edge_index_np_list: 长度为 B 的列表，每个元素是 numpy 数组 (2, E_i)，表示该样本的边
    - B: 批大小
    - N: 每个样本的节点数
    - device: 返回张量的设备（'cpu' 或 'cuda'）
    返回
    - edge_index: torch.LongTensor, 形状 (2, sum_i E_i)
    """
    assert len(edge_index_np_list) == B, f"edge_index_np_list 长度 {len(edge_index_np_list)} 与 B={B} 不一致"
    edges = []
    for b in range(B):
        e_b = offset_edges_for_batch(edge_index_np_list[b], b, N, device=device)  # (2, E_i)
        edges.append(e_b)
    return torch.cat(edges, dim=1) if len(edges) > 0 else torch.zeros(2, 0, dtype=torch.long, device=device)


def build_ring_edges_numpy(N: int):
    src = np.arange(N, dtype=np.int64)
    dst = (src + 1) % N
    e = np.stack([src, dst], axis=0)
    e_rev = np.stack([dst, src], axis=0)
    return np.concatenate([e, e_rev], axis=1)  # (2,2N)


def build_skip_edges_numpy(N: int, hops=(2, 3)):
    src_list, dst_list = [], []
    for i in range(N):
        for h in hops:
            j1 = (i + h) % N
            j2 = (i - h) % N
            src_list += [i, i];
            dst_list += [j1, j2]
    e = np.stack([np.array(src_list), np.array(dst_list)], axis=0)
    e_rev = e[::-1]
    return np.concatenate([e, e_rev], axis=1)  # (2, E_skip*2)


# 批内偏移示意（PyTorch）
def offset_edges_for_batch(edge_index_np: np.ndarray, b: int, N: int, device=None):
    # 将单多边形边索引偏移到第 b 个样本（顶点索引范围 [b*N, (b+1)*N)）
    e = torch.from_numpy(edge_index_np.copy())
    e = e + b * N
    return e.to(device) if device is not None else e


# endregion----------------------------------


def curvature_by_turn_angle(p1, p2, p3, signed=False, eps=1e-12):
    """
    采用“转角除以弧长”近似曲率：
      |kappa| ≈ 2*sin(theta/2) / (|a|+|b|)
    其中 theta 为向量 a=p2-p1 与 b=p3-p2 的夹角。
    - 更适合采样较均匀或需要与转角直接关联的场景。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    a = p2 - p1
    b = p3 - p2
    L1 = np.hypot(a[0], a[1])
    L2 = np.hypot(b[0], b[1])

    # 点积与叉积
    dot = a[0] * b[0] + a[1] * b[1]
    cross = a[0] * b[1] - a[1] * b[0]

    # 夹角 theta = atan2(|cross|, dot)；保留符号用 atan2(cross, dot)
    theta = np.arctan2(cross, dot) if signed else np.arctan2(abs(cross), dot)

    # 2*sin(theta/2) 数值稳定计算：2*sin(t/2) = sqrt(2 - 2*cos t)
    # 直接用 sin(theta/2) 也可以
    # 这里直接用 sin，以简洁为主
    s = np.sin(0.5 * theta)
    denom = (L1 + L2) + eps
    kappa = 2.0 * s / denom

    # 对于无符号模式，已使用 |cross| 近似无符号 theta；若严格无符号，可再取绝对值
    if not signed:
        kappa = abs(kappa)
    return kappa


def curvature_polyline2(points, signed=True, closed=True, eps=1e-12):
    """
    对点序列计算每个中间点的三点曲率。
    - points: array-like, shape (N,2)
    - signed: 是否返回有符号曲率
    - closed: 是否按闭合环处理（首尾相接）
    返回:
      kappa: np.ndarray, shape (N,), 对于非闭合开曲线，端点返回0
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)
    if N < 3:
        return np.zeros(N, dtype=float)

    kappa = np.zeros(N, dtype=float)

    if closed:
        # 环索引
        for i in range(N):
            i_prev = (i - 1) % N
            i_next = (i + 1) % N
            kappa[i] = curvature_by_turn_angle(pts[i_prev], pts[i], pts[i_next],
                                               signed=signed, eps=eps)
    else:
        # 开曲线：两端设为0，中间计算
        for i in range(1, N - 1):
            kappa[i] = curvature_by_turn_angle(pts[i - 1], pts[i], pts[i + 1],
                                               signed=signed, eps=eps)
        kappa[0] = 0.0
        kappa[-1] = 0.0

    return kappa


def build_polygon_features_with_chacts(coords: np.ndarray,
                                       kappa_near_zero: float = 1e-3):
    """
    输入一个多边形坐标coords (N,2)
    输出:
        feats12: (N,12)  前11维=基础特征, 最后一维 曲率
    """

    N = len(coords)
    base13 = build_base_features2(coords)  # (N,11)
    if base13.shape[0] != N:
        print(f"[WARNING] base13.shape[0] ({base13.shape[0]}) != "
              f"coords.shape[0] ({coords.shape[0]})")

    # kappa值是曲率
    # kappa = curvature_polyline2(coords, signed=False, eps=1e-12)

    kappa = curvature_polyline2(coords, signed=False, eps=1e-12)
    kappa = np.asarray(kappa, dtype=np.float32)
    if kappa.ndim == 1:
        kappa = kappa[:, None]  # (N,1)
    if kappa.shape[0] != N:
        print(f"[WARNING] kappa.shape[0] ({kappa.shape[0]}) !="
              f" coords.shape[0] ({coords.shape[0]})")

    feats14 = np.concatenate([base13, kappa], axis=1).astype(np.float32)
    return feats14


def build_polygon_features_with_chacts2(coords: np.ndarray,
                                        kappa_near_zero: float = 1e-3):
    """
    输入一个多边形坐标coords (N,2)
    输出:
        feats16: (N,16)
    """

    N = len(coords)

    base13 = build_base_features2(coords)  # (N,11)
    dis_chord = dis_chord_length2(coords)
    side_angles = polygon_side_angles(coords)
    # 将 list 转为 (N,1)
    dis_chord = np.asarray(dis_chord, dtype=np.float32)
    if dis_chord.ndim == 1:
        dis_chord = dis_chord[:, None]  # (N,1)
    if dis_chord.shape[0] != N:
        print(f"[WARNING] dis_chord.shape[0] ({dis_chord.shape[0]}) != coords.shape[0] ({coords.shape[0]})")

    side_angles = np.asarray(side_angles, dtype=np.float32)
    if side_angles.ndim == 1:
        side_angles = side_angles[:, None]  # (N,1)
    if side_angles.shape[0] != N:
        print(f"[WARNING] side_angles.shape[0] ({side_angles.shape[0]}) != coords.shape[0] ({coords.shape[0]})")

    if base13.shape[0] != N:
        print(f"[WARNING] base13.shape[0] ({base13.shape[0]}) != "
              f"coords.shape[0] ({coords.shape[0]})")

    # kappa值是曲率
    # kappa = curvature_polyline2(coords, signed=False, eps=1e-12)

    kappa = curvature_polyline2(coords, signed=False, eps=1e-12)
    kappa = np.asarray(kappa, dtype=np.float32)
    if kappa.ndim == 1:
        kappa = kappa[:, None]  # (N,1)
    if kappa.shape[0] != N:
        print(f"[WARNING] kappa.shape[0] ({kappa.shape[0]}) !="
              f" coords.shape[0] ({coords.shape[0]})")

    feats16 = np.concatenate([base13.astype(np.float32),
                              kappa,
                              dis_chord,
                              side_angles], axis=1)
    return feats16


def build_polygon_features_with_chacts3(coords: np.ndarray,
                                        kappa_near_zero: float = 1e-3):
    """
    输入一个多边形坐标coords (N,2)
    输出:
        feats14: (N,14)
    """

    N = len(coords)

    base11 = build_base_features3(coords)  # (N,11)
    dis_chord = dis_chord_length2(coords)
    side_angles = polygon_side_angles(coords)
    # 将 list 转为 (N,1)
    dis_chord = np.asarray(dis_chord, dtype=np.float32)
    if dis_chord.ndim == 1:
        dis_chord = dis_chord[:, None]  # (N,1)
    if dis_chord.shape[0] != N:
        print(f"[WARNING] dis_chord.shape[0] ({dis_chord.shape[0]}) != coords.shape[0] ({coords.shape[0]})")

    side_angles = np.asarray(side_angles, dtype=np.float32)
    if side_angles.ndim == 1:
        side_angles = side_angles[:, None]  # (N,1)
    if side_angles.shape[0] != N:
        print(f"[WARNING] side_angles.shape[0] ({side_angles.shape[0]}) != coords.shape[0] ({coords.shape[0]})")

    if base11.shape[0] != N:
        print(f"[WARNING] base13.shape[0] ({base11.shape[0]}) != "
              f"coords.shape[0] ({coords.shape[0]})")

    # kappa值是曲率
    # kappa = curvature_polyline2(coords, signed=False, eps=1e-12)

    kappa = curvature_polyline2(coords, signed=False, eps=1e-12)
    kappa = np.asarray(kappa, dtype=np.float32)
    if kappa.ndim == 1:
        kappa = kappa[:, None]  # (N,1)
    if kappa.shape[0] != N:
        print(f"[WARNING] kappa.shape[0] ({kappa.shape[0]}) !="
              f" coords.shape[0] ({coords.shape[0]})")

    feats14 = np.concatenate([base11.astype(np.float32),
                              kappa,
                              dis_chord,
                              side_angles], axis=1)
    return feats14


def load_training_polygons_and_points(
        polygon_file: str = r"F:\LandUseBoudary\data\SimPos\cleaned_polygons.shp",
        points_geojson: str = r"F:\LandUseBoudary\code\points_keep.geojson",
        id_col_poly: str = "building_id",
        id_col_pt: str = "building_id",
        idx_col_pt: str = "idx",
        keep_col_pt: str = "keep",
        id_min: int = 0,
        id_max: int = 100,
        target_crs: Optional[str] = None,
        in_dim: int = 12,
        use_only_coords: bool = True
) -> List[Dict[str, Any]]:
    """
    返回 list[dict]，每个元素代表一个多边形：
      {
        "building_id": int/str,
        "coords": Tensor (Ni,2),   # 按 idx 排序
        "feats": Tensor (Ni,F) or None,
        "keep": Tensor (Ni,)
      }
    - 仅保留 building_id ∈ [id_min, id_max] 的样本
    - 点来自 points_keep.geojson，作为训练标签来源
    - 可选坐标投影 target_crs
    """
    assert os.path.isfile(polygon_file), f"Polygon file not found: {polygon_file}"
    assert os.path.isfile(points_geojson), f"Point file not found: {points_geojson}"

    gdf_poly = gpd.read_file(polygon_file)
    gdf_pts = gpd.read_file(points_geojson)

    # 统一坐标系（可选）
    if target_crs is not None:
        if gdf_poly.crs is not None and gdf_poly.crs.to_string() != target_crs:
            gdf_poly = gdf_poly.to_crs(target_crs)
        if gdf_pts.crs is not None and gdf_pts.crs.to_string() != target_crs:
            gdf_pts = gdf_pts.to_crs(target_crs)

    # 过滤 building_id 范围
    gdf_poly = gdf_poly[(gdf_poly[id_col_poly] >= id_min) & (gdf_poly[id_col_poly] <= id_max)].copy()
    gdf_pts = gdf_pts[(gdf_pts[id_col_pt] >= id_min) & (gdf_pts[id_col_pt] <= id_max)].copy()

    # 仅保留必要列，去掉 geometry 为 None 的点或面
    gdf_poly = gdf_poly[~gdf_poly.geometry.isna()].copy()
    gdf_pts = gdf_pts[~gdf_pts.geometry.isna()].copy()

    # 将点按 building_id -> list 排序聚合
    polys: List[Dict[str, Any]] = []

    # 为了更鲁棒：先按 building_id 分组点
    pt_groups = gdf_pts.groupby(id_col_pt)

    for _, row in gdf_poly.iterrows():
        bid = row[id_col_poly]

        # 该 building_id 的点集
        if bid not in pt_groups.groups:
            continue  # 没有点的多边形，跳过（训练需要点标签）

        df_pts = pt_groups.get_group(bid).copy()

        # 若没有 idx 列或有缺失，尝试补齐顺序
        if idx_col_pt not in df_pts.columns or df_pts[idx_col_pt].isna().any():
            # 按空间顺序给一个简单索引（例如直接 range），或根据多边形边界最近点排序
            # 简化处理：直接按几何序号排序
            df_pts = df_pts.reset_index(drop=True)
            df_pts[idx_col_pt] = np.arange(len(df_pts), dtype=np.int64)

        # 排序：按 idx 升序
        df_pts = df_pts.sort_values(by=idx_col_pt).reset_index(drop=True)
        # 标签 keep 提取（缺失填0）

        keep = df_pts[keep_col_pt].fillna(0).to_numpy(dtype=np.float32)

        # 坐标提取
        xs = df_pts.geometry.x.to_numpy(dtype=np.float32)
        ys = df_pts.geometry.y.to_numpy(dtype=np.float32)
        coords = np.stack([xs, ys], axis=1)  # (N,2)
        # 去闭合处理：若首尾相同，去掉最后一个点
        if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]  # 使后续 feats 与 coords 一致
            keep = keep[:-1]

        # 特征准备：若无额外特征，则后续 Dataset 用坐标构造（use_only_coords=True）
        feats = build_polygon_features_with_chacts(coords)
        # if not use_only_coords:
        # 	# 这里如果你已有外部顶点特征，按 idx 对齐后填入 feats 为 (N, in_dim)
        # 	# 先放 None，由 Dataset 再次构造
        # 	feats = None

        polys.append({
            "building_id": int(bid) if np.issubdtype(type(bid), np.integer) else bid,
            "coords": torch.from_numpy(coords),  # (N,2)
            "feats": None if feats is None else torch.from_numpy(feats).float(),
            "keep": torch.from_numpy(keep)  # (N,)
        })

    # 过滤：至少3个点
    # polys = [p for p in polys if p["coords"].shape[0] >= 3]

    # 归一化可选：通常在 Dataset 内做按多边形的局部归一化，也可以这里做
    # 此处不做，保持原值，交由前面的 PolygonVertexDataset 构造特征

    return polys


def _norm_single_polygon(coords: np.ndarray, mode: str = 'minmax'):
    if mode is None:
        return coords, None
    if mode == 'minmax':
        xy_min = coords.min(axis=0)  # (2,)
        xy_max = coords.max(axis=0)  # (2,)
        span = np.maximum(xy_max - xy_min, 1e-12)
        coords_norm = (coords - xy_min) / span
        norm_params = {
            'mode': 'minmax',
            'min': xy_min.astype(np.float64),
            'max': xy_max.astype(np.float64)
        }
        return coords_norm, norm_params
    else:
        # 可扩展其他模式
        return coords, None


def check_points_on_polygon_edges(
        polygon_file: str,
        points_geojson: str,
        id_field: str = "building_id",
        idx_field: str = "idx",
        epsilon: float = 1e-2,  # 容差，可按数据单位调整
        use_touches: bool = False  # True 则用 touches 严格判定；False 用 boundary 距离+epsilon
):
    # 读取数据
    poly_gdf = gpd.read_file(polygon_file)
    pts_gdf = gpd.read_file(points_geojson)

    # 字段检查
    if id_field not in poly_gdf.columns:
        raise ValueError(f"多边形文件缺少字段: {id_field}")
    if id_field not in pts_gdf.columns:
        raise ValueError(f"点文件缺少字段: {id_field}")
    if idx_field not in pts_gdf.columns:
        raise ValueError(f"点文件缺少字段: {idx_field}")

    # CRS 对齐：将点投影到多边形 CRS
    # if poly_gdf.crs and pts_gdf.crs and poly_gdf.crs != pts_gdf.crs:
    # 	pts_gdf = pts_gdf.to_crs(poly_gdf.crs)

    # building_id -> polygon（若同 id 多个面，合并为 MultiPolygon）
    poly_map = defaultdict(list)
    for _, row in poly_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        poly_map[row[id_field]].append(geom)

    # 将同一 id 的多个面合并（union 或直接构造 MultiPolygon 都可；
    # 这里直接用 unary_union 更健壮，避免相邻/重叠面导致误差）
    from shapely.ops import unary_union
    union_map = {}
    for bid, geoms in poly_map.items():
        try:
            union_map[bid] = unary_union(geoms)
        except Exception:
            # 回退：若合并失败，取几何集合（后面对 touches 时 shapely 也能处理）
            union_map[bid] = unary_union([g for g in geoms if g.is_valid])

    not_on_boundary = []  # (building_id, idx)

    # 分组检查
    for bid, group in pts_gdf.groupby(id_field):
        polygon = union_map.get(bid, None)
        if polygon is None or polygon.is_empty:
            # 面缺失，则该组点全部记为异常
            for _, prow in group.iterrows():
                not_on_boundary.append((bid, prow[idx_field]))
            continue

        # 逐点判断
        for _, prow in group.iterrows():
            pt = prow.geometry
            if pt is None or pt.is_empty:
                not_on_boundary.append((bid, prow[idx_field]))
                continue

            try:
                if use_touches:
                    # 严格在边界上：pt.touches(polygon)
                    ok = pt.touches(polygon)
                else:
                    # 稳健判断：到边界距离 <= epsilon
                    d = pt.distance(polygon.boundary)
                    ok = (d <= epsilon)
            except Exception:
                ok = False

            if not ok:
                not_on_boundary.append((bid, prow[idx_field]))

    # 输出
    if not_on_boundary:
        print("以下点不在对应多边形的轮廓上：")
        for bid, idxv in not_on_boundary:
            print(f"building_id={bid}, idx={idxv}")
    else:
        print("所有点均在对应多边形轮廓上。")


# if __name__ == "__main__":
#     polygon_file: str = r"F:\LandUseBoudary\data\SimPos\cleaned_polygons.shp"
#     points_geojson: str = r"F:\LandUseBoudary\code\points_keep.geojson"
#
#     check_points_on_polygon_edges(
#         polygon_file=polygon_file,
#         points_geojson=points_geojson,
#         id_field="building_id",
#         idx_field="idx",
#         epsilon=1e-7,       # 根据数据单位调参
#         use_touches=False   # 若你希望严格几何定义，改为 True
#     )

# 数据装载
def load_training_polygons_and_points2(
        polygon_file: str = r"F:\LandUseBoudary\data\SimPos\cleaned_polygons.shp",
        points_geojson: str = r"F:\LandUseBoudary\code\points_keep.geojson",
        id_col_poly: str = "building_id",
        id_col_pt: str = "building_id",
        idx_col_pt: str = "idx",
        keep_col_pt: str = "keep",
        id_min: int = 0,
        id_max: int = 100,
        target_crs: Optional[str] = None,
        in_dim: int = 12,
        use_only_coords: bool = True,
        per_polygon_norm: Optional[str] = 'minmax'  # 新增：每多边形归一化模式
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    返回:
      - polys: list[dict]，每个元素代表一个多边形：
          {
            "building_id": int/str,
            "coords": Tensor (Ni,2),   # 已按 idx 排序，且已做 per-polygon 归一化
            "feats": Tensor (Ni,F) or None,  # 特征基于归一化后的 coords 构造
            "keep": Tensor (Ni,)
          }
      - building_ids: list，与 polys 一一对应
      - norm_params_list: list[dict]，每个元素对应一个多边形的归一化参数
            {'mode':'minmax','min':(2,), 'max':(2,)} 或 {'mode': None}
    """
    assert os.path.isfile(polygon_file), f"Polygon file not found: {polygon_file}"
    assert os.path.isfile(points_geojson), f"Point file not found: {points_geojson}"

    gdf_poly = gpd.read_file(polygon_file)
    gdf_pts = gpd.read_file(points_geojson)

    # 坐标系统一（可选）
    if target_crs is not None:
        if gdf_poly.crs is not None and gdf_poly.crs.to_string() != target_crs:
            gdf_poly = gdf_poly.to_crs(target_crs)
        if gdf_pts.crs is not None and gdf_pts.crs.to_string() != target_crs:
            gdf_pts = gdf_pts.to_crs(target_crs)

    # 过滤 building_id 范围
    gdf_poly = gdf_poly[(gdf_poly[id_col_poly] >= id_min) &
                        (gdf_poly[id_col_poly] <= id_max)].copy()
    gdf_pts = gdf_pts[(gdf_pts[id_col_pt] >= id_min) &
                      (gdf_pts[id_col_pt] <= id_max)].copy()

    # 去除 geometry 缺失
    gdf_poly = gdf_poly[~gdf_poly.geometry.isna()].copy()
    gdf_pts = gdf_pts[~gdf_pts.geometry.isna()].copy()

    # 分组点
    pt_groups = gdf_pts.groupby(id_col_pt)

    polys: List[Dict[str, Any]] = []
    building_ids: List[Any] = []
    norm_params_list: List[Dict[str, Any]] = []

    for _, row in gdf_poly.iterrows():
        bid = row[id_col_poly]

        # 该 building_id 的点集
        if bid not in pt_groups.groups:
            continue  # 无点，跳过

        df_pts = pt_groups.get_group(bid).copy()

        # 索引补齐
        if idx_col_pt not in df_pts.columns or df_pts[idx_col_pt].isna().any():
            df_pts = df_pts.reset_index(drop=True)
            df_pts[idx_col_pt] = np.arange(len(df_pts), dtype=np.int64)

        # 排序
        df_pts = df_pts.sort_values(by=idx_col_pt).reset_index(drop=True)

        # 标签 keep
        keep = df_pts[keep_col_pt].fillna(0).to_numpy(dtype=np.float32)

        # 坐标提取
        xs = df_pts.geometry.x.to_numpy(dtype=np.float64)  # 用 float64 先算，归一后转 float32
        ys = df_pts.geometry.y.to_numpy(dtype=np.float64)
        coords = np.stack([xs, ys], axis=1)  # (N,2)

        # 1116改动，此处不需要去闭合！！！45号建筑物就是受此影响，变成了3个点
        # # 去闭合：若首尾相同去掉最后一点
        # if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        # 	coords = coords[:-1]
        # 	keep = keep[:-1]

        # 在 per-polygon 范围内归一化 coords（minmax）
        coords_norm, norm_params = _norm_single_polygon(coords, mode=per_polygon_norm)
        coords_norm = coords_norm.astype(np.float32)

        # 基于归一化后的坐标构建特征
        # 确保它兼容 float32 输入
        feats = build_polygon_features_with_chacts2(coords_norm)  # shape: (N, 12)
        if feats is not None:
            feats = feats.astype(np.float32)

        polys.append({
            "building_id": int(bid) if (isinstance(bid, (np.integer, int))) else bid,
            "coords": torch.from_numpy(coords_norm),  # (N,2) 归一化后
            "feats": None if feats is None else torch.from_numpy(feats).float(),
            "keep": torch.from_numpy(keep)  # (N,)
        })
        building_ids.append(int(bid) if (isinstance(bid, (np.integer, int))) else bid)
        norm_params_list.append(norm_params if norm_params is not None else {'mode': None})
    # ===== 新增检查：在返回前检查每个元素的 coords 点数 Ni 是否小于 4 =====
    for p in polys:
        bid = p.get("building_id", "<unknown>")
        coords_tensor = p.get("coords", None)
        try:
            n_pts = int(coords_tensor.shape[0]) if coords_tensor is not None else 0
        except Exception:
            n_pts = 0
        if n_pts < 4:
            print(f"{bid}坐标数目小于4，警告!")
    # ===== 检查结束 =====
    return polys, building_ids, norm_params_list


def load_training_polygons_and_points3(
        polygon_file: str = r"F:\LandUseBoudary\data\SimPos\cleaned_polygons.shp",
        points_geojson: str = r"F:\LandUseBoudary\code\points_keep.geojson",
        id_col_poly: str = "building_id",
        id_col_pt: str = "building_id",
        idx_col_pt: str = "idx",
        keep_col_pt: str = "keep",
        id_min: int = 0,
        id_max: int = 100,
        target_crs: Optional[str] = None,
        in_dim: int = 12,
        use_only_coords: bool = True,
        per_polygon_norm: Optional[str] = 'minmax'  # 新增：每多边形归一化模式
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    返回:
      - polys: list[dict]，每个元素代表一个多边形：
          {
            "building_id": int/str,
            "coords": Tensor (Ni,2),   # 已按 idx 排序，且已做 per-polygon 归一化
            "feats": Tensor (Ni,F) or None,  # 特征基于归一化后的 coords 构造
            "keep": Tensor (Ni,)
          }
      - building_ids: list，与 polys 一一对应
      - norm_params_list: list[dict]，每个元素对应一个多边形的归一化参数
            {'mode':'minmax','min':(2,), 'max':(2,)} 或 {'mode': None}
    """
    assert os.path.isfile(polygon_file), f"Polygon file not found: {polygon_file}"
    assert os.path.isfile(points_geojson), f"Point file not found: {points_geojson}"

    gdf_poly = gpd.read_file(polygon_file)
    gdf_pts = gpd.read_file(points_geojson)

    # 坐标系统一（可选）
    if target_crs is not None:
        if gdf_poly.crs is not None and gdf_poly.crs.to_string() != target_crs:
            gdf_poly = gdf_poly.to_crs(target_crs)
        if gdf_pts.crs is not None and gdf_pts.crs.to_string() != target_crs:
            gdf_pts = gdf_pts.to_crs(target_crs)

    # 过滤 building_id 范围
    gdf_poly = gdf_poly[(gdf_poly[id_col_poly] >= id_min) &
                        (gdf_poly[id_col_poly] <= id_max)].copy()
    gdf_pts = gdf_pts[(gdf_pts[id_col_pt] >= id_min) &
                      (gdf_pts[id_col_pt] <= id_max)].copy()

    # 去除 geometry 缺失
    gdf_poly = gdf_poly[~gdf_poly.geometry.isna()].copy()
    gdf_pts = gdf_pts[~gdf_pts.geometry.isna()].copy()

    # 分组点
    pt_groups = gdf_pts.groupby(id_col_pt)

    polys: List[Dict[str, Any]] = []
    building_ids: List[Any] = []
    norm_params_list: List[Dict[str, Any]] = []

    for _, row in gdf_poly.iterrows():
        bid = row[id_col_poly]

        # 该 building_id 的点集
        if bid not in pt_groups.groups:
            continue  # 无点，跳过

        df_pts = pt_groups.get_group(bid).copy()

        # 索引补齐
        if idx_col_pt not in df_pts.columns or df_pts[idx_col_pt].isna().any():
            df_pts = df_pts.reset_index(drop=True)
            df_pts[idx_col_pt] = np.arange(len(df_pts), dtype=np.int64)

        # 排序
        df_pts = df_pts.sort_values(by=idx_col_pt).reset_index(drop=True)

        # 标签 keep
        keep = df_pts[keep_col_pt].fillna(0).to_numpy(dtype=np.float32)

        # 坐标提取
        xs = df_pts.geometry.x.to_numpy(dtype=np.float64)  # 用 float64 先算，归一后转 float32
        ys = df_pts.geometry.y.to_numpy(dtype=np.float64)
        coords = np.stack([xs, ys], axis=1)  # (N,2)

        # 1116改动，此处不需要去闭合！！！45号建筑物就是受此影响，变成了3个点
        # # 去闭合：若首尾相同去掉最后一点
        # if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        # 	coords = coords[:-1]
        # 	keep = keep[:-1]

        # 在 per-polygon 范围内归一化 coords（minmax）
        coords_norm, norm_params = _norm_single_polygon(coords, mode=per_polygon_norm)
        coords_norm = coords_norm.astype(np.float32)

        # 基于归一化后的坐标构建特征
        # 确保它兼容 float32 输入
        # feats = build_polygon_features_with_chacts3(coords_norm)  # shape: (N, 12)
        feats = build_polygon_features_with_chacts3_multiscale(coords_norm)

        if feats is not None:
            feats = feats.astype(np.float32)

        polys.append({
            "building_id": int(bid) if (isinstance(bid, (np.integer, int))) else bid,
            "coords": torch.from_numpy(coords_norm),  # (N,2) 归一化后
            "feats": None if feats is None else torch.from_numpy(feats).float(),
            "keep": torch.from_numpy(keep)  # (N,)
        })
        building_ids.append(int(bid) if (isinstance(bid, (np.integer, int))) else bid)
        norm_params_list.append(norm_params if norm_params is not None else {'mode': None})
    # ===== 新增检查：在返回前检查每个元素的 coords 点数 Ni 是否小于 4 =====
    for p in polys:
        bid = p.get("building_id", "<unknown>")
        coords_tensor = p.get("coords", None)
        try:
            n_pts = int(coords_tensor.shape[0]) if coords_tensor is not None else 0
        except Exception:
            n_pts = 0
        if n_pts < 4:
            print(f"{bid}坐标数目小于4，警告!")
    # ===== 检查结束 =====
    return polys, building_ids, norm_params_list


def load_training_polygons_and_points4_no_poly(
        points_geojson: str,
        id_col_pt: str = "building_id",
        idx_col_pt: str = "idx",
        keep_col_pt: str = "keep",
        id_min: Optional[int] = None,  # 若为 None，则自动取 0
        id_max: Optional[int] = None,  # 若为 None，则自动取 points 中的最大 building_id
        target_crs: Optional[str] = None,
        per_polygon_norm: Optional[str] = 'minmax'  # 每多边形归一化模式
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    仅基于 points_geojson 构造训练多边形数据：
      - 自动以 points 的 building_id 分组，范围为 [0, max_building_id]（可被 id_min/id_max 覆盖）
      - 返回与原 load_training_polygons_and_points3 相同的三元组
    返回:
      - polys: list[dict]，每个元素代表一个多边形：
          {
            "building_id": int/str,
            "coords": Tensor (Ni,2),   # 已按 idx 排序，且已做 per-polygon 归一化
            "feats": Tensor (Ni,F) or None,  # 特征基于归一化后的 coords 构造
            "keep": Tensor (Ni,)
          }
      - building_ids: list，与 polys 一一对应
      - norm_params_list: list[dict]，每个元素对应一个多边形的归一化参数
            {'mode':'minmax','min':(2,), 'max':(2,)} 或 {'mode': None}
    """
    assert os.path.isfile(points_geojson), f"Point file not found: {points_geojson}"
    gdf_pts = gpd.read_file(points_geojson)

    # 坐标系统一（可选，仅对点数据）
    if target_crs is not None and gdf_pts.crs is not None:
        if gdf_pts.crs.to_string() != target_crs:
            gdf_pts = gdf_pts.to_crs(target_crs)

    # 去除 geometry 缺失
    gdf_pts = gdf_pts[~gdf_pts.geometry.isna()].copy()

    # 自动确定 id_min/id_max：从 points 中读取 building_id 的最大值
    if id_col_pt not in gdf_pts.columns:
        raise ValueError(f"'{id_col_pt}' not found in points file columns: {list(gdf_pts.columns)}")

    # 过滤出有效的 building_id（去除 NaN）
    valid_id_series = gdf_pts[id_col_pt].dropna()
    if valid_id_series.empty:
        raise ValueError("No valid building_id found in points file.")

    max_bid_in_pts = int(valid_id_series.max())
    auto_id_min = 0 if id_min is None else id_min
    auto_id_max = max_bid_in_pts if id_max is None else id_max

    # 过滤 building_id 范围 [auto_id_min, auto_id_max]
    # gdf_pts = gdf_pts[(gdf_pts[id_col_pt] >= auto_id_min) &
    #                   (gdf_pts[id_col_pt] <= auto_id_max)].copy()

    # 再次确认非空
    if gdf_pts.empty:
        raise ValueError(f"No points after filtering building_id in [{auto_id_min}, {auto_id_max}].")

    # 分组点
    pt_groups = gdf_pts.groupby(id_col_pt)

    polys: List[Dict[str, Any]] = []
    building_ids: List[Any] = []
    norm_params_list: List[Dict[str, Any]] = []

    # 遍历所有 building_id 分组
    for bid, df_pts in pt_groups:
        # 索引补齐
        if idx_col_pt not in df_pts.columns or df_pts[idx_col_pt].isna().any():
            df_pts = df_pts.reset_index(drop=True)
            df_pts[idx_col_pt] = np.arange(len(df_pts), dtype=np.int64)

        # 排序
        df_pts = df_pts.sort_values(by=idx_col_pt).reset_index(drop=True)

        # 标签 keep（若无则默认0）
        if keep_col_pt in df_pts.columns:
            keep = df_pts[keep_col_pt].fillna(0).to_numpy(dtype=np.float32)
        else:
            keep = np.zeros(len(df_pts), dtype=np.float32)

        # 坐标提取（float64计算、归一后转float32）
        xs = df_pts.geometry.x.to_numpy(dtype=np.float64)
        ys = df_pts.geometry.y.to_numpy(dtype=np.float64)
        coords = np.stack([xs, ys], axis=1)  # (N,2)

        # 1116改动说明保持：不去掉重复闭合点
        # 若你后续希望统一去重，可恢复这段逻辑：
        # if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        #     coords = coords[:-1]
        #     keep = keep[:-1]

        # 在 per-polygon 范围内归一化 coords（minmax / None）
        coords_norm, norm_params = _norm_single_polygon(coords, mode=per_polygon_norm)
        coords_norm = coords_norm.astype(np.float32)

        # 基于归一化后的坐标构建特征（与你现有的构造函数保持一致）
        # 0320改进，高斯滤波里面的sigma改成了[0.5,1,2]，区别之前的[1,2,3]
        feats = build_polygon_features_with_chacts3_multiscale(coords_norm)
        if feats is not None:
            feats = feats.astype(np.float32)

        polys.append({
            "building_id": int(bid) if isinstance(bid, (np.integer, int)) else bid,
            "coords": torch.from_numpy(coords_norm),  # (N,2)
            "feats": None if feats is None else torch.from_numpy(feats).float(),
            "keep": torch.from_numpy(keep)  # (N,)
        })
        building_ids.append(int(bid) if isinstance(bid, (np.integer, int)) else bid)
        norm_params_list.append(norm_params if norm_params is not None else {'mode': None})

    # 返回前检查每个元素的 coords 点数 Ni 是否小于 4
    for p in polys:
        bid = p.get("building_id", "<unknown>")
        coords_tensor = p.get("coords", None)
        try:
            n_pts = int(coords_tensor.shape[0]) if coords_tensor is not None else 0
        except Exception:
            n_pts = 0
        if n_pts < 4:
            print(f"{bid} 坐标数目小于4，警告!")

    return polys, building_ids, norm_params_list


def load_training_polygons_and_points4_no_poly0324(
        points_geojson: str,
        id_col_pt: str = "building_id",
        idx_col_pt: str = "idx",
        keep_col_pt: str = "keep",
        id_min: Optional[int] = None,  # 若为 None，则自动取 0
        id_max: Optional[int] = None,  # 若为 None，则自动取 points 中的最大 building_id
        target_crs: Optional[str] = None,
        per_polygon_norm: Optional[str] = 'minmax'  # 每多边形归一化模式
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    仅基于 points_geojson 构造训练多边形数据：不要高斯平滑的特征
      - 自动以 points 的 building_id 分组，范围为 [0, max_building_id]（可被 id_min/id_max 覆盖）
      - 返回与原 load_training_polygons_and_points3 相同的三元组
    返回:
      - polys: list[dict]，每个元素代表一个多边形：
          {
            "building_id": int/str,
            "coords": Tensor (Ni,2),   # 已按 idx 排序，且已做 per-polygon 归一化
            "feats": Tensor (Ni,F) or None,  # 特征基于归一化后的 coords 构造
            "keep": Tensor (Ni,)
          }
      - building_ids: list，与 polys 一一对应
      - norm_params_list: list[dict]，每个元素对应一个多边形的归一化参数
            {'mode':'minmax','min':(2,), 'max':(2,)} 或 {'mode': None}
    """
    assert os.path.isfile(points_geojson), f"Point file not found: {points_geojson}"
    gdf_pts = gpd.read_file(points_geojson)

    # 坐标系统一（可选，仅对点数据）
    if target_crs is not None and gdf_pts.crs is not None:
        if gdf_pts.crs.to_string() != target_crs:
            gdf_pts = gdf_pts.to_crs(target_crs)

    # 去除 geometry 缺失
    gdf_pts = gdf_pts[~gdf_pts.geometry.isna()].copy()

    # 自动确定 id_min/id_max：从 points 中读取 building_id 的最大值
    if id_col_pt not in gdf_pts.columns:
        raise ValueError(f"'{id_col_pt}' not found in points file columns: {list(gdf_pts.columns)}")

    # 过滤出有效的 building_id（去除 NaN）
    valid_id_series = gdf_pts[id_col_pt].dropna()
    if valid_id_series.empty:
        raise ValueError("No valid building_id found in points file.")

    max_bid_in_pts = int(valid_id_series.max())
    auto_id_min = 0 if id_min is None else id_min
    auto_id_max = max_bid_in_pts if id_max is None else id_max

    # 过滤 building_id 范围 [auto_id_min, auto_id_max]
    # gdf_pts = gdf_pts[(gdf_pts[id_col_pt] >= auto_id_min) &
    #                   (gdf_pts[id_col_pt] <= auto_id_max)].copy()

    # 再次确认非空
    if gdf_pts.empty:
        raise ValueError(f"No points after filtering building_id in [{auto_id_min}, {auto_id_max}].")

    # 分组点
    pt_groups = gdf_pts.groupby(id_col_pt)

    polys: List[Dict[str, Any]] = []
    building_ids: List[Any] = []
    norm_params_list: List[Dict[str, Any]] = []

    # 遍历所有 building_id 分组
    for bid, df_pts in pt_groups:
        # 索引补齐
        if idx_col_pt not in df_pts.columns or df_pts[idx_col_pt].isna().any():
            df_pts = df_pts.reset_index(drop=True)
            df_pts[idx_col_pt] = np.arange(len(df_pts), dtype=np.int64)

        # 排序
        df_pts = df_pts.sort_values(by=idx_col_pt).reset_index(drop=True)

        # 标签 keep（若无则默认0）
        if keep_col_pt in df_pts.columns:
            keep = df_pts[keep_col_pt].fillna(0).to_numpy(dtype=np.float32)
        else:
            keep = np.zeros(len(df_pts), dtype=np.float32)

        # 坐标提取（float64计算、归一后转float32）
        xs = df_pts.geometry.x.to_numpy(dtype=np.float64)
        ys = df_pts.geometry.y.to_numpy(dtype=np.float64)
        coords = np.stack([xs, ys], axis=1)  # (N,2)

        # 1116改动说明保持：不去掉重复闭合点
        # 若你后续希望统一去重，可恢复这段逻辑：
        # if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        #     coords = coords[:-1]
        #     keep = keep[:-1]

        # 在 per-polygon 范围内归一化 coords（minmax / None）
        coords_norm, norm_params = _norm_single_polygon(coords, mode=per_polygon_norm)
        coords_norm = coords_norm.astype(np.float32)

        # 基于归一化后的坐标构建特征（与你现有的构造函数保持一致）
        # 0320改进，高斯滤波里面的sigma改成了[0.5,1,2]，区别之前的[1,2,3]
        feats = build_polygon_features_with_no_multiscale(coords_norm)
        if feats is not None:
            feats = feats.astype(np.float32)

        polys.append({
            "building_id": int(bid) if isinstance(bid, (np.integer, int)) else bid,
            "coords": torch.from_numpy(coords_norm),  # (N,2)
            "feats": None if feats is None else torch.from_numpy(feats).float(),
            "keep": torch.from_numpy(keep)  # (N,)
        })
        building_ids.append(int(bid) if isinstance(bid, (np.integer, int)) else bid)
        norm_params_list.append(norm_params if norm_params is not None else {'mode': None})

    # 返回前检查每个元素的 coords 点数 Ni 是否小于 4
    for p in polys:
        bid = p.get("building_id", "<unknown>")
        coords_tensor = p.get("coords", None)
        try:
            n_pts = int(coords_tensor.shape[0]) if coords_tensor is not None else 0
        except Exception:
            n_pts = 0
        if n_pts < 4:
            print(f"{bid} 坐标数目小于4，警告!")

    return polys, building_ids, norm_params_list
