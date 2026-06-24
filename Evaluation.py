import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, Point

from shapely.ops import unary_union
from shapely.validation import make_valid  # shapely>=2.0


def hausdorff_distance(A, B):
    """
    A: numpy array of shape (m, d)
    B: numpy array of shape (n, d)
    returns: Hausdorff distance (float)
    """
    # 距离矩阵 (m, n)：A中每点到B中每点的欧氏距离
    # 为了数值稳定和效率，可先算平方距离再开方
    # dist^2 = ||A||^2 + ||B||^2 - 2 A B^T
    # 但这里直接用逐差更直观
    diff = A[:, None, :] - B[None, :, :]
    D = np.linalg.norm(diff, axis=2)  # (m, n)

    # A到B的定向Hausdorff：每个a到B的最近距离，取最大
    h_AB = np.max(np.min(D, axis=1))

    # B到A的定向Hausdorff：每个b到A的最近距离，取最大
    h_BA = np.max(np.min(D, axis=0))

    return max(h_AB, h_BA)


def polygon_area(coords):
    """
    coords: numpy array of shape (n, 2) 或可转成此形状的列表
            顶点按边界顺序排列，首尾可不重复（函数内部按环处理）
    返回：绝对面积（float）
    """
    pts = np.asarray(coords, dtype=float)
    n = len(pts)
    if n < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    # 环相邻索引
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    signed_area = 0.5 * np.sum(x * y_next - x_next * y)
    return abs(signed_area)


def area_ratio_B_over_A(A, B):
    """
    A, B: numpy arrays/lists of shape (m,2), (n,2)
    返回：B的面积 / A的面积
    """
    area_A = polygon_area(A)
    area_B = polygon_area(B)
    if area_A == 0.0:
        raise ValueError("A的面积为0，无法计算面积之比（分母为0）。请检查A是否为有效多边形。")
    return area_B / area_A


def polygon_from_coords(coords):
    """
    将点序列转换为 shapely Polygon。
    - coords: [(x,y), ...] 顶点按边界顺序给出，首尾可不重复。
    - 若不足3个点，返回 None。
    """
    if coords is None or len(coords) < 3:
        return None
    poly = Polygon(coords)
    # 修复可能的自交/无效情况
    if not poly.is_valid:
        poly = make_valid(poly)
        # make_valid 可能返回 MultiPolygon/GeometryCollection
        # 取面积最大的面作为代表（也可按需保留全部）
        if poly.geom_type == "GeometryCollection":
            # 过滤出面
            polys = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if len(polys) == 0:
                return None
            poly = unary_union(polys)
        if poly.geom_type == "MultiPolygon":
            # 取面积最大
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty:
            return None
    return poly


def intersection_union_iou(A_coords, B_coords):
    """
    输入：
      - A_coords, B_coords: 与你之前相同格式的二维点数组/列表
    输出：
      - inter_area: 交集面积
      - union_area: 并集面积
      - iou: 交并比 = inter_area / union_area（若并集为0，则返回0或抛错）
    """
    A_poly = polygon_from_coords(A_coords)
    B_poly = polygon_from_coords(B_coords)

    if A_poly is None or B_poly is None:
        raise ValueError("A 或 B 不是有效多边形（点数不足或几何无效）")

    # 交、并
    inter = A_poly.intersection(B_poly)
    union = A_poly.union(B_poly)

    inter_area = inter.area if not inter.is_empty else 0.0
    union_area = union.area if not union.is_empty else 0.0

    if union_area == 0.0:
        # 两个多边形都为空或退化到面积为0的情形
        iou = 0.0
    else:
        iou = inter_area / union_area

    return iou


def area_ratio(A_coords, B_coords):
    """
    输入：
      - A_coords, B_coords: 与你之前相同格式的二维点数组/列表
    输出：

    """
    A_poly = polygon_from_coords(A_coords)
    B_poly = polygon_from_coords(B_coords)

    ratio = A_poly.area / B_poly.area

    return ratio


def area_change(A_coords, B_coords):
    """
    输入：
      - A_coords, B_coords: 与你之前相同格式的二维点数组/列表
    输出：

    """
    A_poly = polygon_from_coords(A_coords)
    B_poly = polygon_from_coords(B_coords)
    if A_poly is None or B_poly is None:
        return polygon_area(A_coords) + polygon_area(B_coords)
    change = math.fabs(A_poly.area - B_poly.area)

    return change


# # 示例用法
# A = [(0.0,0.0), (2.0,0.0), (2.0,1.0), (0.0,1.0)]
# B = [(1.0,0.5), (3.0,0.5), (3.0,1.5), (1.0,1.5)]
#
# inter_area, union_area, iou = intersection_union_iou(A, B)
# print("交集面积:", inter_area)
# print("并集面积:", union_area)
# print("交并比 IoU:", iou)


def calculate_orthogonality(points):
    """
    计算简化后点集的正交性 (Orthogonality, OR) 指标。

    参数:
        points (np.array): 二维数组，形状为 (n, 2)，每行表示一个点的 (x, y) 坐标。

    返回:
        float: 正交性 (OR) 指标，值范围为 [0, 1]。
    """
    # 计算点与点之间的向量
    vectors = points - np.roll(points, shift=-1, axis=0)

    # 计算向量之间的角度
    angles = []
    for i in range(len(vectors)):
        v1 = vectors[i]
        v2 = vectors[(i + 1) % len(vectors)]
        # 计算向量夹角
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 限制cos值在[-1, 1]范围内
        angles.append(angle)

    # 计算正交性 OR
    n = len(angles)
    orthogonality = (2 / (n * np.pi)) * np.sum(np.abs(np.array(angles) - np.pi / 2))

    return orthogonality


# region---------------------------------计算turning function函数----------------------------------


def calculate_turning_function(coords):
    """
    计算多边形每条边的夹角和归一化边长，并绘制 Turning Function 图。

    参数:
        coords (np.ndarray): 二维数组，形状为 (n, 2)，每行表示一个点的 (x, y) 坐标。

    返回:
        None
    """
    # 确保多边形闭合
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    # 计算每条边的向量
    vectors = coords[1:] - coords[:-1]

    # 计算每条边的长度
    edge_lengths = np.linalg.norm(vectors, axis=1)
    total_perimeter = np.sum(edge_lengths)  # 总周长
    normalized_lengths = edge_lengths / total_perimeter  # 归一化边长

    # 计算每条边相对于 x 轴正方向的夹角（范围 0 到 360 度）
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = np.mod(angles, 360)  # 将角度限制在 [0, 360)

    # 构造 Turning Function 数据
    cumulative_lengths = np.cumsum(normalized_lengths)  # 累积归一化长度
    cumulative_lengths = np.insert(cumulative_lengths, 0, 0)  # 起点为 0
    turning_function = np.cumsum(np.insert(angles, 0, 0))  # 累积角度
    return angles, cumulative_lengths  # y轴数据，x轴数据


def calculate_turning_function2(coords):
    # 确保闭合
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    # 边向量与长度
    vectors = coords[1:] - coords[:-1]  # shape: (n, 2)
    edge_lengths = np.linalg.norm(vectors, axis=1)  # (n,)
    perim = edge_lengths.sum()
    norm_lengths = edge_lengths / (perim + 1e-12)

    # 每条边的方向角 [0, 360)
    edge_dirs = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    edge_dirs = np.mod(edge_dirs, 360.0)

    cum_s = np.concatenate([[0.0], np.cumsum(norm_lengths)])  # 长度 n+1（便于画阶梯）
    cum_theta = np.concatenate([[0.0], np.cumsum(edge_dirs)])
    return cum_theta, cum_s


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def calculate_shape_context(fid, coords_A, coords_B):
    """
    计算两个多边形的 Turning Function 之间的 Shape Context (SC) 距离。

    参数:
        f_ori (np.ndarray): 原始多边形的 Turning Function (y 轴数据)。
        s_ori (np.ndarray): 原始多边形的归一化边长 (x 轴数据)。
        f_sim (np.ndarray): 比较多边形的 Turning Function (y 轴数据)。
        s_sim (np.ndarray): 比较多边形的归一化边长 (x 轴数据)。

    返回:
        float: Shape Context (SC) 距离。
    """
    f_ori, s_ori = calculate_turning_function2(coords_A)
    f_sim, s_sim = calculate_turning_function2(coords_B)

    # 确保两个函数的 s 范围一致，使用插值对齐
    interp_ori = interp1d(s_ori, f_ori, kind='linear', fill_value="extrapolate")
    interp_sim = interp1d(s_sim, f_sim, kind='linear', fill_value="extrapolate")

    def sc_error(params):
        theta, t = params
        shifted_ori = interp_ori(np.mod(s_ori + t, 1)) + theta  # 平移和旋转
        diff = shifted_ori - interp_sim(s_ori)
        return np.trapz(diff ** 2, s_ori)  # 计算积分 (误差平方和)

    # 优化 theta 和 t 使误差最小
    result = minimize(sc_error, x0=[0, 0], bounds=[(-360, 360), (-1, 1)])
    sc_min = result.fun  # 最小误差

    # 计算最终的 SC 值
    sc = (1 / (2 * np.pi)) * np.sqrt(sc_min)
    # print(f"The {fid} building's Turning Function is {sc}")
    return sc


#
# # 示例多边形的 Turning Function 数据
# f_ori = np.array([0, 90, 180, 270, 360])  # 原始多边形角度
# s_ori = np.array([0, 0.25, 0.5, 0.75, 1])  # 原始多边形归一化边长
#
# f_sim = np.array([0, 100, 190, 280, 370])  # 比较多边形角度
# s_sim = np.array([0, 0.25, 0.5, 0.75, 1])  # 比较多边形归一化边长
#
# # 计算 Shape Context 距离
# sc = calculate_shape_context(f_ori, s_ori, f_sim, s_sim)
# print(f"Shape Context (SC) 距离: {sc}")


# endregion----------------------------------------------------

def show_turning_function(fid, coords_A, coords_B):
    # 计算两个多边形的 Turning Function 数据
    turning_function_A, cumulative_lengths_A = calculate_turning_function2(coords_A)
    turning_function_B, cumulative_lengths_B = calculate_turning_function2(coords_B)

    # 绘制阶梯图
    plt.figure(figsize=(8, 4))
    plt.step(cumulative_lengths_B, turning_function_B, where='post', color='green', label="Polygon B")
    plt.step(cumulative_lengths_A, turning_function_A, where='post', color='red', label="Polygon A")

    plt.xlabel("Normalized Edge Length (s)")
    plt.ylabel("Turning Function (degrees)")
    plt.title(f"The {fid} th building's Turning Function Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def pts_from_gdf(gdf, id_col='building_id', idx_col='idx'):
    result = {}
    for bid, grp in gdf.groupby(id_col, sort=False):
        grp_sorted = grp.sort_values(idx_col, kind='mergesort')
        pts = np.array([[geom.x, geom.y] for geom in grp_sorted.geometry], dtype=float)
        result[bid] = pts
    return result


def pts_from_gdf2(gdf, id_col='building_id', idx_col='idx', keep_col='keep_pred'):
    """
    从GeoDataFrame按id_col分组、按idx_col稳定排序后，筛选keep_col==1的点，
    返回 {building_id: np.ndarray(N, 2)} 的字典。
    """
    if keep_col not in gdf.columns:
        raise KeyError(f"缺少列: {keep_col}")
    result = {}
    for bid, grp in gdf.groupby(id_col, sort=False):
        # 稳定排序，保持与原逻辑一致
        grp_sorted = grp.sort_values(idx_col, kind='mergesort')
        # 仅保留 keep_pred == 1 的行
        grp_kept = grp_sorted[grp_sorted[keep_col] == 1]
        # 将几何点转为 ndarray
        pts = np.array([[geom.x, geom.y] for geom in grp_kept.geometry], dtype=float)
        result[bid] = pts
    return result


def count_buildings_over_20_closed(coords_org, max_num, tol=1e-9):
    def effective_len(a):
        if a.shape[0] >= 2 and np.allclose(a[0], a[-1], atol=tol, rtol=0):
            return a.shape[0] - 1
        return a.shape[0]

    cnt = 0
    for arr in coords_org.values():
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 2:
            if effective_len(arr) > max_num:
                cnt += 1
    return cnt


def evaluation_train_dataset(file_org):
    gdf_org = gpd.read_file(file_org)
    # gdf_hand_labelling = gpd.read_file(file_hand_labelling)
    coords_org = pts_from_gdf(gdf_org)
    max_num_pts = 50
    num_max_than_10 = count_buildings_over_20_closed(coords_org, max_num_pts)
    print(f'原始数据集里面超过{max_num_pts}个的有{num_max_than_10}个')


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    """
    计算点 p 到线段 ab 的欧氏距离。
    """
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    px, py = p.x, p.y

    # 向量
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay

    # 线段长度平方
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        # a 与 b 重合，返回 p 到 a 的距离
        return np.hypot(px - ax, py - ay)

    # 投影参数 t ∈ [0,1]
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))

    # 最近点坐标
    cx = ax + t * abx
    cy = ay + t * aby
    return np.hypot(px - cx, py - cy)


def cyclic_indices_between(n: int, i: int, j: int):
    """
    在环长度 n 上，从 i 到 j（不含端点 i、j）按顺时针方向的索引列表。
    例如 n=8, i=2, j=6 -> [3,4,5]
         n=8, i=6, j=2 -> [7,0,1]
    """
    res = []
    k = (i + 1) % n
    while k != j:
        res.append(k)
        k = (k + 1) % n
    return res


def max_mean_perpendicular_distance(gdf_sim: "GeoDataFrame") -> float:
    """
    对每个 building_id 的轮廓环，找相邻特征点对之间的非特征点到该两点连线的平均垂距，
    返回所有这些平均值中的最大值。
    """
    # 确保必要列存在
    required_cols = {"building_id", "idx", "keep_pred", "geometry"}
    missing = required_cols - set(gdf_sim.columns)
    if missing:
        raise ValueError(f"gdf_sim 缺少必要列: {missing}")

    # 按 building_id 分组处理
    max_mean = -np.inf

    for b_id, df_b in gdf_sim.groupby("building_id"):
        # 按 idx 排序，构成环
        df_b = df_b.sort_values("idx").reset_index(drop=True)

        # 建立索引到行位置的映射，和便于环访问的列表
        idx_list = df_b["idx"].tolist()
        n = len(idx_list)
        if n < 2:
            continue

        # 统一建立按环顺序的数组
        keep = df_b["keep_pred"].to_numpy()
        points = df_b["geometry"].to_numpy()

        # 找出特征点的环索引位置（在 0..n-1 的位置）
        feat_positions = [pos for pos in range(n) if int(keep[pos]) == 1]
        if len(feat_positions) < 2:
            # 没有相邻特征点对
            continue

        # 将特征点位置按环顺序配对相邻：pos[k] 与 pos[(k+1)%m]
        m = len(feat_positions)
        for k in range(m):
            i = feat_positions[k]
            j = feat_positions[(k + 1) % m]

            # 取两特征点之间（沿环顺时针）所有中间索引
            mid_positions = cyclic_indices_between(n, i, j)
            if not mid_positions:
                # 没有非特征点
                continue

            # 过滤出非特征点
            nonfeat_positions = [p for p in mid_positions if int(keep[p]) == 0]
            if not nonfeat_positions:
                continue

            A = points[i]
            B = points[j]
            # 计算这些非特征点到 AB 线段的距离
            dists = [point_to_segment_distance(points[p], A, B) for p in nonfeat_positions]
            mean_dist = float(np.mean(dists))

            if mean_dist > max_mean:
                max_mean = mean_dist

    # 若所有建筑都无有效间隔，返回 NaN 或者抛错，这里返回 NaN 更友好
    if max_mean == -np.inf:
        return float("nan")
    return max_mean


def max_mean_perpendicular_distance_stats(gdf_sim: "GeoDataFrame"):
    """
    对每个 building_id 的轮廓环：
      - 找相邻特征点对之间的非特征点到该两点连线的平均垂距 mean_dist
      - 对该建筑物内所有 mean_dist 取最大值，记为该建筑的 max_mean
    收集所有建筑的 max_mean，返回它们的平均值和中值，并附带每个建筑的 max_mean 字典。

    返回:
      {
        "avg": float,     # 所有建筑 max_mean 的平均值
        "median": float,  # 所有建筑 max_mean 的中值
        "per_building_max": dict  # {building_id: max_mean}
      }
    若某建筑没有有效间隔，则不纳入统计。若全局都无有效值，avg 和 median 为 NaN。
    """
    # 确保必要列存在
    required_cols = {"building_id", "idx", "keep_pred", "geometry"}
    missing = required_cols - set(gdf_sim.columns)
    if missing:
        raise ValueError(f"gdf_sim 缺少必要列: {missing}")

    per_building_max = {}  # 收集每个建筑的最大 mean_dist

    # 按 building_id 分组处理
    for b_id, df_b in gdf_sim.groupby("building_id"):
        # 按 idx 排序，构成环
        df_b = df_b.sort_values("idx").reset_index(drop=True)

        n = len(df_b)
        if n < 2:
            continue

        # 统一建立按环顺序的数组
        keep = df_b["keep_pred"].to_numpy()
        points = df_b["geometry"].to_numpy()

        # 找出特征点的环索引位置（在 0..n-1 的位置）
        feat_positions = [pos for pos in range(n) if int(keep[pos]) == 1]
        if len(feat_positions) < 2:
            # 没有相邻特征点对
            continue

        # 在该建筑内收集所有 mean_dist
        building_mean_dists = []

        # 将特征点位置按环顺序配对相邻：pos[k] 与 pos[(k+1)%m]
        m = len(feat_positions)
        for k in range(m):
            i = feat_positions[k]
            j = feat_positions[(k + 1) % m]

            # 取两特征点之间（沿环顺时针）所有中间索引
            mid_positions = cyclic_indices_between(n, i, j)
            if not mid_positions:
                continue

            # 过滤出非特征点
            nonfeat_positions = [p for p in mid_positions if int(keep[p]) == 0]
            if not nonfeat_positions:
                continue

            A = points[i]
            B = points[j]
            # 计算这些非特征点到 AB 线段的距离
            dists = [point_to_segment_distance(points[p], A, B) for p in nonfeat_positions]
            mean_dist = float(np.mean(dists))
            building_mean_dists.append(mean_dist)

        # 该建筑的最大 mean_dist
        if building_mean_dists:
            per_building_max[b_id] = float(np.max(building_mean_dists))

    # 全部建筑的最大值集合
    if not per_building_max:
        return {
            "avg": float("nan"),
            "median": float("nan"),
            "per_building_max": {}
        }

    max_values = np.array(list(per_building_max.values()), dtype=float)
    avg = float(np.mean(max_values))
    median = float(np.median(max_values))

    return {
        "avg": avg,
        "median": median,
        "per_building_max": per_building_max
    }


def evaluation_simplification_result(file_org, file_sim):
    gdf_org = gpd.read_file(file_org)
    gdf_sim = gpd.read_file(file_sim)
    n_org = len(gdf_org)
    n_sim = len(gdf_sim)
    n_ratio = n_sim / n_org
    max_dis = max_mean_perpendicular_distance(gdf_sim)
    coords_org = pts_from_gdf(gdf_org)
    num_max_than_10 = count_buildings_over_20_closed(coords_org, 10)

    coords_sim = pts_from_gdf(gdf_sim)
    print('比较选取后的点的重要性')
    # 找出coords_org最后一个键值+1，就是所有多边形的个数，
    try:
        max_key = max(coords_org.keys())
        num_polygons = max_key + 1
    except Exception:
        num_polygons = len(coords_org)
    print(f'多边形数量（估算）：{num_polygons}')

    # 针对每个多边形，计算ori和sim的hausdorff距离，面积变化和正交性变化，
    # 分别用hausdorff_distance函数，intersection_union_iou函数，calculate_orthogonality函数
    results = []
    # 统一使用两个字典键的交集，避免某些id缺失
    common_ids = sorted(set(coords_org.keys()).intersection(set(coords_sim.keys())))
    if not common_ids:
        print('原始与简化数据没有重叠的多边形id')
        return []
    hd_list, iou_list, delta_ortho_list, sc_list = [], [], [], []
    for pid in common_ids:
        ori = coords_org[pid]
        sim = coords_sim[pid]

        sc = calculate_shape_context(pid, ori, sim)

        # show_turning_function(pid, ori, sim)

        hd = hausdorff_distance(ori, sim)
        area_inter_union = intersection_union_iou(ori, sim)
        ortho_ori = calculate_orthogonality(ori)
        ortho_sim = calculate_orthogonality(sim)
        delta_ortho = ortho_sim / (ortho_ori + 1e-9)
        # print(f"id={pid}, HD={hd}, ΔA={area_inter_union}, ΔOrtho={delta_ortho}, sc = {sc}")
        # 记录当前值
        hd_list.append(hd)
        iou_list.append(area_inter_union)
        delta_ortho_list.append(delta_ortho)
        sc_list.append(sc)
    avg_hd = sum(hd_list) / len(hd_list)
    avg_iou = sum(iou_list) / len(iou_list)
    avg_delta_ortho = sum(delta_ortho_list) / len(delta_ortho_list)
    avg_sc = sum(sc_list) / len(sc_list)
    print("——— 汇总平均值 ———")
    print(f"节点压缩比率= {n_ratio}")
    print(f"位移平均值= {max_dis}")
    print(f"HD平均值={avg_hd}")
    print(f"ΔA平均值={avg_iou}")
    print(f"ΔOrtho平均值={avg_delta_ortho}")
    print(f"Δsc平均值={avg_sc}")


def angle_between(p_prev, p_curr, p_next):
    # 向量
    v1 = np.array([p_prev.x - p_curr.x, p_prev.y - p_curr.y], dtype=float)
    v2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y], dtype=float)
    # 处理零向量
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_val))
    return angle_deg


def count_right_angles_for_group(df, low=80.0, high=100.0):
    if df.empty:
        return 0, []  # 返回计数与该组的 keep==1 的 idx 列表

    # # 按 idx 排序，构建环绕邻接
    # df = df.sort_values("idx").reset_index(drop=True)
    # idx_list = df["idx"].tolist()

    n = len(df)
    if n < 3:
        return 0

    cnt = 0
    geom = df.geometry.values
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        ang = angle_between(geom[i_prev], geom[i], geom[i_next])
        if not np.isnan(ang) and (low <= ang <= high):
            cnt += 1
    return cnt


# def count_right_angles_by_building(gdf_sim, low=80.0, high=100.0):
# 	# 断言必需列存在
# 	required_cols = {"building_id", "idx", "keep_pred", "geometry"}
# 	miss = required_cols - set(gdf_sim.columns)
# 	if miss:
# 		raise ValueError(f"Missing required columns: {miss}")
#
# 	result = {}
# 	for bid, grp in gdf_sim.groupby("building_id", sort=False):
# 		result[bid] = count_right_angles_for_group(grp, low=low, high=high)
# 	return result

def count_right_angles_for_group_ori(gdf_ori, idx_dict, low=80.0, high=100.0):
    """
    参数：
    - gdf_ori: 简化前的点类型 GeoDataFrame，至少包含列 ["building_id", "idx", "geometry"]
    - idx_dict: {building_id: [idx1, idx2, ...]}，来自简化后统计的 keep_pred==1 的 idx 列表
    - low, high: 角度阈值，默认 [80, 100] 度

    返回：
    - result_dict: {building_id: 计数}，仅对 idx_dict 中出现的 building_id 计算并返回
    """
    required_cols = {"building_id", "idx", "geometry"}
    miss = required_cols - set(gdf_ori.columns)
    if miss:
        raise ValueError(f"Missing required columns in gdf_ori: {miss}")

    result_dict = {}

    # 只遍历 idx_dict 中出现的 building_id，避免无关计算
    for bid, idx_list in idx_dict.items():
        # 取该 building 的所有原始点
        grp = gdf_ori[gdf_ori["building_id"] == bid].copy()
        if grp.empty:
            result_dict[bid] = 0
            continue

        # 按 idx 排序，保证环绕邻接一致性（假设 idx 描述轮廓顺序）
        grp = grp.sort_values("idx").reset_index(drop=True)

        # 为了能通过“原始 idx 值”定位到排序后的位置，建立映射
        original_idx_sorted = grp["idx"].tolist()
        # 构建 {原始idx值: 排序后位置} 的字典
        pos_map = {orig_idx: pos for pos, orig_idx in enumerate(original_idx_sorted)}

        # 过滤只保留 idx_dict 给定的 idx 值，同时存在于 gdf_ori 该组中
        valid_positions = []
        for idx_val in idx_list:
            if idx_val in pos_map:
                valid_positions.append(pos_map[idx_val])

        if len(valid_positions) == 0:
            result_dict[bid] = 0
            continue

        # 计算这些位置对应点的夹角（邻接按整个组的环绕定义，不仅限于 idx_dict 子集）
        geom = grp.geometry.values
        n = len(grp)
        cnt = 0
        for pos in valid_positions:
            pos_prev = (pos - 1) % n
            pos_next = (pos + 1) % n
            ang = angle_between(geom[pos_prev], geom[pos], geom[pos_next])
            if not np.isnan(ang) and (low <= ang <= high):
                cnt += 1

        result_dict[bid] = cnt

    return result_dict


def stats_by_building(gdf_sim, low=80.0, high=100.0):
    """
    返回两个字典：
    - count_dict: {building_id: 该组内夹角在 [low, high] 的点计数}
    - idx_dict: {building_id: 该组内所有 keep_pred==1 的点的 idx 列表（按 idx 升序）}
    """
    required_cols = {"building_id", "idx", "keep_pred", "geometry"}
    miss = required_cols - set(gdf_sim.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # 仅保留 keep_pred == 1 的行
    gdf_kept = gdf_sim[gdf_sim["keep_pred"] == 1]

    count_dict = {}
    idx_dict = {}

    for bid, grp in gdf_kept.groupby("building_id", sort=False):
        # 按 idx 稳定排序
        grp_sorted = grp.sort_values("idx", kind="mergesort")
        # 收集 idx 列表（升序）
        idxs_sorted = grp_sorted["idx"].tolist()
        idx_dict[bid] = idxs_sorted

        cnt = count_right_angles_for_group(grp_sorted, low=low, high=high)
        count_dict[bid] = cnt

    return count_dict, idx_dict


def ratios_and_mean(n_90angles_sim, n_90angles_ori):
    """
    输入：
      - n_90angles_sim: {building_id: count_sim}
      - n_90angles_ori: {building_id: count_ori}
    输出：
      - ratio_dict: {building_id: ratio (sim/or i)}
      - mean_ratio: 所有 ratio 的平均值（跳过分母为 0 的键）
    """
    keys = set(n_90angles_sim.keys()) & set(n_90angles_ori.keys())
    ratio_dict = {}
    ratios = []

    for k in keys:
        denom = n_90angles_ori.get(k, 0)
        num = n_90angles_sim.get(k, 0)
        if denom is None or denom == 0:
            # 跳过分母为 0 或缺失的情况
            continue
        r = num / denom
        ratio_dict[k] = r
        ratios.append(r)

    mean_ratio = sum(ratios) / len(ratios) if len(ratios) > 0 else None
    median_ratio = np.median(ratios) if len(ratios) > 0 else None
    return mean_ratio, median_ratio


# 示例调用
# ratio_dict, mean_ratio = ratios_and_mean(n_90angles_sim, n_90angles_ori)
# print(ratio_dict)
# print("mean ratio:", mean_ratio)

def median_of_list(nums):
    if not nums:
        raise ValueError("list 不能为空")
    arr = sorted(nums)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    else:
        return (arr[mid - 1] + arr[mid]) / 2


def pts_ratio(gdf_org, gdf_sim):
    # 假设已有 gdf_sim, gdf_org，且都包含字段: building_id
    # 1) 在 gdf_sim 中筛选 keep_pred==1 的点
    gdf_sim_keep = gdf_sim[gdf_sim["keep_pred"] == 1]

    # 2) 分别按 building_id 分组计数
    # gdf_sim 中保留点的个数（每个 building_id 的点数）
    sim_cnt = gdf_sim_keep.groupby("building_id").size().rename("sim_keep_cnt")

    # gdf_org 中对应 building_id 的总点数
    org_cnt = gdf_org.groupby("building_id").size().rename("org_cnt")

    # 3) 对齐并求比值（sim_keep_cnt / org_cnt）
    # 使用外连接以保留所有 building_id；如只想要双方都存在的则用 how="inner"
    ratio_df = pd.concat([sim_cnt, org_cnt], axis=1)

    # 可选：只保留双方都存在的 building_id
    # ratio = ratio.dropna(subset=["sim_keep_cnt", "org_cnt"])

    # 处理除零与缺失
    # 若 org_cnt 为 0（几乎不可能，因为来自 groupby.size，但以防万一），或缺失，则设为 NaN
    ratio_df["ratio"] = ratio_df["sim_keep_cnt"] / ratio_df["org_cnt"]

    # 如果想把缺失填 0 或其他值：
    # ratio["ratio"] = ratio["ratio"].fillna(0)
    mean_val = ratio_df["ratio"].dropna().mean()
    median_val = ratio_df["ratio"].dropna().median()
    return mean_val, median_val


def evaluation_simplification_result2(file_org, file_sim):
    gdf_org = gpd.read_file(file_org)
    gdf_sim = gpd.read_file(file_sim)

    n_90angles_sim, n_idxs = stats_by_building(gdf_sim)
    n_90angles_ori = count_right_angles_for_group_ori(gdf_org, idx_dict=n_idxs)

    avg_delta_ortho, median_delta_ortho = ratios_and_mean(n_90angles_sim, n_90angles_ori)

    ratio_avg, ratio_median = pts_ratio(gdf_org, gdf_sim)

    max_dis_dic = max_mean_perpendicular_distance_stats(gdf_sim)

    coords_org = pts_from_gdf(gdf_org)
    num_max_than_10 = count_buildings_over_20_closed(coords_org, 10)

    coords_sim = pts_from_gdf2(gdf_sim)
    print('比较选取后的点的重要性')
    # 找出coords_org最后一个键值+1，就是所有多边形的个数，
    try:
        max_key = max(coords_org.keys())
        num_polygons = max_key + 1
    except Exception:
        num_polygons = len(coords_org)
    print(f'多边形数量（估算）：{num_polygons}')

    # 针对每个多边形，计算ori和sim的hausdorff距离，面积变化和正交性变化，
    # 分别用hausdorff_distance函数，intersection_union_iou函数，calculate_orthogonality函数
    results = []
    # 统一使用两个字典键的交集，避免某些id缺失
    common_ids = sorted(set(coords_org.keys()).intersection(set(coords_sim.keys())))
    if not common_ids:
        print('原始与简化数据没有重叠的多边形id')
        return []
    hd_list, area_change_list, delta_ortho_list, sc_list = [], [], [], []
    for pid in common_ids:
        ori = coords_org[pid]
        sim = coords_sim[pid]

        sc = calculate_shape_context(pid, ori, sim)

        # show_turning_function(pid, ori, sim)

        # area_change = area_ratio(sim, ori)
        areachange = area_change(sim, ori)

        area_change_list.append(areachange)

        sc_list.append(sc)

    avg_area = sum(area_change_list) / len(area_change_list)
    median_area = np.median(area_change_list)

    avg_sc = sum(sc_list) / len(sc_list)
    median_sc = np.median(sc_list)
    avg = max_dis_dic.get('avg')  # 如果确保存在，也可用 max_dis_dic['avg']
    median = max_dis_dic.get('median')

    print("——— 汇总平均值 ———")
    print(f"位移平均值= {avg}")
    print(f"位移中位数= {median}")
    print(f"ΔOrtho正交性改变的平均值={avg_delta_ortho}")
    print(f"ΔOrtho正交性改变的中位值={median_delta_ortho}")
    print(f"Δsc形状改变的平均值={avg_sc}")
    print(f"Δsc形状改变的中位数={median_sc}")

    print(f"Δ面积改变的平均值 ={avg_area}")
    print(f"Δ面积改变的中位值 ={median_area}")
    print(f"节点压缩比率平均值= {ratio_avg}")
    print(f"节点压缩比率中间值 = {ratio_median}")
    print("——— 汇总平均值 ———")


if __name__ == '__main__':
    evaluation_simplification_result2(r"F:\LandUseBoudary\code\data\test0310pts.geojson",
                                      r"F:\LandUseBoudary\code\data\val_keep_pred_0324.geojson")

    evaluation_train_dataset(r"F:\LandUseBoudary\code\data\train_pts_0_599.geojson")

    print('end')
