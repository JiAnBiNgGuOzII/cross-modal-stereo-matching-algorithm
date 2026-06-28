import numpy as np
import cv2

def compute_census_signature(img, window_size=5):
    """
    计算图像的 Census 布尔签名矩阵
    """
    H, W = img.shape
    half_w = window_size // 2
    num_neighbors = window_size**2 - 1
    signature = np.zeros((num_neighbors, H, W), dtype=bool)

    idx = 0
    for dy in range(-half_w, half_w + 1):
        for dx in range(-half_w, half_w + 1):
            if dy == 0 and dx == 0: continue

            shifted_img = np.zeros_like(img)
            y_start, y_end = max(0, -dy), min(H, H - dy)
            x_start, x_end = max(0, -dx), min(W, W - dx)
            shift_y_start, shift_y_end = max(0, dy), min(H, H + dy)
            shift_x_start, shift_x_end = max(0, dx), min(W, W + dx)

            shifted_img[y_start:y_end, x_start:x_end] = img[shift_y_start:shift_y_end, shift_x_start:shift_x_end]
            signature[idx, :, :] = shifted_img < img
            idx += 1
    return signature

def build_aggregated_cost_volume(img_left, img_right, max_disp, agg_window=9):
    """
    计算 Census 代价并执行代价聚合 (均值滤波)
    """
    H, W = img_left.shape
    cost_volume = np.zeros((H, W, max_disp), dtype=np.float32)
    sig_left = compute_census_signature(img_left)
    sig_right = compute_census_signature(img_right)

    for d in range(max_disp):
        sig_right_shifted = np.zeros_like(sig_right)
        if d > 0:
            sig_right_shifted[:, :, d:] = sig_right[:, :, :-d]
        else:
            sig_right_shifted = sig_right

        # 计算汉明距离
        xor_result = np.bitwise_xor(sig_left, sig_right_shifted)
        raw_cost = np.sum(xor_result, axis=0).astype(np.float32)

        # 代价聚合：对当前视差层的代价值进行局部空间平滑
        cost_volume[:, :, d] = cv2.boxFilter(raw_cost, -1, (agg_window, agg_window))

    return cost_volume

def compute_left_right_consistency(cost_volume):
    """
    执行左右一致性检验 (LRC)，严格向量化计算
    """
    H, W, D = cost_volume.shape

    # 1. 计算左视差图 (WTA策略)
    disp_left = np.argmin(cost_volume, axis=2).astype(np.float32)

    # 2. 从左代价体推导右代价体，并计算右视差图
    right_cost_volume = np.full((H, W, D), np.inf, dtype=np.float32)
    for d in range(D):
        if d == 0:
            right_cost_volume[:, :, d] = cost_volume[:, :, d]
        else:
            right_cost_volume[:, :-d, d] = cost_volume[:, d:, d]
    disp_right = np.argmin(right_cost_volume, axis=2).astype(np.float32)

    # 3. LRC 几何校验 (向量化坐标映射)
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # 映射左图坐标到右图对应坐标
    x_right = xv - disp_left
    valid_bounds = (x_right >= 0) & (x_right < W)

    disp_right_matched = np.zeros_like(disp_left)
    # 取整数索引进行匹配映射
    valid_x_right = x_right[valid_bounds].astype(np.int32)
    valid_yv = yv[valid_bounds].astype(np.int32)

    disp_right_matched[valid_bounds] = disp_right[valid_yv, valid_x_right]

    # 校验条件：绝对误差 <= 1
    lrc_mask = np.abs(disp_left - disp_right_matched) <= 1
    final_valid_mask = valid_bounds & lrc_mask

    # 将无效区域视差置零
    final_disp = disp_left.copy()
    final_disp[~final_valid_mask] = 0

    return final_disp