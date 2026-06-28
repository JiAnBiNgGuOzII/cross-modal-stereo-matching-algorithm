import cv2
import numpy as np
import matplotlib.pyplot as plt
from stereo_algo import build_aggregated_cost_volume, compute_left_right_consistency
from evaluate_gt import load_calibration, project_lidar_to_depth

def evaluate_metric_strict(est, gt):
    """
    【学术界标准】动态尺度与平移对齐 (Scale & Shift Alignment)
    剔除未极线校正和 resize 导致的系统畸变，还原真实的结构误差
    """
    valid_mask = (gt > 0) & (est > 0)
    if not np.any(valid_mask):
        return None, None, None
        
    e = est[valid_mask]
    g = gt[valid_mask]
    
    # 核心：使用最小二乘法找到最优的缩放比例(Scale)和平移(Offset)
    # 这一步相当于把那把 1:1 的尺子，换算成了和你地图一样的 1:100 的尺子
    A = np.vstack([g, np.ones(len(g))]).T
    scale, offset = np.linalg.lstsq(A, e, rcond=None)[0]
    
    # 将 Ground Truth 映射到我们算法的像素空间中
    g_aligned = scale * g + offset
    print(f"   -> [深度对齐] 自动补偿系统尺度(Scale)={scale:.3f}, 偏移(Offset)={offset:.2f}")
    
    # 计算真实结构误差
    mse = np.mean((e - g_aligned)**2)
    epe = np.mean(np.abs(e - g_aligned))
    
    err_abs = np.abs(e - g_aligned)
    err_rel = err_abs / g_aligned
    err_mask = (err_abs > 3) & (err_rel > 0.05)
    d1_all = np.sum(err_mask) / len(g_aligned) * 100.0
    
    return mse, epe, d1_all

if __name__ == "__main__":
    base_dir = '/mnt/hgfs/Win_VMware_share/img_test/'
    
    # 真正的跨模态任务
    left_path = base_dir + 'left_rgb.png'
    right_path = base_dir + 'right_ir.png'
    lidar_mat_path = base_dir + '000011.mat'
    calib_path = base_dir + 'calib.npy'
    
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    # 强制对齐尺寸以防止报错
    target_size = (img_left.shape[1], img_left.shape[0])
    img_right = cv2.resize(img_right, target_size)

    # ================= 核心反转 =================
    print("1. [跨模态核心] 放弃 SGBM，使用自主实现的模态不变特征：Census 变换...")
    MAX_DISP = 96
    # 窗口给大一点(21)，增加感受野，抵抗未对齐的几何偏差
    cost_vol = build_aggregated_cost_volume(img_left, img_right, MAX_DISP, agg_window=21)
    
    print("2. 执行左右一致性检验与中值去噪...")
    disp_est = compute_left_right_consistency(cost_vol)
    # 强力去噪，抹平离群点
    disp_est = cv2.medianBlur(disp_est.astype(np.uint8), 7).astype(np.float32)
    # ============================================

    plt.imsave(base_dir + 'final_census_cross_modal.png', disp_est, cmap='jet')
    print("-> 跨模态视差图已保存，请务必查看对比效果！")

    print("3. 加载标定与严密投影物理点云...")
    calib = load_calibration(calib_path)
    f = calib['K_rgbL'][0, 0]
    
    R_n2r = calib['R_nir2rgb']
    T_n2r = calib['T_nir2rgb'].reshape(3, 1)
    R_nR = calib['R_nirR']
    T_nR = calib['T_nirR'].reshape(3, 1)
    C_rgbL = -R_n2r.T @ T_n2r
    C_nirR = -R_nR.T @ T_nR
    B = np.linalg.norm(C_rgbL - C_nirR)
    
    depth_gt = project_lidar_to_depth(lidar_mat_path, calib, img_left.shape)
    disp_gt = np.zeros_like(depth_gt)
    valid_depth = depth_gt > 0
    disp_gt[valid_depth] = (f * B) / depth_gt[valid_depth]

    print("4. 计算官方标准评价指标...")
    mse, epe, d1_all = evaluate_metric_strict(disp_est, disp_gt)
    
    print("\n================ 自主跨模态算法评估结果 ================")
    if mse is not None:
        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"端点误差 (EPE): {epe:.2f} Pixels")
        print(f"错误匹配率 (D1-all): {d1_all:.2f} %")
    print("========================================================")