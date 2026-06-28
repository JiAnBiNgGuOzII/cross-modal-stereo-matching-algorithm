import cv2
import numpy as np
import os
import glob
from stereo_algo import build_aggregated_cost_volume, compute_left_right_consistency
from evaluate_gt import load_calibration, project_lidar_to_depth

def evaluate_metric_aligned(est, gt):
    """
    动态尺度与平移对齐 (Scale & Shift Alignment)
    剔除未极线校正和 resize 导致的系统畸变
    """
    valid_mask = (gt > 0) & (est > 0)
    if not np.any(valid_mask):
        return None, None, None
        
    e = est[valid_mask]
    g = gt[valid_mask]
    
    A = np.vstack([g, np.ones(len(g))]).T
    try:
        scale, offset = np.linalg.lstsq(A, e, rcond=None)[0]
    except:
        return None, None, None
    
    g_aligned = scale * g + offset
    
    mse = np.mean((e - g_aligned)**2)
    epe = np.mean(np.abs(e - g_aligned))
    
    err_abs = np.abs(e - g_aligned)
    err_rel = err_abs / g_aligned
    err_mask = (err_abs > 3) & (err_rel > 0.05)
    d1_all = np.sum(err_mask) / len(g_aligned) * 100.0
    
    return mse, epe, d1_all

if __name__ == "__main__":
    # ==========================================
    # 1. 路径配置
    # ==========================================
    dataset_root = "/mnt/hgfs/Win_VMware_share/stereo_project/ms2_dataset" 
    
    rgb_dir = os.path.join(dataset_root, "rgb", "img_left")
    ir_dir = os.path.join(dataset_root, "nir", "img_right")
    lidar_dir = os.path.join(dataset_root, "lidar", "left")
    calib_path = os.path.join(dataset_root, "calib.npy")
    
    output_vis_dir = os.path.join(dataset_root, "output_vis")
    os.makedirs(output_vis_dir, exist_ok=True)
    
    # 穿透搜索所有的 png 图像
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "**", "*.png"), recursive=True))
    total_files = len(rgb_files)
    
    if total_files == 0:
        print(f"致命错误: 在 {rgb_dir} 下找不到图片，请检查路径！")
        exit()
        
    print(f"成功扫描到 {total_files} 组数据。")
    print(f"视差图可视化结果将被自动保存在: {output_vis_dir}")
    print("准备开始批处理 ...")
    
    calib = load_calibration(calib_path)
    f = calib['K_rgbL'][0, 0]
    R_n2r = calib['R_nir2rgb']
    T_n2r = calib['T_nir2rgb'].reshape(3, 1)
    R_nR = calib['R_nirR']
    T_nR = calib['T_nirR'].reshape(3, 1)
    C_rgbL = -R_n2r.T @ T_n2r
    C_nirR = -R_nR.T @ T_nR
    B = np.linalg.norm(C_rgbL - C_nirR)

    total_mse, total_epe, total_d1, valid_count = 0, 0, 0, 0

    print("\n" + "="*50)
    print("开始执行跨模态双目立体匹配 (RGB vs IR)...")
    print("="*50 + "\n")

    for idx, rgb_path in enumerate(rgb_files):
        filename = os.path.basename(rgb_path)
        base_name = os.path.splitext(filename)[0]
        
        ir_search = glob.glob(os.path.join(ir_dir, "**", filename), recursive=True)
        lidar_search = glob.glob(os.path.join(lidar_dir, "**", base_name + ".mat"), recursive=True)
        
        if not ir_search or not lidar_search:
            continue
            
        ir_path = ir_search[0]
        lidar_path = lidar_search[0]
            
        img_left = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if img_left is None or img_right is None:
             continue
             
        # 强制对齐尺寸
        target_size = (img_left.shape[1], img_left.shape[0])
        img_right = cv2.resize(img_right, target_size)

        # ---------------- 核心算法 ----------------
        cost_vol = build_aggregated_cost_volume(img_left, img_right, max_disp=96, agg_window=21)
        disp_est = compute_left_right_consistency(cost_vol)
        
        # 视觉优化：中值去噪
        disp_est = cv2.medianBlur(disp_est.astype(np.uint8), 7).astype(np.float32)
        # ------------------------------------------
        
        # 【OpenCV 级专业可视化】：彻底抛弃 Matplotlib
        vis_save_path = os.path.join(output_vis_dir, f"disp_{base_name}.png")
        valid_mask = disp_est > 0
        
        if np.any(valid_mask):
            # 1. 动态归一化有效视差到 0-255，防止崩溃
            disp_min = disp_est[valid_mask].min()
            disp_max = disp_est[valid_mask].max()
            disp_norm = np.zeros_like(disp_est, dtype=np.uint8)
            
            if disp_max > disp_min:
                norm_vals = (disp_est[valid_mask] - disp_min) / (disp_max - disp_min) * 255.0
                disp_norm[valid_mask] = norm_vals.astype(np.uint8)
            else:
                disp_norm[valid_mask] = 128
                
            # 2. 伪彩色映射
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
            
            # 3. 将无效区域强行涂黑
            disp_color[~valid_mask] = [0, 0, 0]
            cv2.imwrite(vis_save_path, disp_color)
        else:
            # 如果算法全军覆没没有任何匹配点，存一张纯黑图
            cv2.imwrite(vis_save_path, np.zeros((img_left.shape[0], img_left.shape[1], 3), dtype=np.uint8))

        try:
            depth_gt = project_lidar_to_depth(lidar_path, calib, img_left.shape)
            disp_gt = np.zeros_like(depth_gt)
            
            valid_depth = (depth_gt > 500) & np.isfinite(depth_gt)
            if not np.any(valid_depth):
                continue
                
            disp_gt[valid_depth] = (f * B) / depth_gt[valid_depth]

            mse, epe, d1_all = evaluate_metric_aligned(disp_est, disp_gt)
            
            if epe is not None and np.isfinite(epe):
                total_mse += mse
                total_epe += epe
                total_d1 += d1_all
                valid_count += 1
                
                # 每 10 组打印一次进度
                if valid_count % 10 == 0 or idx == total_files - 1:
                    print(f"-> 已完成 [{valid_count}/{total_files}] 组 | 最新帧 EPE: {epe:.2f} Px, D1: {d1_all:.1f}%")
        except Exception as e:
            continue
                
    if valid_count > 0:
        print("\n\n" + "="*50)
        print(f" 抽样验证集统计评估报告 (共成功处理 {valid_count} 组)")
        print("="*50)
        print(f" 平均 均方误差 (MSE)   : {total_mse / valid_count:.2f}")
        print(f" 平均 端点误差 (EPE)   : {total_epe / valid_count:.2f} Pixels")
        print(f" 平均 错误匹配率 (D1-all): {total_d1 / valid_count:.2f} %")
        print("="*50)
    else:
        print("\n评估失败，没有成功处理任何有效数据。")