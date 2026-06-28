import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from stereo_algo import build_aggregated_cost_volume, compute_left_right_consistency
from evaluate_gt import load_calibration, project_lidar_to_depth

def evaluate_metric_aligned(est, gt):
    """动态尺度与平移对齐评估"""
    valid_mask = (gt > 0) & (est > 0) & np.isfinite(gt) & np.isfinite(est)
    if not np.any(valid_mask): 
        return None, None, None
    e, g = est[valid_mask], gt[valid_mask]
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
    dataset_root = "/mnt/hgfs/Win_VMware_share/stereo_project/ms2_dataset"
    rgb_l_dir = os.path.join(dataset_root, "rgb", "img_left")
    rgb_r_dir = os.path.join(dataset_root, "rgb", "img_right")
    ir_dir = os.path.join(dataset_root, "nir", "img_right")
    lidar_dir = os.path.join(dataset_root, "lidar", "left")
    calib_path = os.path.join(dataset_root, "calib.npy")
    
    output_vis_dir = os.path.join(dataset_root, "output_vis_comparison")
    os.makedirs(output_vis_dir, exist_ok=True)

    # 先获取所有按照顺序排列的图片列表
    all_rgb_files = sorted(glob.glob(os.path.join(rgb_l_dir, "**", "*.png"), recursive=True))

    # 自由组合需要的数据段
    # all_rgb_files[:150]   # [:150] 表示从第 0 个取到第 150 个
    # all_rgb_files[-150:]   # [-150:] 表示从倒数第 150 个取到最后
    # all_rgb_files[100:200]：跳过前 100 张，取中间的第 100 到 200 张。

    # 将两段数据拼接在一起
    rgb_files = all_rgb_files[:150] + all_rgb_files[-150:]
    
    total_files = len(rgb_files)
    
    if total_files == 0:
        print("未找到图片，请检查 dataset_root 路径！")
        exit()
        
    calib = load_calibration(calib_path)
    f = calib['K_rgbL'][0, 0]
    
    # 1：跨模态基线 (RGB Left 到 NIR Right)
    C_rgbL = -np.linalg.inv(calib['R_nir2rgb']) @ calib['T_nir2rgb'].reshape(3,1)
    C_nirR = -np.linalg.inv(calib['R_nirR']) @ calib['T_nirR'].reshape(3,1)
    B_cross = np.linalg.norm(C_rgbL - C_nirR)
    
    # 2：同模态基线 (RGB Left 到 RGB Right)
    B_homo = np.linalg.norm(calib['T_rgbR'])
    
    # 初始化 OpenCV SGBM
    window_size = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=window_size,
        P1=8 * 1 * window_size**2, P2=32 * 1 * window_size**2,
        disp12MaxDiff=-1, uniquenessRatio=0, speckleWindowSize=0, speckleRange=0,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 双轨统计累加器
    tc_mse, tc_epe, tc_d1, count_c = 0, 0, 0, 0 # Census (Cross-modal)
    ts_mse, ts_epe, ts_d1, count_s = 0, 0, 0, 0 # SGBM (Co-modal)

    print(f"\n开始同模态与跨模态对比评测，共测试 {total_files} 组数据...")
    print("="*60)

    for idx, rgb_l_path in enumerate(rgb_files):
        base_name = os.path.basename(rgb_l_path).split('.')[0]
        ir_path = rgb_l_path.replace("rgb/img_left", "nir/img_right")
        rgb_r_path = rgb_l_path.replace("img_left", "img_right")
        lidar_path = rgb_l_path.replace("rgb/img_left", "lidar/left").replace(".png", ".mat")
        
        img_l = cv2.imread(rgb_l_path, cv2.IMREAD_GRAYSCALE)
        img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(rgb_r_path, cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_ir is None or img_r is None: continue
        
        target_size = (img_l.shape[1], img_l.shape[0])
        img_ir = cv2.resize(img_ir, target_size)
        img_r = cv2.resize(img_r, target_size)

        # ---------------- 1. 跨模态：Census 算法 ----------------
        cost_vol = build_aggregated_cost_volume(img_l, img_ir, max_disp=96, agg_window=45)
        disp_census = compute_left_right_consistency(cost_vol)
        disp_census = cv2.medianBlur(disp_census.astype(np.uint8), 7).astype(np.float32)
        plt.imsave(os.path.join(output_vis_dir, f"census_cross_{base_name}.png"), disp_census, cmap='jet')

        # ---------------- 2. 同模态：SGBM 对照组 ----------------
        disp_sgbm = sgbm.compute(img_l, img_r).astype(np.float32) / 16.0
        disp_sgbm[disp_sgbm < 0] = 0
        plt.imsave(os.path.join(output_vis_dir, f"sgbm_ref_{base_name}.png"), disp_sgbm, cmap='jet')
        
        # ---------------- 3. 雷达投影与双轨评测 ----------------
        try:
            depth_gt = project_lidar_to_depth(lidar_path, calib, img_l.shape)
            valid_depth = (depth_gt > 500) & np.isfinite(depth_gt)
            if not np.any(valid_depth): continue
            
            # 分别生成对应物理基线的 Ground Truth 视差
            disp_gt_census = np.zeros_like(depth_gt)
            disp_gt_sgbm = np.zeros_like(depth_gt)
            
            disp_gt_census[valid_depth] = (f * B_cross) / depth_gt[valid_depth]
            disp_gt_sgbm[valid_depth] = (f * B_homo) / depth_gt[valid_depth]

            # 评测 Census
            mse_c, epe_c, d1_c = evaluate_metric_aligned(disp_census, disp_gt_census)
            if epe_c is not None:
                tc_mse += mse_c; tc_epe += epe_c; tc_d1 += d1_c; count_c += 1
                
            # 评测 SGBM
            mse_s, epe_s, d1_s = evaluate_metric_aligned(disp_sgbm, disp_gt_sgbm)
            if epe_s is not None:
                ts_mse += mse_s; ts_epe += epe_s; ts_d1 += d1_s; count_s += 1
                
            if (idx + 1) % 10 == 0 or idx == total_files - 1:
                print(f"-> 进度 [{idx+1}/{total_files}] | "
                      f"当前帧 EPE -> Census: {epe_c:.2f}px, SGBM: {epe_s:.2f}px")
        except Exception as e:
            continue

    # ==========================================
    # 输出对比
    # ==========================================
    if count_c > 0 and count_s > 0:
        print("\n\n" + "="*60)
        print(f" 大规模数据双轨统计评估报告 (共处理 {count_c} 组有效数据)")
        print("="*60)
        print(f"{'评测指标':<15} | {'同模态 (SGBM 基准)':<20} | {'跨模态 (Census 挑战)':<20}")
        print("-" * 60)
        print(f"{'平均 MSE':<15} | {ts_mse/count_s:<20.2f} | {tc_mse/count_c:<20.2f}")
        print(f"{'平均 EPE (px)':<15} | {ts_epe/count_s:<20.2f} | {tc_epe/count_c:<20.2f}")
        print(f"{'D1-all 错误率':<15} | {ts_d1/count_s:<19.2f}% | {tc_d1/count_c:<19.2f}%")
        print("="*60)