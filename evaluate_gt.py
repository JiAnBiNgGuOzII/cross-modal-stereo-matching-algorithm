import numpy as np
import cv2
import scipy.io as sio

def load_calibration(calib_path):
    """读取 numpy 字典格式的标定文件"""
    # allow_pickle=True 是读取复杂字典对象必须的参数
    calib = np.load(calib_path, allow_pickle=True).item()
    return calib

def project_lidar_to_depth(mat_path, calib, image_shape):
    """
    将 Lidar 点云严密投影到【校正后 (Rectified)】的 RGB Left 图像平面
    """
    H, W = image_shape
    
    mat_data = sio.loadmat(mat_path)
    key = [k for k in mat_data.keys() if not k.startswith('__')][-1]
    points_lidar = mat_data[key]
    
    if points_lidar.shape[1] > 3:
        points_lidar = points_lidar[:, :3] 
        
    # 统一单位：米 -> 毫米
    if np.max(np.abs(points_lidar)) < 1000:
        points_lidar = points_lidar * 1000.0
        
    # 1. 提取基础外参 (Lidar -> NIR Left -> RGB Left)
    R_n2l = calib['R_nir2lidarL']
    T_n2l = calib['T_nir2lidarL'].reshape(3, 1)
    R_n2r = calib['R_nir2rgb']
    T_n2r = calib['T_nir2rgb'].reshape(3, 1)
    
    # 2. 【核心修复】提取极线校正矩阵 (Unrectified -> Rectified)
    R_rect_rgbL = calib['R_rgbL']
    T_rect_rgbL = calib['T_rgbL'].reshape(3, 1)
    K_rgbL = calib['K_rgbL']
    
    # --- 严格的 3D 空间变换链路 ---
    points_lidar_T = points_lidar.T # (3, N)
    
    # A. 转换到 NIR Left
    P_nirL = R_n2l.T @ (points_lidar_T - T_n2l)
    
    # B. 转换到 RGB Left (未校正)
    P_rgbL_unrect = R_n2r @ P_nirL + T_n2r
    
    # C. 极线校正旋转与平移 (消除空间扭曲！)
    P_rgbL_rect = R_rect_rgbL @ P_rgbL_unrect + T_rect_rgbL
    
    # 过滤相机背后的点
    valid_depth_mask = P_rgbL_rect[2, :] > 0
    P_rgbL_rect = P_rgbL_rect[:, valid_depth_mask]
    
    # D. 投影到 2D 像素
    P_img = K_rgbL @ P_rgbL_rect
    u = np.round(P_img[0, :] / P_img[2, :]).astype(int)
    v = np.round(P_img[1, :] / P_img[2, :]).astype(int)
    Z = P_rgbL_rect[2, :] # 真实深度 mm
    
    depth_gt = np.zeros((H, W), dtype=np.float32)
    valid_uv_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    u = u[valid_uv_mask]
    v = v[valid_uv_mask]
    Z = Z[valid_uv_mask]
    
    # 赋值生成深度图
    depth_gt[v, u] = Z
    return depth_gt

def calculate_effective_baseline(calib):
    """
    计算 RGB Left 和 NIR Right 之间的等效物理基线 B
    """
    R_n2r = calib['R_nir2rgb']
    T_n2r = calib['T_nir2rgb'].reshape(3, 1)

    R_nR = calib['R_nirR']
    T_nR = calib['T_nirR'].reshape(3, 1)

    # 计算两个相机在 NIR Left 坐标系下的光心位置 C = -R^T * T
    C_rgbL = -R_n2r.T @ T_n2r
    C_nirR = -R_nR.T @ T_nR

    # 等效基线即为两光心之间的欧氏距离
    B_effective = np.linalg.norm(C_rgbL - C_nirR)
    return B_effective

def calculate_rgb_baseline(calib):
    """
    计算 RGB Left 和 RGB Right 之间的物理基线 B
    """
    # 根据 readme，T_rgbR 就是 rgbL 到 rgbR 的平移向量 (单位：mm)
    T_rgbR = calib['T_rgbR']
    return np.linalg.norm(T_rgbR)
