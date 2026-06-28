import numpy as np
import scipy.io as sio

def lidar_to_disparity(lidar_path):
    """
    加载 .mat 文件并提取 'data' 键中的视差/深度信息
    """
    mat_data = sio.loadmat(lidar_path)
    # 注意：这里直接读取 'data' 键
    data = mat_data['data'].astype(np.float32)
    
    # 根据数据集说明，'data' 内部存储的可能是视差值(Disparity)
    # 如果是视差，直接返回；如果是深度(Depth)，请在此处加转换公式：(f*b)/data
    return data