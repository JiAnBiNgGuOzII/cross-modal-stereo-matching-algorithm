import scipy.io as sio
import os

# 随便取一个 LiDAR 文件的路径（请确保路径正确，可以从你的文件夹里复制一个文件名）
lidar_path = "/mnt/hgfs/Win_VMware_share/stereo_project/ms2_dataset/lidar/left/000000.mat" 

# 加载数据
data = sio.loadmat(lidar_path)

# 打印所有键名
print(f"DEBUG: 文件 {lidar_path} 中的键名 (Keys) 为: {data.keys()}")

# 尝试查看其中一个键的数据形状 (如果键名比较多，先挑一个看看)
# for key in data.keys():
#     if not key.startswith('__'): # 过滤掉系统自带的 metadata
#         print(f"键 '{key}' 的数据形状: {data[key].shape}")