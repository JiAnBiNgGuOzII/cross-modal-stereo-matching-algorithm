import torch
from torch.utils.data import Dataset
from utils import lidar_to_disparity
import cv2
import numpy as np
import os
import glob

class CrossModalDataset(Dataset):
    def __init__(self, root_dir):
        """
        根据确认的路径结构构建索引：
        root_dir: /mnt/hgfs/Win_VMware_share/stereo_project/ms2_dataset
        RGB路径:  root_dir/rgb/img_left/
        NIR路径:  root_dir/nir/img_right/
        """
        # 使用 os.path.join 确保路径拼接在 Linux 下正确
        rgb_left_dir = os.path.join(root_dir, 'rgb', 'img_left')
        nir_right_dir = os.path.join(root_dir, 'nir', 'img_right')
        lidar_left_dir = os.path.join(root_dir, 'lidar', 'left')

        rgb_files = sorted(glob.glob(os.path.join(rgb_left_dir, '*.png')))
        nir_files = sorted(glob.glob(os.path.join(nir_right_dir, '*.png')))
        lidar_files = sorted(glob.glob(os.path.join(lidar_left_dir, '*.mat'))) # 或者 .npy, .png

        # 验证数量是否一致
        assert len(rgb_files) == len(nir_files) == len(lidar_files), "文件数量不匹配，请检查数据集完整性！"

        # 严格配对：(RGB_L, NIR_R, LiDAR_L)
        self.samples = list(zip(rgb_files, nir_files, lidar_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nir_path, lidar_path = self.samples[idx]
        
        # 1. 读取 RGB 和 NIR
        img_l = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 读取 LiDAR 真值 (GT)
        disp_gt = lidar_to_disparity(lidar_path)
        
        # 3. 统一尺寸 (这里必须将所有数据都 Resize 到相同大小)
        target_size = (1224, 384)
        img_l = cv2.resize(img_l, target_size).astype(np.float32) / 255.0
        img_r = cv2.resize(img_r, target_size).astype(np.float32) / 255.0
        disp_gt = cv2.resize(disp_gt, target_size)
        
        # 4. 转换为 Tensor
        return (torch.from_numpy(img_l).unsqueeze(0), 
                torch.from_numpy(img_r).unsqueeze(0), 
                torch.from_numpy(disp_gt).unsqueeze(0))

if __name__ == "__main__":
    # 使用相对路径，这样无论你在哪里运行，只要目录结构不变都能找到数据
    root = "./ms2_dataset" 
    # 如果通过参数传入，则优先使用参数
    import sys
    if len(sys.argv) > 1:
        root = sys.argv[1]
        
    dataset = CrossModalDataset(root)
    print(f"成功加载数据集，共找到 {len(dataset)} 组配对数据。")