import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CrossModalDataset
from model import StereoNet

def train():
    # 1. 配置参数
    # 使用 GPU 加速（如果在 Colab 上），若在本地则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备: {device}")
    
    batch_size = 2
    lr = 1e-4
    epochs = 10
    
    # 2. 加载数据集
    root_path = "/mnt/hgfs/Win_VMware_share/stereo_project/ms2_dataset"
    dataset = CrossModalDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 初始化模型、优化器与损失函数
    model = StereoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 使用 SmoothL1Loss，对异常值更具鲁棒性
    criterion = nn.SmoothL1Loss()
    
    # 4. 训练循环
    model.train()
    for epoch in range(epochs):
        for i, (img_l, img_r, disp_gt) in enumerate(dataloader):
            # 将数据移动到设备
            img_l, img_r, disp_gt = img_l.to(device), img_r.to(device), disp_gt.to(device)
            
            optimizer.zero_grad()
            
            # 模型前向传播
            disp_pred = model(img_l, img_r)
            
            # 5. 维度对齐检查（防止卷积层 padding 导致的尺寸差异）
            if disp_pred.shape != disp_gt.shape:
                disp_gt = F.interpolate(disp_gt, size=disp_pred.shape[2:], mode='bilinear', align_corners=False)
            
            # 计算 Loss
            loss = criterion(disp_pred, disp_gt)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            # 打印训练进度
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")
            
            # 在验证阶段，我们可以去掉 break
            # if i % 10 == 0: ...
        
        print(f"Epoch {epoch+1} 完成。")

if __name__ == "__main__":
    train()