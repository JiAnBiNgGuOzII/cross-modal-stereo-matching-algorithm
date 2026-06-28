import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """特征提取器：将图像映射到高维特征空间"""
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class StereoNet(nn.Module):
    def __init__(self, max_disp=48):
        super(StereoNet, self).__init__()
        self.max_disp = max_disp
        self.rgb_branch = FeatureExtractor(1)
        self.ir_branch = FeatureExtractor(1)

    def build_cost_volume(self, feat_l, feat_r):
        batch, channels, height, width = feat_l.shape
        cost_volume = torch.zeros((batch, channels * 2, self.max_disp, height, width)).to(feat_l.device)
        
        for d in range(self.max_disp):
            if d == 0:
                cost_volume[:, :, d, :, :] = torch.cat((feat_l, feat_r), dim=1)
            else:
                shifted_feat_r = torch.zeros_like(feat_r)
                shifted_feat_r[:, :, :, d:] = feat_r[:, :, :, :-d]
                cost_volume[:, :, d, :, :] = torch.cat((feat_l, shifted_feat_r), dim=1)
        return cost_volume

    def forward(self, rgb_left, ir_right):
        # 1. 特征提取
        feat_l = self.rgb_branch(rgb_left)
        feat_r = self.ir_branch(ir_right)
        
        # 2. 构建代价体 [B, 128, 48, H, W]
        cost_vol = self.build_cost_volume(feat_l, feat_r)
        
        # 3. 计算视差 (Soft-Argmax)
        score = -torch.mean(cost_vol, dim=1)  # [B, 48, H, W]
        prob = torch.softmax(score, dim=1)
        
        # 构建视差索引 [0, ..., 47]
        disp_indices = torch.arange(self.max_disp).view(1, -1, 1, 1).to(cost_vol.device).float()
        
        # 计算期望视差
        disp_map = torch.sum(prob * disp_indices, dim=1, keepdim=True)
        return disp_map

if __name__ == "__main__":
    model = StereoNet(max_disp=48)
    test_l = torch.randn(1, 1, 384, 1224)
    test_r = torch.randn(1, 1, 384, 1224)
    
    # 这一行现在只会接收一个返回值：disp_map
    disp_map = model(test_l, test_r)
    print(f"预测视差图输出形状: {disp_map.shape}")