import torch
import cv2
import os
import numpy as np
from model import StereoNet
from dataset import CrossModalDataset

def run_inference(model_path, data_root, output_dir, num_samples=300):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = StereoNet(max_disp=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. 加载数据集
    dataset = CrossModalDataset(data_root)
    
    epe_list = []
    
    print(f"开始批量推理，共 {num_samples} 组...")
    
    with torch.no_grad():
        for i in range(num_samples):
            img_l, img_r, disp_gt = dataset[i]
            # 增加 Batch 维度: [1, 1, 384, 1224]
            img_l, img_r = img_l.unsqueeze(0).to(device), img_r.unsqueeze(0).to(device)
            
            # 推理
            pred_disp = model(img_l, img_r).squeeze().cpu().numpy()
            
            # 3. 可视化保存 (伪彩色图)
            pred_vis = cv2.normalize(pred_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, f"disp_{i:03d}.png"), pred_vis)
            
            # 4. 计算指标 (EPE)
            epe = np.mean(np.abs(pred_disp - disp_gt.squeeze().numpy()))
            epe_list.append(epe)
            
    print(f"推理完成！平均 EPE: {np.mean(epe_list):.4f}")

if __name__ == "__main__":
    # 在 Colab 中，这里路径就是 /content/stereo_project/ms2_dataset
    run_inference("best_model.pth", "./ms2_dataset", "./results")