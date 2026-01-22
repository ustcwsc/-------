import argparse
import torch
import json
import os
import sys
from torch.utils.data import DataLoader
from dataloader import DefectDataset 
from model import CNN
from utils import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型结构
    model = CNN(device)
    
    # 2. 加载权重 (增加健壮性处理)
    model_path = 'saved_model.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Make sure you trained the model first.")
        return

    try:
        # map_location 确保在只有 CPU 的机器上也能跑
        state = torch.load(model_path, map_location=device)
        
        # 逐层加载参数 (对应 model.py 中的定义)
        layers = [
            'conv1_weight', 'conv1_bias', 'c2_w', 'c2_b',
            'conv2_weight', 'conv2_bias', 'c4_w', 'c4_b',
            'c5_w', 'c5_b', 'c6_w', 'c6_b',
            'fc_weight', 'fc_bias'
        ]
        
        for layer in layers:
            if layer in state:
                # getattr(model, layer) 获取模型中的属性，.data.copy_() 原地更新数据
                getattr(model, layer).data.copy_(state[layer].to(device))
            else:
                print(f"Warning: Missing key {layer} in saved model.")
                
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 3. 加载数据
    test_dataset = DefectDataset(args.test_data_path, augment=False)
    # batch_size=1 保证最稳妥的逐张预测，shuffle=False 保证顺序
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. 推理
    y_true = []
    y_pred = []
    results = {}
    print(f"Start inference on {len(test_dataset)} images...")
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            img = img.to(device)
            output = model.forward(img)
            
            # 阈值判断 (0.5)
            pred_score = output.item()
            pred_label = 1 if pred_score >= 0.5 else 0
            
            # --- 核心：获取文件名用于生成 JSON ---
            # 假设 test_dataset.samples[idx] = (img_path, label, aug_flag)
            try:
                img_path = test_dataset.samples[idx][0]
                # 获取文件名 (不带路径) -> "glass_001.png"
                basename = os.path.basename(img_path)
                # 去除扩展名 -> "glass_001"
                file_key = os.path.splitext(basename)[0]
                
                # 存入字典: Task 1 要求 True(有缺陷) / False(无缺陷)
                results[file_key] = True if pred_label == 1 else False
                
            except Exception as e:
                print(f"Error processing filename for index {idx}: {e}")
            y_true.append(label.item())
            y_pred.append(pred_label)
    #5. 保存结果为 JSON (这是评分的关键！)
    student_id = 'PB23050934' 
    json_filename = f'{student_id}.json'
    
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f">> JSON result saved to {json_filename}")

    # 6. 计算 F1 分数 (仅当有标签时)
    # 检查 y_true 是否包含有效标签 (如果全是 0 且没有 txt 文件，可能是隐藏测试集)
    # 这里简单尝试计算
    try:
        precision, recall, f1 = compute_metrics(y_true, y_pred)
        # 按照要求输出学号和分数
        print(f'{student_id}:{f1:.2f}')
        print(f"Detailed Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print("Skipping metric calculation (Ground truth might be missing or all non-defective).")

if __name__ == '__main__':
    main()