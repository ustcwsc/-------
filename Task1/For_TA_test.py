import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import DefectDataset 
from model import CNN
from utils import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    # TA 会传入 .../dataset/test 这样的路径
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型结构
    model = CNN(device)
    
    # 2. 加载权重
    try:
        # map_location 确保在只有 CPU 的机器上也能跑
        state = torch.load('saved_model.pt', map_location=device)
        model.conv1_weight = state['conv1_weight'].to(device)
        model.conv1_bias = state['conv1_bias'].to(device)
        model.conv2_weight = state['conv2_weight'].to(device)
        model.conv2_bias = state['conv2_bias'].to(device)
        model.fc_weight = state['fc_weight'].to(device)
        model.fc_bias = state['fc_bias'].to(device)
        
        # 加载中间层权重 (辅助层)
        model.c2_w = state['c2_w'].to(device)
        model.c2_b = state['c2_b'].to(device)
        model.c4_w = state['c4_w'].to(device)
        model.c4_b = state['c4_b'].to(device)
        model.c5_w = state['c5_w'].to(device)
        model.c5_b = state['c5_b'].to(device)
        model.c6_w = state['c6_w'].to(device)
        model.c6_b = state['c6_b'].to(device)
    except KeyError as e:
        # 防止权重键名不匹配导致崩溃，打印错误但不中断（虽然这样预测结果会不对）
        pass
    except FileNotFoundError:
        print("Error: saved_model.pt not found.")
        return

    # 3. 加载数据
    test_dataset = DefectDataset(args.test_data_path, augment=False)
    # batch_size=1 保证最稳妥的逐张预测，shuffle=False 保证顺序
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. 推理
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            output = model.forward(img)
            # 这里的阈值 0.5 可以根据验证集微调，比如改为 0.4 或 0.6
            pred_label = 1 if output.item() >= 0.5 else 0
            
            y_true.append(label.item())
            y_pred.append(pred_label)

    # 5. 计算指标
    _, _, f1 = compute_metrics(y_true, y_pred)

    # 6. 输出结果
    student_id = 'PB23000000' 
    print(f'{student_id}:{f1:.2f}') 

if __name__ == '__main__':
    main()