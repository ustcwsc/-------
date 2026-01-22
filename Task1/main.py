import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataloader import DefectDataset
from model import CNN
from utils import compute_metrics


def main():
    # -------------------------------
    # 1. Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to .pt file to resume from')
    args = parser.parse_args()

    # -------------------------------
    # 2. Set device
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # -------------------------------
    # 3. Load dataset & dataloader
    # -------------------------------
    # 训练集开启增强，验证集关闭增强
    train_dataset = DefectDataset(args.train_data_path, augment=True)
    val_dataset = DefectDataset(args.val_data_path, augment=False)

    batch_size = 64
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # -------------------------------
    # 4. Initialize model
    # -------------------------------
    model = CNN(device)
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"Resuming training from {args.resume} ...")
            # map_location 确保跨设备加载不出错
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 使用 try-except 块以防旧模型文件缺少某些层的参数
            try:
                # Block 1
                model.conv1_weight = checkpoint['conv1_weight'].to(device)
                model.conv1_bias = checkpoint['conv1_bias'].to(device)
                model.c2_w = checkpoint['c2_w'].to(device)
                model.c2_b = checkpoint['c2_b'].to(device)

                # Block 2
                model.conv2_weight = checkpoint['conv2_weight'].to(device)
                model.conv2_bias = checkpoint['conv2_bias'].to(device)
                model.c4_w = checkpoint['c4_w'].to(device)
                model.c4_b = checkpoint['c4_b'].to(device)

                # Block 3
                model.c5_w = checkpoint['c5_w'].to(device)
                model.c5_b = checkpoint['c5_b'].to(device)
                model.c6_w = checkpoint['c6_w'].to(device)
                model.c6_b = checkpoint['c6_b'].to(device)

                # FC
                model.fc_weight = checkpoint['fc_weight'].to(device)
                model.fc_bias = checkpoint['fc_bias'].to(device)
                
                print(" >> Weights loaded successfully!")
                
            except KeyError as e:
                print(f" >> Warning: Key {e} not found in checkpoint. Some layers might be initialized randomly.")
        else:
            print(f" >> Error: Checkpoint file {args.resume} not found!")
    # -------------------------------
    # 5. Hyperparameters (经过优化)
    # -------------------------------
    num_epochs = 30    # 增加轮数，确保充分收敛
    lr = 2e-4          # 降低初始学习率，防止震荡
    momentum = 0.9     # 引入动量，加速收敛并冲过局部极小值

    # -------------------------------
    # 6. Metric storage
    # -------------------------------
    train_losses, train_accs = [], []
    val_precisions, val_recalls, val_f1s = [], [], []

    # -------------------------------
    # 7. Training Loop
    # -------------------------------
    print(f"Start training: Epochs={num_epochs}, LR={lr}, Momentum={momentum}")
    
    with open('score.txt', 'w') as f:
        f.write("Model Hyperparameters:\n")
        f.write(f"Initial Learning rate: {lr}\n")
        f.write(f"Momentum: {momentum}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n\n")
        f.write("Training process:\n")

        for epoch in range(num_epochs):
            model_loss = 0.0
            correct = 0
            total = 0

            # 学习率衰减策略：每10个epoch减半
            if epoch > 0 and epoch % 10 == 0:
                lr = lr * 0.5
                print(f" >> Learning Rate decayed to {lr:.6f}")

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                unit="batch"
            )

            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                # --- Forward ---
                out = model.forward(xb)
                pos_weight = 2.0
                # --- Loss Calculation (Manual BCE) ---
                # 添加 epsilon 防止 log(0)
                loss = -(pos_weight*yb * torch.log(out + 1e-6) +
                         (1 - yb) * torch.log(1 - out + 1e-6)).mean()

                # --- Backward (Manual with Momentum) ---
                # 注意：这需要 model.py 中的 backward 支持 momentum 参数
                model.backward(yb, lr, momentum)

                # --- Statistics ---
                batch_size_now = yb.size(0)
                model_loss += loss.item() * batch_size_now
                preds = (out >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += batch_size_now

                pbar.set_postfix(loss=loss.item(), lr=lr)

            # Epoch 统计
            avg_loss = model_loss / total
            accuracy = correct / total
            train_losses.append(avg_loss)
            train_accs.append(accuracy)

            # -------------------------------
            # 8. Validation
            # -------------------------------
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model.forward(xb)
                    # 阈值 0.5，可根据验证结果微调
                    preds = (out >= 0.5).int().cpu().tolist()
                    y_pred.extend(preds)
                    y_true.extend(yb.int().tolist())

            precision, recall, f1 = compute_metrics(y_true, y_pred)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(f1)

            log_line = (
                f"Epoch {epoch+1}: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                f"Val Precision={precision:.4f}, "
                f"Val Recall={recall:.4f}, Val F1={f1:.4f}\n"
            )
            print(log_line, end='')
            f.write(log_line)
            
            # (可选) 可以在这里添加 Early Stopping：如果 F1 连续不上升则停止

    # -------------------------------
    # 9. Save Model (CRITICAL FIX)
    # -------------------------------
    # 必须保存所有定义的参数，包括中间层的 c2, c4, c5, c6
    print("Saving model to saved_model.pt...")
    torch.save({
        # Block 1
        'conv1_weight': model.conv1_weight.cpu(),
        'conv1_bias': model.conv1_bias.cpu(),
        'c2_w': model.c2_w.cpu(), 
        'c2_b': model.c2_b.cpu(),
        
        # Block 2
        'conv2_weight': model.conv2_weight.cpu(),
        'conv2_bias': model.conv2_bias.cpu(),
        'c4_w': model.c4_w.cpu(), 
        'c4_b': model.c4_b.cpu(),
        
        # Block 3
        'c5_w': model.c5_w.cpu(), 
        'c5_b': model.c5_b.cpu(),
        'c6_w': model.c6_w.cpu(), 
        'c6_b': model.c6_b.cpu(),
        
        # FC
        'fc_weight': model.fc_weight.cpu(),
        'fc_bias': model.fc_bias.cpu()
    }, 'saved_model.pt')
    print("Model saved successfully.")

    # -------------------------------
    # 10. Plot metrics
    # -------------------------------
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='x')
    plt.plot(epochs, val_precisions, label='Val Precision', marker='s')
    plt.plot(epochs, val_recalls, label='Val Recall', marker='d')
    plt.plot(epochs, val_f1s, label='Val F1', marker='^')

    plt.title('Training & Validation Metrics (Improved)')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics_pretty.png')
    plt.show()

if __name__ == '__main__':
    main()