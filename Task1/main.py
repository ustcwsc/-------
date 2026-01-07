# main.py
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm   # ⭐ 新增
from dataloader import load_images_and_labels
from model import CNN
from utils import compute_metrics


def main():
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='train')
    parser.add_argument('--val_data_path', type=str, default='val')
    args = parser.parse_args()

    # -------------------------------
    # Set device
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------
    # Load data
    # -------------------------------
    train_data = load_images_and_labels(args.train_data_path, device)
    val_data = load_images_and_labels(args.val_data_path, device)

    # -------------------------------
    # Initialize model
    # -------------------------------
    model = CNN(device)

    # -------------------------------
    # Hyperparameters
    # -------------------------------
    num_epochs = 5
    lr = 0.001

    # -------------------------------
    # Metric storage
    # -------------------------------
    train_losses, train_accs = [], []
    val_precisions, val_recalls, val_f1s = [], [], []

    # -------------------------------
    # Open score.txt
    # -------------------------------
    with open('score.txt', 'w') as f:
        f.write("Model Hyperparameters:\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Number of epochs: {num_epochs}\n\n")
        f.write("Training process:\n")

        # -------------------------------
        # Training loop
        # -------------------------------
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0

            # ⭐ tqdm 进度条（核心）
            for img, label in tqdm(
                train_data,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                unit="img",
                leave=False
            ):
                output = model.forward(img)
                prob = output.clamp(1e-6, 1 - 1e-6)
                y_tensor = torch.tensor(label, device=device, dtype=torch.float32)

                loss = -(y_tensor * torch.log(prob) + (1 - y_tensor) * torch.log(1 - prob))
                total_loss += loss.item()

                pred_label = 1 if output.item() >= 0.5 else 0
                if pred_label == label:
                    correct += 1

                model.backward(label, lr)

            # Training metrics
            avg_loss = total_loss / len(train_data)
            accuracy = correct / len(train_data)
            train_losses.append(avg_loss)
            train_accs.append(accuracy)

            # -------------------------------
            # Validation
            # -------------------------------
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                for img, label in val_data:
                    output = model.forward(img)
                    pred_label = 1 if output.item() >= 0.5 else 0
                    y_true_val.append(label)
                    y_pred_val.append(pred_label)

            precision, recall, f1 = compute_metrics(y_true_val, y_pred_val)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(f1)

            log_line = (
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
                f"Val Precision={precision:.4f}, Val Recall={recall:.4f}, Val F1={f1:.4f}\n"
            )
            print(log_line, end='')
            f.write(log_line)

    # -------------------------------
    # Save model
    # -------------------------------
    torch.save({
        'conv1_weight': model.conv1_weight.cpu(),
        'conv1_bias': model.conv1_bias.cpu(),
        'conv2_weight': model.conv2_weight.cpu(),
        'conv2_bias': model.conv2_bias.cpu(),
        'fc_weight': model.fc_weight.cpu(),
        'fc_bias': model.fc_bias.cpu()
    }, 'saved_model.pt')

    # -------------------------------
    # Plot metrics
    # -------------------------------
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='x')
    plt.plot(epochs, val_precisions, label='Val Precision', marker='s')
    plt.plot(epochs, val_recalls, label='Val Recall', marker='d')
    plt.plot(epochs, val_f1s, label='Val F1', marker='^')

    plt.title('Training & Validation Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics_pretty.png')
    plt.show()


if __name__ == '__main__':
    main()
