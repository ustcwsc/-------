这是一个 二分类图像识别项目（Task1），使用**手写 CNN（不依赖 PyTorch 自动求导）**完成模型训练、评估和结果可视化。

本 README 面向 第一次接触项目的同学，请严格按步骤操作。

📁 项目结构说明
PB23050934_mla_project/
│
├── Task1/
│   ├── dataloader.py          # 读取图像数据和标签
│   ├── model.py               # CNN 模型（手写 forward / backward）
│   ├── utils.py               # Precision / Recall / F1 计算
│   ├── main.py                # 主程序（训练 + 测试 + 可视化）
│   ├── saved_model.pt         # 训练后保存的模型参数
│   ├── score.txt              # 训练日志（超参数 + 指标 + 网络结构）
│   └── training_metrics_pretty.png  # 训练过程指标曲线图
│
├── Task2/                     # 另一个任务（暂不需要）
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略文件
└── README.md                  # 项目说明（本文件）

🧠 项目做了什么？

使用 320*320 灰度图像

构建一个 两层卷积 + 一层全连接的 CNN

手动实现：

前向传播

反向传播

SGD 参数更新

在训练过程中：

计算 Loss、Accuracy

在验证集上计算 Precision / Recall / F1

画出随 Epoch 变化的曲线

保存：

训练好的模型参数

所有训练指标和网络结构信息

✅ 一、运行前准备（必须做）
1️⃣ 安装 Python（建议 3.9 或以上）

在命令行检查：

python --version

2️⃣ 安装依赖（只需一次）

在项目根目录运行：

pip install -r requirements.txt


如果失败，可以手动安装：

pip install torch matplotlib numpy pillow

📂 二、数据集准备（非常重要）

你的数据目录结构应类似：

dataset/
├── train/
│   ├── img/        # 有缺陷图像（label = 1）
│   └── txt/        # 正常图像（label = 0）
│
└── val/
    ├── img
    └── txt


📌 注意：

文件名随意

图片必须能被 PIL 打开

dataloader.py 会根据文件夹名自动赋标签

▶️ 三、如何运行项目

进入 Task1 所在目录：
例如：
cd "C:\Users\35013\Desktop\PB23050934_mla_project\Task1"


运行主程序（示例路径，按你自己数据路径改）：

python main.py ^
  --train_data_path "E:\dataset\train" ^
  --val_data_path   "E:\dataset\val"


如果你是 Mac / Linux：

python main.py \
  --train_data_path /path/to/train \
  --val_data_path   /path/to/val

📊 四、运行后你会得到什么？
1️⃣ 命令行输出（每个 Epoch）
Epoch 1: Loss=0.3815, Accuracy=0.8409, Val Precision=0.62, Val Recall=0.55, Val F1=0.58

2️⃣ score.txt（重要！）

内容包括：

学习率、Epoch 数


每个 Epoch 的训练 / 验证指标




📌 这是实验报告中必须写的内容

3️⃣ 训练过程曲线图

文件名：

training_metrics_pretty.png


图中包含：

训练 Loss

训练 Accuracy

验证 Precision / Recall / F1