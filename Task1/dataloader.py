import os
import random
from PIL import Image, ImageEnhance
import torch


def augment_image(img):
    """
    对缺陷图像进行随机数据增强
    """
    # 随机翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 随机旋转
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)

    # 亮度 / 对比度扰动
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    return img


def load_images_and_labels(data_path, device, augment=True, defect_aug_times=4):
    """
    data_path/
      ├── img/
      └── txt/
    """

    img_dir = os.path.join(data_path, "img")
    txt_dir = os.path.join(data_path, "txt")

    data = []
    img_ext = ('.png', '.jpg', '.jpeg', '.bmp')

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(img_ext):
            continue

        img_path = os.path.join(img_dir, fname)
        base, _ = os.path.splitext(fname)
        txt_path = os.path.join(txt_dir, base + '.txt')

        # 标签定义（严格按题目）
        has_defect = os.path.exists(txt_path)
        label = 1 if has_defect else 0

        img = Image.open(img_path).convert('L')

        # 原始样本
        def process(img_pil):
            img_pil = img_pil.resize((224, 224))
            pixels = torch.tensor(list(img_pil.getdata()), dtype=torch.float32) / 255.0
            return pixels.view(1, 224, 224).to(device)

        data.append((process(img), label))

        # 对缺陷样本进行增强式过采样
        if augment and has_defect:
            for _ in range(defect_aug_times):
                aug_img = augment_image(img.copy())
                data.append((process(aug_img), label))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data
