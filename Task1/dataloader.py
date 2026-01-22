import os
import random
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def augment_image(img):
    """只对缺陷样本做增强"""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)

    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    return img

class DefectDataset(Dataset):
    def __init__(self, root, augment=True, defect_aug_times=3):
        self.img_dir = os.path.join(root, "img")
        self.txt_dir = os.path.join(root, "txt")

        self.samples = []
        self.augment = augment


        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),  # 改为 320x320
            transforms.ToTensor(),
        ])
        # ====================

        if not os.path.exists(self.img_dir):
            print(f"Warning: {self.img_dir} does not exist.")
            img_files = []
        else:
            img_files = [
                f for f in os.listdir(self.img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]

        for fname in img_files:
            base, _ = os.path.splitext(fname)
            img_path = os.path.join(self.img_dir, fname)
            txt_path = os.path.join(self.txt_dir, base + ".txt")

            has_defect = os.path.exists(txt_path)
            label = 1 if has_defect else 0

            self.samples.append((img_path, label, False))

            if has_defect and augment: 
                for _ in range(defect_aug_times):
                    self.samples.append((img_path, label, True))

        print(f"Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, do_aug = self.samples[idx]
        img = Image.open(img_path).convert('L') # 保持灰度读取

        if do_aug and self.augment:
            img = augment_image(img)

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label