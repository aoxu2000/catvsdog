import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CatsDogsTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 图片文件所在的目录路径.
            transform (callable, optional): 对样本应用的可选转换（数据增强和预处理）.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        # 从文件名中提取标签
        if 'cat' in img_name:
            label = 0  # 代表猫
        elif 'dog' in img_name:
            label = 1  # 代表狗
        else:
            raise ValueError("图片文件名中必须包含 'cat' 或 'dog' 关键字")

        if self.transform:
            resnetImage = self.transform['resnet'](image)

        return resnetImage, label, img_path


