import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms


class CatsDogsTrainDatasetStage2(Dataset):
    def __init__(self, csv_file, transform=None):
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        # 过滤第二列 "Loss" 大于 0.5 的行
        self.data = self.data[self.data['Loss'] > 0.5]
        self.transform = transform

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取图像路径和损失值
        img_path = self.data.iloc[idx, 0]
        loss = self.data.iloc[idx, 1]

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 应用转换（如果提供）
        if self.transform:
            image = self.transform(image)

        return image, loss


