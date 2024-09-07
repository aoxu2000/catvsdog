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
            image = self.transform(image)

        return image, label, img_path


if __name__ == '__main__':
    # 定义图像的预处理变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
    ])

    # 数据集路径
    train_dir = './dogs-vs-cats-redux-kernels-edition/train'

    # 创建数据集和数据加载器
    train_dataset = CatsDogsDataset(root_dir=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 测试数据加载器
    for images, labels in train_loader:
        print(images.shape)  # 打印批量的图像形状
        print(labels)  # 打印批量的标签
        break  # 仅查看第一个批次
