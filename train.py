import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model.resnet50 import Resnet50
from utils.loadtrain import CatsDogsTrainDataset
from tqdm import tqdm
import csv
from PIL import Image

def train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda', save_path='./best_resnet152_cats_dogs.pth'):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 创建一个进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (resnetImage, labels, _) in progress_bar:
            resnetInputs = resnetImage.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(resnetInputs, inceptionInputs)
            loss = criterion(outputs, labels)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条描述信息
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        # 打印每个 epoch 的平均 loss 和准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%")


def recordLoss(model, train_loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        # 在加载数据时加上 tqdm 进度条
        for resnetImage, inceptionImage, labels, imgPath in tqdm(train_loader, desc="Processing batches"):

            resnetInputs = resnetImage.to(device)
            inceptionInputs = inceptionImage.to(device)
            labels = labels.to(device)

            outputs = model(resnetInputs, inceptionInputs)
            for i, output in enumerate(outputs):
                label = labels[i]
                loss = criterion(output, label)
                losses.append((imgPath[i], loss.item(), label.item()))  # 转换为 Python 标量

    # 按照损失值降序排序
    sorted_losses = sorted(losses, key=lambda x: x[1], reverse=True)

    # 保存为 CSV 文件
    with open('./sorted_losses.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Loss', 'Label'])  # 写入表头
        writer.writerows(sorted_losses)  # 写入数据


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a ResNet model on Cats vs Dogs dataset.")
    parser.add_argument('--train_dir', type=str, required=False, default='./dogs-vs-cats-redux-kernels-edition/train', help='Path to the training dataset directory.')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--save_path', type=str, default='./ckpt/best_resnet50_cats_dogs.pth', help='Path to save the best model weights.')
    parser.add_argument('--val_split', type=float, default=0, help='Percentage of training data to use as validation.')
    args = parser.parse_args()

    # 数据转换
    transform = {
        # transforms for ResNet50
        'resnet': transforms.Compose([transforms.Resize(250),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

        # transforms for Inception v3
        'inception': transforms.Compose([transforms.Resize(299),
                                         transforms.RandomCrop(299),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    }

    # 创建数据集
    stage1_dataset = CatsDogsTrainDataset(root_dir=args.train_dir, transform=transform)

    # 创建数据加载器1
    train1_loader = DataLoader(stage1_dataset, batch_size=args.batch_size, shuffle=True)

    # 测试实例化和参数冻结情况
    criterion = nn.CrossEntropyLoss()
    device = 'cuda'

    # Train
    num_epochs = 2
    model = Resnet50()
    init_lr = 0.001
    optimizer = torch.optim.SGD([{'params': model.classifier.parameters()},
                                 {'params': model.resnet.layer4.parameters(), 'lr': init_lr * 0.1}],
                                lr=init_lr, momentum=0.91)

    # 打印需要训练的参数数目
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    train(model, train1_loader, criterion, optimizer, num_epochs=num_epochs, save_path=args.save_path)

    torch.save(model.state_dict(), './ckpt/resnet50.pth')