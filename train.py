import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model.resnet import ResNet152
from utils.loadtrain import CatsDogsTrainDataset
from tqdm import tqdm

def train_resnet(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', save_path='./best_resnet152_cats_dogs.pth'):
    model = model.to(device)
    best_val_accuracy = 0.0  # 初始化最优验证准确率

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 创建一个进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
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

        # 验证集评估并保存最优模型
        val_accuracy = validate(model, val_loader, criterion, device)

        # 如果当前 epoch 的验证准确率更好，则保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Validation Accuracy: {best_val_accuracy:.2f}%")

def validate(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a ResNet model on Cats vs Dogs dataset.")
    parser.add_argument('--train_dir', type=str, required=False, default='./dogs-vs-cats-redux-kernels-edition/train', help='Path to the training dataset directory.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--save_path', type=str, default='./best_resnet152_cats_dogs.pth', help='Path to save the best model weights.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Percentage of training data to use as validation.')
    args = parser.parse_args()

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 通常使用 224x224 的输入
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 创建数据集
    train_dataset = CatsDogsTrainDataset(root_dir=args.train_dir, transform=transform)

    # 划分训练集和验证集
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 实例化模型、损失函数和优化器
    model = ResNet152(num_classes=2)  # 对于二分类任务，猫和狗
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 开始训练
    train_resnet(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs, save_path=args.save_path)
