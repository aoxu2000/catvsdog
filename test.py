import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from model.resnet import ResNet50
from utils.load import CatsDogsDataset


def load_model(model_path, device='cuda'):
    model = ResNet50(num_classes=2)  # 假设我们有2个类别：猫和狗
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model


def predict(model, test_loader, device='cuda'):
    predictions = []
    model = model.to(device)

    with torch.no_grad():
        for idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


if __name__ == "__main__":
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 通常使用 224x224 的输入
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 数据集路径
    test_dir = './dogs-vs-cats-redux-kernels-edition/test'

    # 创建测试集和数据加载器
    test_dataset = CatsDogsDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 加载预训练模型
    model_path = './model.pth'  # 训练时保存的模型文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # 获取预测结果
    predictions = predict(model, test_loader, device)

    # 创建 CSV 文件
    results = pd.DataFrame({
        'id': range(1, len(predictions) + 1),
        'label': predictions
    })
    results.to_csv('submission.csv', index=False)

    print("结果已保存到 'submission.csv' 文件中。")
