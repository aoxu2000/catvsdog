import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm  # 导入 tqdm
from model.resnet import ResNet152
from utils.loadtest import CatsDogsTestDataset


def load_model(model_path, device='cuda'):
    model = ResNet152(num_classes=2)  # 假设我们有2个类别：猫和狗
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model


def predict(model, test_loader, device='cuda'):
    predictions = []
    model = model.to(device)

    with torch.no_grad():
        # 使用 tqdm 创建一个进度条
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches")

        for idx, inputs in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

            # 更新进度条描述信息
            progress_bar.set_postfix(batch_idx=idx)

    return predictions


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Predict using a pre-trained ResNet model.")
    parser.add_argument('--test_dir', type=str, required=False, default='./dogs-vs-cats-redux-kernels-edition/test', help='Path to the test dataset directory.')
    parser.add_argument('--model_path', type=str, required=False, default='./ckpt/resnet152_cats_dogs.pth', help='Path to the trained model (.pth file).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing.')
    args = parser.parse_args()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 通常使用 224x224 的输入
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 数据集路径
    test_dir = args.test_dir

    # 创建测试集和数据加载器
    test_dataset = CatsDogsTestDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载预训练模型
    model_path = args.model_path
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
