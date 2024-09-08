import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm  # 导入 tqdm
from model.resnet50 import Resnet50
from utils.loadtest import CatsDogsTestDataset
import torch.nn.functional as F


def load_model(model_path, device='cuda'):
    model = Resnet50()  # 假设我们有2个类别：猫和狗
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model


def predict(model, test_loader, device='cuda'):
    model = model.to(device)

    with torch.no_grad():
        # 使用 tqdm 创建一个进度条
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches")
        df = pd.DataFrame(columns=['id', 'label'])

        for idx, (resnetInput, id) in progress_bar:
            resnetInput = resnetInput.to(device)
            outputs = model(resnetInput)
            dog_probs = F.softmax(outputs, dim=1).data[:,1]
            # _, predicted = torch.max(outputs, 1)
            df = df.append(pd.DataFrame({'id': id, 'label': dog_probs.cpu()}), ignore_index=True)

            # 更新进度条描述信息
            progress_bar.set_postfix(batch_idx=idx)

    return df


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Predict using a pre-trained ResNet model.")
    parser.add_argument('--test_dir', type=str, required=False, default='./dogs-vs-cats-redux-kernels-edition/test', help='Path to the test dataset directory.')
    parser.add_argument('--model_path', type=str, required=False, default='./ckpt/resnet50.pth', help='Path to the trained model (.pth file).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing.')
    args = parser.parse_args()

    # 图像预处理
    transform = transforms.Compose([transforms.Resize(250),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

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
    df = predict(model, test_loader, device)

    csv_file = 'submission.csv'
    df.sort_values(by='id', inplace=True)
    df.to_csv(csv_file, index=False)
    print(df)

    print("结果已保存到 'submission.csv' 文件中。")
