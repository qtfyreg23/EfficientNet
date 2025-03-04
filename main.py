import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 设置数据集路径
XLSX_FILE = 'odir.xlsx'  # 眼底疾病的标签文件
IMAGE_FOLDER = 'pre'  # 预处理后的图像文件夹
MODEL_SAVE_PATH = 'efficientnet_model.pth'  # 训练好的模型保存路径

# 设备选择，支持 GPU 加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标签类别映射，将疾病类别映射到整数索引
labels_dict = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}


# 读取图像数据
def load_images():
    image_dict = {}  # 存储文件名对应的图片矩阵
    for filename in os.listdir(IMAGE_FOLDER):  # 遍历文件夹中的所有图像文件
        file_path = os.path.join(IMAGE_FOLDER, filename)  # 获取完整路径
        image_dict[filename] = np.array(Image.open(file_path))  # 读取图像并转换为 NumPy 数组
    return image_dict

# 读取标签数据
def load_labels():
    df = pd.read_excel(XLSX_FILE,nrows=20)  # 读取整个Excel文件
    image_label_map = {}
    for index, row in df.iterrows():
        left_eye_filename = row['Left-Fundus']
        right_eye_filename = row['Right-Fundus']
        label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        label_values = [row[column] for column in label_columns]
        label = tuple(label_values)

        if left_eye_filename in image_label_map:
            # 如果左眼文件名已经在字典中，将右眼文件名和对应的标签元组追加到值中
            image_label_map[left_eye_filename].append((right_eye_filename, label))
        else:
            # 否则，创建一个新的键值对
            image_label_map[left_eye_filename] = [(right_eye_filename, label)]

        if right_eye_filename in image_label_map:
            # 如果右眼文件名已经在字典中，将左眼文件名和对应的标签元组追加到值中
            image_label_map[right_eye_filename].append((left_eye_filename, label))
        else:
            # 否则，创建一个新的键值对
            image_label_map[right_eye_filename] = [(left_eye_filename, label)]
    return image_label_map

# 组合数据，将图像数据和标签匹配
def combine_data(image_dict, fundus_labels_dict):
    return {key: [image_dict[key], fundus_labels_dict[key]] for key in fundus_labels_dict if key in image_dict}

# 定义自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # 存储图像数据
        self.labels = labels  # 存储图像对应的标签
        self.transform = transform  # 图像变换

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])  # 从 NumPy 数组转换为 PIL 图像
        label = torch.tensor(self.labels[idx])  # 转换标签为 Tensor
        if self.transform:
            img = self.transform(img)  # 进行图像变换
        return img, label  # 返回图像和标签

# 获取 DataLoader，用于批量加载数据
def get_data_loader(data, labels, batch_size=16, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 PyTorch Tensor
    ])
    dataset = MyDataset(data, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 定义 EfficientNet 模型
class MyEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(MyEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # 加载 EfficientNet 预训练模型
        self.fc = nn.Linear(1000, num_classes)  # 替换最后一层，使其适应当前任务类别数

    def forward(self, x):
        x = self.efficientnet(x)  # 经过 EfficientNet 提取特征
        x = self.fc(x)  # 通过全连接层分类
        return x

# 训练模型
def train_model(model, train_loader, num_epochs=1):
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

    model.to(device)  # 将模型转移到 GPU 或 CPU
    model.train()  # 设定模型为训练模式

    for epoch in range(num_epochs):
        total_loss, correct_predictions = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)  # 迁移数据到 GPU 或 CPU
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # 计算预测结果
            correct_predictions += (predicted == labels).sum().item()  # 统计正确个数

        epoch_loss = total_loss / len(train_loader)  # 计算平均损失
        epoch_acc = correct_predictions / len(train_loader.dataset)  # 计算准确率
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)  # 保存模型
    print(f"Model saved to {MODEL_SAVE_PATH}")

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()  # 设置为评估模式
    all_labels, all_predictions = [], []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 迁移到 GPU 或 CPU
        with torch.no_grad():  # 关闭梯度计算
            outputs = model(inputs)  # 进行预测
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
        all_labels.extend(labels.cpu().numpy())  # 记录真实标签
        all_predictions.append(predicted.item())  # 记录预测标签

    accuracy = accuracy_score(all_labels, all_predictions)  # 计算准确率
    confusion = confusion_matrix(all_labels, all_predictions)  # 计算混淆矩阵
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{confusion}')

# 主函数
def main():
    # 打印提示信息，通知用户正在加载图像和标签数据
    print("Loading images and labels...")

    # 调用 `load_images()` 函数，加载并返回所有图像数据的字典，其中键是图像文件名，值是图像矩阵
    image_dict = load_images()

    # 调用 `load_labels()` 函数，加载标签数据，返回一个字典，其中键是图像文件名，值是对应的疾病标签
    fundus_labels_dict = load_labels()

    # 调用 `combine_data()` 函数，将 `image_dict` 和 `fundus_labels_dict` 组合，返回包含图像矩阵和对应标签的字典
    combined_dict = combine_data(image_dict, fundus_labels_dict)

    # 统计每个类别的样本数量
    # `combined_dict.values()` 获取所有图像-标签对
    # `x[1]` 获取每个样本的标签
    # `Counter()` 统计每种疾病的出现次数，返回一个字典-like 结构，键是疾病标签，值是对应样本数量
    label_counts = Counter([x[1] for x in combined_dict.values()])

    # 遍历 `label_counts`，打印每种疾病类别对应的图像数量
    for label, count in label_counts.items():
        print(f'{label}的图片共有: {count} 张')  # 输出格式示例： "N的图片共有: 2928 张"

    # 打印所有不同类别的总数
    print(f'总共有 {len(label_counts)} 个类别')  # 例如："总共有 8 个类别"

    # 将 `combined_dict` 中的图像数据提取出来，形成 `data_pic`，用于训练和测试
    data_pic = [x[0] for x in combined_dict.values()]

    # 将 `combined_dict` 中的标签转换为对应的整数索引，形成 `data_lab`，用于训练和测试
    data_lab = [labels_dict[x[1]] for x in combined_dict.values()]

    # 使用 `train_test_split` 将数据集划分为训练集 (80%) 和测试集 (20%)
    # `random_state=42` 确保每次运行的划分结果一致
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_pic, data_lab, test_size=0.2, random_state=42
    )

    # 通过 `get_data_loader()` 函数构建训练数据的 DataLoader
    # `batch_size=16` 代表一次训练时使用 16 张图片
    # `shuffle=True` 代表每个 epoch 训练时随机打乱数据，提高泛化能力
    train_loader = get_data_loader(train_data, train_labels, batch_size=16, shuffle=True)

    # 通过 `get_data_loader()` 函数构建测试数据的 DataLoader
    # `batch_size=1` 代表测试时一次只传入 1 张图片
    # `shuffle=False` 代表测试集不打乱，保证结果的稳定性
    test_loader = get_data_loader(test_data, test_labels, batch_size=1, shuffle=False)

    # 计算分类任务的类别数，`label_counts` 中的键数即为类别总数
    num_classes = len(label_counts)

    # 创建 `MyEfficientNet` 模型，传入类别数 `num_classes`
    model = MyEfficientNet(num_classes)

    # 训练模型，调用 `train_model()` 函数
    # 传入训练数据 `train_loader`，训练 10 轮 (`num_epochs=10`)
    train_model(model, train_loader, num_epochs=10)

    # 加载训练好的模型权重
    # `torch.load(MODEL_SAVE_PATH, map_location=device)` 读取保存在 `MODEL_SAVE_PATH` 位置的模型参数
    # `map_location=device` 允许在 CPU 上加载 GPU 训练的模型（或相反）
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # 将模型移动到 `device`（CPU 或 GPU）
    model.to(device)

    # 评估模型，调用 `evaluate_model()` 函数，在测试集 `test_loader` 上进行预测和评估
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    main()  # 运行主函数
