import os  # 导入操作系统模块，用于路径拼接
import pandas as pd  # 导入 pandas，用于读取 Excel 文件
import numpy as np  # 导入 numpy，用于数值计算
from PIL import Image  # 导入 Pillow 中的 Image 类，用于图像加载与处理
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg'、'Qt5Agg' 等
import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import random
import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入神经网络模块
from torch.utils.data import Dataset, DataLoader, random_split  # 导入数据集及数据加载模块
import torchvision.transforms as transforms  # 导入 torchvision 中的图像预处理工具
# 使用新版 API 加载 EfficientNet_b0 及其预训练权重枚举类
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import precision_score, recall_score  # 导入精确率和召回率的计算函数
from openpyxl import Workbook
from openpyxl.styles import Font

#######################参数修改区############################
excel_path = "odir.xlsx"  # 指定 Excel 文件路径
images_dir = "pre"  # 指定图片存放目录
save_path = "checkpoints"  # 指定模型存放目录
read_rows = 20  # 只读取数据前20行，输入0则全部读取
batch_size = 16  # 每个批次样本数
num_epochs = 2  # 训练周期数（测试时较少周期）
lr = 1e-4  # 学习率
percent = 0.8  # 划分训练集和验证集（80%训练，20%验证）
saved_enabled = True  # 开启保存模式
vision_enabled = True  # 开启视图模式
saved_interval = 2  # 设置模型保存间隔
point_size = 224  # 像素归一化像素大小 正方形
angle = 15  # 数据增强旋转角度
############################################################

# 1. 定义眼疾数据集类
class EyeDataset(Dataset):
    def __init__(self, excel_path, images_dir, transform=None , rows=read_rows):
        """
        excel_path: 包含病人信息及图片文件名的xlsx文件路径
        images_dir: 存放所有眼底图片的目录
        transform: 图像预处理及数据增强方法
        """
        if rows == 0:
            self.data = pd.read_excel(excel_path)  # 读取所有 Excel 文件
        else:
            self.data = pd.read_excel(excel_path, nrows=rows)  # 读取 Excel 文件前20行数据
        self.images_dir = images_dir  # 保存图片目录
        self.transform = transform  # 保存图像预处理方法
        self.label_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # 定义8个类别的标签

    def __len__(self):
        return len(self.data)  # 返回数据集样本总数

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # 获取第 idx 行数据
        id_name = row['ID']  # 获取id
        left_img_name = row['Left-Fundus']  # 获取左眼图片文件名
        right_img_name = row['Right-Fundus']  # 获取右眼图片文件名
        left_img_path = os.path.join(self.images_dir, left_img_name)  # 拼接左眼图片路径
        right_img_path = os.path.join(self.images_dir, right_img_name)  # 拼接右眼图片路径

        left_img = Image.open(left_img_path).convert('RGB')  # 加载左眼图片并转换为 RGB 模式
        right_img = Image.open(right_img_path).convert('RGB')  # 加载右眼图片并转换为 RGB 模式

        if self.transform is not None:
            left_img = self.transform(left_img)  # 对左眼图片进行预处理
            right_img = self.transform(right_img)  # 对右眼图片进行预处理

        labels = row[self.label_cols].values.astype(np.float32)  # 获取标签数据，转换为 float32 数组
        return id_name, left_img, right_img, torch.tensor(labels)  # 返回左右眼图像及对应标签（转换为张量）



# 2. 定义数据预处理和数据增强方式
train_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(point_size),  # 从中心裁剪224x224
    transforms.RandomRotation(angle),  # 随机旋转图像，角度范围 ±15 度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 均值归一化
                         std=[0.229, 0.224, 0.225])  # 使用 ImageNet 标准差归一化
])
val_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(point_size),  # 从中心裁剪224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 均值归一化
                         std=[0.229, 0.224, 0.225])  # 标准差归一化
])

#像素归一化
point_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(point_size),  # 从中心裁剪224x224
    transforms.ToTensor(),  # 转换为张量
])
#RGB归一化
RGB_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(point_size),  # 从中心裁剪224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 均值归一化
                         std=[0.229, 0.224, 0.225])  # 使用 ImageNet 标准差归一化
])
#数据增强
data_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(point_size),  # 从中心裁剪224x224
    transforms.RandomRotation(angle),  # 随机旋转图像，角度范围 ±15 度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ToTensor(),  # 转换为张量
])

def vision(image_paths, visions= True):
    #image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # 替换为实际图片路径
    if visions:
        # 获取所有图片文件
        all_images = [img for img in os.listdir(image_paths) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 随机选择三张图片
        image_paths = random.sample(all_images, 3) if len(all_images) >= 3 else all_images
        fig, axes = plt.subplots(len(image_paths), 4, figsize=(12, 12))
        for i, img_path in enumerate(image_paths):
            img_paths = os.path.join(images_dir, img_path)
            original = Image.open(img_paths).convert('RGB')
            point_t = point_transform(original)
            rgb_t = RGB_transform(original)
            data_t = data_transform(original)

            point_img = point_t.permute(1, 2, 0).numpy()
            rgb_img = rgb_t.permute(1, 2, 0).numpy()
            data_img = data_t.permute(1, 2, 0).numpy()

            point_img = np.clip( point_img, 0, 1)  # 将值裁剪到 [0, 1] 范围内
            rgb_img = np.clip(rgb_img, 0, 1)  # 将值裁剪到 [0, 1] 范围内
            data_img = np.clip(data_img, 0, 1)  # 将值裁剪到 [0, 1] 范围内

            # 显示图像
            axes[i, 0].imshow(original)
            axes[i, 0].set_title("原图")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(point_img)
            axes[i, 1].set_title("像素归一化")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(rgb_img)
            axes[i, 2].set_title("RGB归一化")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(data_img)
            axes[i, 3].set_title("数据增强")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()
# 3. 定义基于 EfficientNet 的双路网络，用于同时处理左右眼图像
class DualEfficientNet(nn.Module):
    def __init__(self, num_classes=8):
        super(DualEfficientNet, self).__init__()
        # 使用新版 API 加载预训练 EfficientNet_b0 模型
        self.eff_left = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.eff_right = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.eff_left.classifier = nn.Identity()  # 移除左眼模型的分类层
        self.eff_right.classifier = nn.Identity()  # 移除右眼模型的分类层
        self.fc = nn.Linear(1280 * 2, num_classes)  # 全连接层，将左右眼特征拼接后映射到目标类别

    def forward(self, img_left, img_right):
        feat_left = self.eff_left(img_left)  # 获取左眼特征，形状 (B, 1280)
        feat_right = self.eff_right(img_right)  # 获取右眼特征，形状 (B, 1280)
        features = torch.cat([feat_left, feat_right], dim=1)  # 拼接左右眼特征，形状 (B, 2560)
        out = self.fc(features)  # 全连接层输出 logits
        return out  # 返回模型输出
#模型保存
def save_model(model, epoch, save_interval, save_enabled, save_dir= save_path, test_ids=None, true_labels=None,
               pred_labels=None):
    """
    保存模型的函数。
    :param model: 需要保存的模型
    :param epoch: 当前训练的 epoch
    :param save_interval: 每隔多少个 epoch 保存一次
    :param save_enabled: 是否启用模型保存
    :param save_path: 保存模型的目录
    """
    if save_enabled and epoch % save_interval == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_filename = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"模型已保存: {model_filename}")
        # 创建 Excel 文件
        wb = Workbook()
        ws = wb.active
        ws.title = "Predictions"
        ws.append(["id", "N", "D", "G", "C", "A", "H", "M", "O"])  # 表头

        # 填充数据
        num_samples = len(test_ids)
        for i in range(num_samples):
            sample_id = test_ids[i]
            pred_row = pred_labels[i]
            true_row = true_labels[i]

            # 先写入 ID 和预测值
            row = [sample_id] + pred_row
            ws.append(row)

            # 比对预测值和真实值，错误的标红
            for col in range(2, 10):  # 预测值从第 2 列到第 9 列
                if pred_row[col - 2] != true_row[col - 2]:  # 检查对应真实值是否相同
                    ws.cell(row=i + 2, column=col).font = Font(color="FF0000")  # 标红错误预测值

        # 保存 Excel 文件
        excel_filename = os.path.join(save_dir, f"predictions_epoch_{epoch}.xlsx")
        wb.save(excel_filename)
        print(f"预测结果已保存: {excel_filename}")





# 4. 定义训练和验证函数，仅计算 loss、精确率和召回率
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, device='cuda',save_interval=saved_interval, save_enabled=saved_enabled):
    train_losses = []  # 记录训练 loss
    val_losses = []  # 记录验证 loss
    val_precisions = []  # 记录验证阶段精确率
    val_recalls = []  # 记录验证阶段召回率

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化当前 epoch 累计训练 loss
        for _, img_left, img_right, labels in train_loader:
            img_left = img_left.to(device)  # 将左眼图像移动到设备
            img_right = img_right.to(device)  # 将右眼图像移动到设备
            labels = labels.to(device)  # 将标签移动到设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(img_left, img_right)  # 前向传播，得到 logits
            loss = criterion(outputs, labels)  # 计算 loss（采用 BCEWithLogitsLoss）
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            running_loss += loss.item() * img_left.size(0)  # 累加当前批次 loss

        epoch_loss = running_loss / len(train_loader.dataset)  # 计算当前 epoch 的平均训练 loss
        train_losses.append(epoch_loss)  # 保存训练 loss

        # 开始验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0  # 初始化验证 loss 累计值
        all_labels = []  # 保存所有验证样本的真实标签
        all_outputs = []  # 保存所有验证样本的输出 logits
        all_id =[]  #保存所有患者id
        with torch.no_grad():
            for id, img_left, img_right, labels in val_loader:
                img_left = img_left.to(device)
                img_right = img_right.to(device)
                labels = labels.to(device)
                outputs = model(img_left, img_right)
                loss = criterion(outputs, labels)  # 计算验证 loss
                val_loss += loss.item() * img_left.size(0)
                all_id.append(id)
                all_labels.append(labels.cpu().numpy())  # 保存真实标签
                all_outputs.append(outputs.cpu().numpy())  # 保存模型输出

        epoch_val_loss = val_loss / len(val_loader.dataset)  # 计算平均验证 loss
        val_losses.append(epoch_val_loss)

        # 合并所有验证批次数据
        all_labels = np.vstack(all_labels)  # 形状 (N, 8)
        all_outputs = np.vstack(all_outputs)  # 形状 (N, 8)
        all_id = np.vstack(all_id)
        all_probs = 1.0 / (1.0 + np.exp(-all_outputs))  # 对 logits 应用 sigmoid，得到概率
        all_pred_labels = (all_probs > 0.5).astype(int)  # 根据阈值0.5二值化

        print(f"预测值为\n{all_pred_labels}")
        print(f"真实值为\n{all_labels.astype(int)}")

        # 计算精确率和召回率（将所有标签展平后计算 micro 平均）
        precision = precision_score(all_labels.flatten(), all_pred_labels.flatten(),  zero_division=0)
        recall = recall_score(all_labels.flatten(), all_pred_labels.flatten(), zero_division=0)
        val_precisions.append(precision)
        val_recalls.append(recall)

        # 打印当前 epoch 的指标
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        # 调用保存模型函数
        all_labels_list = all_labels.astype(int).tolist()  # 转换为列表
        all_pred_labels_list = all_pred_labels.tolist()  # 转换为列表
        all_id_list = all_id.flatten().tolist()  # 转换为列表
        print(type(all_id_list),type(all_labels_list),type(all_pred_labels_list))
        save_model(model, epoch + 1, save_interval, save_enabled, save_path, all_id_list, all_labels_list, all_pred_labels_list)

    # 绘制训练/验证 loss 曲线和验证阶段精确率、召回率曲线
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 6))

    # 子图1：训练和验证 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & Val Loss')
    plt.legend()

    # 子图2：验证阶段精确率与召回率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_precisions, label='Precision', marker='o', color='orange')
    plt.plot(epochs, val_recalls, label='Recall', marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision & Recall')
    plt.legend()

    plt.tight_layout()  # 自动调整子图间间距
    plt.show()  # 展示图像

    return model, train_losses, val_losses, val_precisions, val_recalls


# 5. 主程序：加载数据、划分数据集、初始化模型并开始训练
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU（若可用）或 CPU

    full_dataset = EyeDataset(excel_path, images_dir, transform=point_transform, rows=read_rows)  # 创建完整数据集对象
    vision(images_dir, vision_enabled)

    # 划分训练集和验证集（80%训练，20%验证）
    train_size = int(percent * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_extra = EyeDataset(excel_path, images_dir, transform=data_transform, rows=read_rows) #数据增强数据集
    train_dataset = train_dataset+train_extra

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 创建训练数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # 创建验证数据加载器

    model = DualEfficientNet(num_classes=8).to(device)  # 初始化模型并移动至设备
    criterion = nn.BCEWithLogitsLoss()  # 定义多标签分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 定义 Adam 优化器

    # 开始训练并评估，返回模型及各项指标
    model, train_losses, val_losses, val_precisions, val_recalls = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, device=device)
