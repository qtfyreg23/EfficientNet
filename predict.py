import torch
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np  # 导入 numpy，用于数值计算
from efficientnet import DualEfficientNet

# 定义模型路径和测试图片目录
model_path = 'checkpoints/model_epoch_2.pth'  # 训练好的模型路径
test_images_dir = 'pre'  # 测试图片目录
output_csv_path = 'output.csv'  # 输出 CSV 文件路径

# 定义图片预处理方式
test_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(224),  # 从中心裁剪224x224
    transforms.ToTensor(),  # 转换为张量

])


# 定义加载训练好的模型
def load_model(model_path):
    model = DualEfficientNet(num_classes=8)  # 初始化模型
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置模型为评估模式
    return model


# 预测函数，返回预测结果
def predict(model, left_img_path, right_img_path):
    # 加载并预处理左右眼图像
    left_img = Image.open(left_img_path).convert('RGB')
    right_img = Image.open(right_img_path).convert('RGB')

    left_img = test_transform(left_img).unsqueeze(0)  # 添加 batch 维度
    right_img = test_transform(right_img).unsqueeze(0)  # 添加 batch 维度

    # 将图像输入模型
    with torch.no_grad():
        left_features = model.eff_left(left_img)  # 获取左眼图像的特征
        right_features = model.eff_right(right_img)  # 获取右眼图像的特征
        features = torch.cat((left_features, right_features), dim=1)  # 拼接特征

        # 通过一个全连接层进行分类
        output = model.fc(features)  # 假设模型中有一个全连接层（具体要根据模型结构调整）
        probabilities = torch.sigmoid(output)  # 使用 sigmoid 获取各类的概率值

    return probabilities.squeeze().numpy()  # 返回概率值


# 处理测试数据并输出结果
def process_test_data(model, test_images_dir, output_csv_path):
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []  # 存储预测结果
    results_dict = {}
    for img_name in test_images:
        # 假设文件名格式为：ID_left.jpg 和 ID_right.jpg
        id_name = int(img_name.split('_')[0])
        left_img_path = os.path.join(test_images_dir, f"{id_name}_left.jpg")
        right_img_path = os.path.join(test_images_dir, f"{id_name}_right.jpg")
        # 进行预测
        probabilities = predict(model, left_img_path, right_img_path)
        all_probabilities = (probabilities > 0.5).astype(int)
        # 将预测结果和 ID 存储
        if id_name not in results_dict:
            results_dict[id_name] = all_probabilities
        else:
            # 如果 ID 已经存在，合并左右眼的预测结果
            results_dict[id_name] = np.maximum(results_dict[id_name], all_probabilities)
    # 最终结果
    # 按照 id_name 从小到大排序
    sorted_results = sorted(results_dict.items(), key=lambda x: x[0])
    for id_name, probabilities in sorted_results:
        results.append([id_name] + probabilities.tolist())
    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(results, columns=['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    df.to_csv(output_csv_path, index=False)
    print(f"预测结果已保存到 {output_csv_path}")


#    主程序
if __name__ == "__main__":
    # 加载训练好的模型
    model = load_model(model_path)

    # 处理测试数据并生成预测结果
    process_test_data(model, test_images_dir, output_csv_path)
