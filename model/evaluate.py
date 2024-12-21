#from model_onelinear import MyNet
from model.PRC import PRL_Net
from data.dataset import generator_t
from torch.utils.data import DataLoader
import einops
import torch
import datetime

# 获取当前时间
now = datetime.datetime.now()

# 格式化时间为 "月+日+时刻" 的形式
formatted_time = now.strftime("%m%d_%H%M%S")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(num_trails=1000):

    print("Start dataset preparation!!!")
    # 模拟产生数据集
    G = generator_t()
    X, y, _, _, _ = G.generate(numTrials=num_trails)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    X_tensor = torch.transpose(X_tensor, 1, 2)
    y_tensor = torch.transpose(y_tensor, 1, 2)

    # 划分训练集和测试集
    # 计算划分索引
    train_size = int(0.8 * X_tensor.size(0))  # 80%的训练集
    test_size = X_tensor.size(0) - train_size  # 20%的测试集

    # 随机打乱数据（可选）
    indices = torch.randperm(X_tensor.size(0))  # 生成随机索引
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]

    ############################################################
    print("Start evaluation!!!")
    myNet = torch.load("model/models/myNet_1130_142138.pth").to(device)
    myNet.evaluate(X_shuffled, y_shuffled, device=device)



if __name__ == "__main__":

    evaluate(num_trails=5000)#测试