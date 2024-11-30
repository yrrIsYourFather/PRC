#from model_onelinear import MyNet
from model.PRL import PRL_Net
from data.dataset import generator_t
from torch.utils.data import DataLoader
import einops
import torch
import datetime

# 获取当前时间
now = datetime.datetime.now()

# 格式化时间为 "月+日+时刻" 的形式
formatted_time = now.strftime("%m%d_%H%M%S")

# 模型文件名
filename = f"model/models/myNet_{formatted_time}.pth"

# 使用 gpu 训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(num_trails=1000):

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

    # 切片成训练集和测试集
    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]

    X_test = X_shuffled[train_size:]
    y_test = y_shuffled[train_size:]

    ############################################################

    myNet = PRL_Net(T=70, n_blocks=2)
    print("Start training!!!")
    myNet.fit(X_train, y_train, device=device)
    torch.save(myNet, filename)
    
    print("Start evaluation!!!")
    myNet.evaluate(X_test, y_test, device=device)



    # # 转为torch封装
    # t_dataset = train_dataset(x,y)
    # dataloader = DataLoader(t_dataset, num_workers=8, batch_size=32, shuffle=True)

    # for idx, data in enumerate(dataloader):
    #     batch_x,batch_y=data['x'],data['y']
    #     batch_x = einops.rearrange(batch_x,'b c l -> b l c')
    #     batch_y = einops.rearrange(batch_y, 'b c l -> b l c')

    #     myNet.fit(batch_x, batch_y)



if __name__ == "__main__":

    train(num_trails=5)#统一用4000训练