import pandas as pd
import torch
from scipy.optimize import curve_fit
import torch.nn.functional as F
from data.dataset import SeqDataSet
from model.PRC import PRL_Net
import einops
from tqdm import trange
import openpyxl
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('model/models/myNet_1130_142138.pth').to(device)
weight = np.linspace(-0.9, 0.9, 10)

def get_choice(t_dataset,bs=1):
    model.eval()

    x,y,end,t_shapes,labels,ends = t_dataset.x,t_dataset.y,t_dataset.shape_end,t_dataset.shapes,t_dataset.choices,t_dataset.shape_end

    pred_result = []
    label_result = []
    shapes = []

    # 进行推理
    # 获得形状序列对应的决策结果
    for idx in trange(t_dataset.get_len()):
        batch_x, batch_y, shape,label,end = x[idx],y[idx],t_shapes[idx],labels[idx],ends[idx]
        batch_x,batch_y = torch.from_numpy(batch_x).unsqueeze(dim=0),torch.from_numpy(batch_y).unsqueeze(dim=0)
        batch_x = einops.rearrange(batch_x, 'b c l -> b l c')
        batch_y = einops.rearrange(batch_y, 'b c l -> b l c')
        feats = model.rc_step(batch_x,end = end)
        feats = [torch.from_numpy(feat).float().to(device) for feat in feats]
        for feat in feats:
            feat.requires_grad = False
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred_choice = -1
        label_choice = -1

        logits = []

        for t in range(batch_x.shape[1]):
            if pred_choice >0 and label_choice>0:
                break

            if pred_choice==-1:
                pred = model.linear_process(t, feats[t])
                pred = torch.argmax(F.sigmoid(pred),dim=1).cpu().numpy()[0]
                logits.append(pred)
                if pred==1 or pred==2:
                    pred_choice=pred

        if pred_choice !=-1:
            # pred_choice = label_choice
            pred_result.append(pred_choice)
            label_result.append(2-label)
            shapes.append(np.array(shape))

    # 统计每个形状的数量
    shape_num = []
    for i,shape in enumerate(shapes):
        temp = np.zeros(10)
        for j in range(shape.__len__()):
            temp[shape[j]] += 1
        shape_num.append(temp)
    shape_num = np.array(shape_num).astype(int)

    # 进行logLR求和
    sum_weight = np.array([np.round(np.sum(weight*shape_num[i]),1) for i in range(shape_num.shape[0])])

    pred_result = np.array(pred_result)
    label_result = np.array(label_result)

    # acc  = np.sum(pred_result == label_result)/label_result.shape[0]
    return pred_result,sum_weight


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

if __name__ == "__main__":

    test_dataset = SeqDataSet(num=2000,mode=0)
    # 模型推理
    data_y,data_x = get_choice(test_dataset,bs=1)

    data = np.column_stack([data_x, data_y])

    # 建表、分组，以满足拟合的数据格式
    columns = ["x"] + ["y"]
    df = pd.DataFrame(data, columns=columns)

    df['group_key'] = df.iloc[:, :-1].apply(tuple, axis=1)

    grouped = df.groupby('group_key')

    x = []
    y = []

    for group, sub_df in grouped:
        sub_df = sub_df.to_numpy()
        p = (np.sum(sub_df[:,1]==1))/sub_df.shape[0]
        x.append(sub_df[0, :1])
        y.append(p)

    x = np.array(x)[:,0]
    y = np.array(y)

    # 拟合 Logistic 曲线
    popt, pcov = curve_fit(logistic, x, y, p0=[1, 2, 0])  # 初始参数猜测
    L, k, x0 = popt

    # 生成拟合曲线
    x_fit = np.linspace(-2, 2, 500)
    y_fit = logistic(x_fit, L, k, x0)

    # 绘图
    plt.scatter(x, y, marker='o', color='green', label="Data points")
    plt.plot(x_fit, y_fit, color='black', label="Fitted curve")  # 拟合曲线
    plt.xlabel("Accumulated evidence for left target (total logLR)")
    plt.ylabel("Probability of choosing the left target")
    plt.legend()
    plt.show()
    plt.savefig('figB.png',dpi=600)