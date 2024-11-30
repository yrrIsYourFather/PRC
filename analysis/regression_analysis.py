import random
from time import sleep

import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from data.dataset import SeqDataSet,generator_t
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm, trange
import openpyxl

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split

import datetime

# 获取当前时间
now = datetime.datetime.now()
# 格式化时间为 "月+日+时刻" 的形式
formatted_time = now.strftime("%m%d_%H%M%S")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = torch.load('models/myNet_1125_113549.pth').to(device)


def regression(X,y):
    # Ridge 回归用于 Q 的估计
    alpha_range = np.logspace(-3, 3, 50)  # 超参数范围
    best_alpha = None
    best_score = -np.inf

    for alpha in alpha_range:
        ridge = Ridge(alpha=alpha)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(ridge, X, y, scoring='neg_mean_squared_error', cv=kf)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    # 使用最佳 alpha 重新拟合模型
    #ridge = Ridge(alpha=best_alpha)
#     ridge = Ridge(alpha=best_alpha, fit_intercept=False)
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X, y)

    # 提取结果
    beta_0 = ridge.intercept_  # 偏置项
    beta = ridge.coef_  # 各形状的权重

    # 打印结果
    print(f"最佳 alpha: {best_alpha}")
    print(f"偏置 β0: {beta_0}")
    print(f"权重 β1-10: {beta}")


def get_choice(t_dataset, bs=1, is_pred=True):

    model.eval()
    #model.module.eval()

    # 模拟产生数据集
    dataloader = DataLoader(t_dataset, num_workers=8, batch_size=bs, shuffle=True)

    x,y,end,t_shapes = t_dataset.x,t_dataset.y,t_dataset.shape_end,t_dataset.shapes


    correct = 0
    len = 0

    pred_result = []
    label_result = []
    shapes = []

    if_make_decision = [] #记录这一个样本模型是否做出决策


    for idx in trange(dataloader.__len__()):
        batch_x, batch_y, shape = x[idx],y[idx],t_shapes[idx]
        batch_x,batch_y = torch.from_numpy(batch_x).unsqueeze(dim=0),torch.from_numpy(batch_y).unsqueeze(dim=0)
        batch_x = einops.rearrange(batch_x, 'b c l -> b l c')
        batch_y = einops.rearrange(batch_y, 'b c l -> b l c')
#         feats = model.module.rc_step(batch_x)
        feats = model.rc_step(batch_x)
        feats = [torch.from_numpy(feat).float().to(device) for feat in feats]
        # for feat in feats:
        #     feat.requires_grad = False
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        #
        pred_choice = -1
        label_choice = -1
        #
        for t in range(batch_x.shape[1]):
            if pred_choice >0 and label_choice>0:
                break

            if label_choice == -1:
                label = torch.argmax(F.sigmoid(batch_y[:,t,:]),dim=1).cpu().numpy()[0]
                if label==1 or label == 2:
                    label_choice = label

            if pred_choice==-1:
                pred = model.linear_process(t, feats[t])
                pred = torch.argmax(F.sigmoid(pred),dim=1).cpu().numpy()[0]
                if pred==1 or pred==2:
                    pred_choice=pred

            # 最后时刻后模型仍然没有做决策
            if pred_choice==-1 and t==batch_x.shape[1]-1:
                # 决策置为label_choice
                pred_choice = label_choice
                 # 强制做决策
# #                 pred_last = model.module.linear_process(t, feats[t]) #[1, 4]
#                 pred_last = model.linear_process(t, feats[t]) #[1, 4]
#                 if(pred_last[0, 1] > pred_last[0, 2]):
#                     pred_choice = 1
#                 else:
#                     pred_choice = 2
        #
        # 模型做出决策才加进去
        if pred_choice !=-1:
            pred_result.append(pred_choice)
            if_make_decision.append(True)
        else:
            if_make_decision.append(False)
        #
        #     pred_result.append(pred_choice)
        label_result.append(label_choice)
        shapes.append(np.array(shape))

    # 统计每个形状的数量
    shape_num_pred = [] #模型做出预测的时候才加进去
    shape_num_all = []
    for i,shape in enumerate(shapes):
        temp = np.zeros(10)
        for j in range(shape.__len__()):
            temp[shape[j]] += 1
        shape_num_all.append(temp)
        if(if_make_decision[i]==True):
            shape_num_pred.append(temp)

    shape_num_all = np.array(shape_num_all).astype(int)
    shape_num_pred = np.array(shape_num_pred).astype(int)

#     return np.array(pred_result)- 1, shape_num_pred, np.array(label_result)-1, shape_num_all
    # 模型预测情况的回归
    if is_pred==True:
        return np.array(pred_result)- 1, shape_num_pred
    # gt 的回归
    else:
        return np.array(label_result)-1, shape_num_all

def data_trans(pred,shapes):

    shape_num = []
    for i,shape in enumerate(shapes):
        temp = np.zeros(10)
        for j in range(len(shape)):
            temp[shape[j]] += 1
        shape_num.append(temp)
    shape_num = np.array(shape_num).astype(int)

    # for i in range(shape_num.shape[0]):
    #     if i>0:
    #         shape_num[i] += shape_num[i-1]


    pred = np.array(pred).astype(int)
    pred = pred-1
    Q = np.zeros_like(pred).astype(float)
    for i in range(pred.shape[0]):
        # P = (i+1 - np.sum(pred[:i+1])) / (i+1)
        P = 1 if pred[i]==0 else 0
        q = -np.log10(1/(P+1e-8) - 1 + 1e-8)
        Q[i] = q

    # weight = np.linspace(-0.9, 0.9, 10)
    #
    # sum_weight = np.sum(weight*shape_num,axis=1)

    # data = np.concatenate([shape_num,Q.reshape(-1,1)],axis=1)
    #
    # df = pd.DataFrame(data)
    #
    # df.to_excel('result.xlsx')

    return shape_num[10:,:],Q[10:]


if __name__ == "__main__":

    test_dataset = SeqDataSet(num=10000)
    data_y, data_x= get_choice(test_dataset, bs=1, is_pred=True)
#     pred_y, pred_x, label_y, label_x = get_choice(test_dataset, bs=1, is_pred=True)

    df = pd.DataFrame(data_x, columns=[f'shape_{i+1}' for i in range(10)])  # 为每列特征命名
    df['y'] = data_y  # 将data_y作为新的一列添加到DataFrame中
    pred_path = f'data/data_{formatted_time}.xlsx'  # 指定Excel文件的路径和名称
    df.to_excel(pred_path, index=False)  # 写入Excel文件，不包含行索引

#     df_label = pd.DataFrame(label_x, columns=[f'shape_{i+1}' for i in range(10)])  # 为每列特征命名
#     df_label['y'] = label_y  # 将data_y作为新的一列添加到DataFrame中
#     label_path = f'data/data_{formatted_time}_label.xlsx'  # 指定Excel文件的路径和名称
#     df_label.to_excel(label_path, index=False)  # 写入Excel文件，不包含行索引

    data = np.hstack([data_x, data_y.reshape(-1, 1)])  # 合并成 11维度的 [x, y]

    # 创建 DataFrame
    columns = [f"x{i}" for i in range(10)] + ["y"]
    df = pd.DataFrame(data, columns=columns)

    # print("原始数据：")
    # print(df)

    # 将 10维的 x 转换为元组，作为分组键
    df['group_key'] = df.iloc[:, :-1].apply(tuple, axis=1)

    # 按 group_key 分组
    grouped = df.groupby('group_key')

    x = []
    y = []
    # 查看每组数据
    for group, sub_df in grouped:
        # print(f"组: {group}")
        sub_df = sub_df.to_numpy()
        #p = (sub_df.shape[0] - np.sum(sub_df[:,10]))/sub_df.shape[0]
        p = (np.sum(sub_df[:,10]==0))/sub_df.shape[0]
        # if 0.0<p<1.0:

        q = -np.log10(1 / (p+1e-8) - 1 +1e-8 )
        x.append(sub_df[0, :10])
        y.append(q)
    x = np.array(x)
    y = np.array(y)



    regression(x,y)
