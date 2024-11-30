import pandas as pd
import torch
import torch.nn.functional as F
from data.dataset import SeqDataSet
import einops
from tqdm import trange
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from model.PRL import PRL_Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('myNet_1127_145306.pth').to(device)
real_weight = np.linspace(-0.9, 0.9, 10)


def regression(X,y):
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

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X, y)

    beta_0 = ridge.intercept_  # 偏置项
    beta = ridge.coef_  # 各形状的权重

    print(f"最佳 alpha: {best_alpha}")
    print(f"偏置 β0: {beta_0}")
    print(f"权重 β1-10: {beta}")

    return beta


def get_choice(t_dataset,bs=1):
    model.eval()
    x,y,end,t_shapes,labels = t_dataset.x,t_dataset.y,t_dataset.shape_end,t_dataset.shapes,t_dataset.choices

    pred_result = []
    label_result = []
    shapes = []

    # 进行推理
    # 获得形状序列对应的决策结果
    for idx in trange(t_dataset.get_len()):
        batch_x, batch_y, shape,label = x[idx],y[idx],t_shapes[idx],labels[idx]
        batch_x,batch_y = torch.from_numpy(batch_x).unsqueeze(dim=0),torch.from_numpy(batch_y).unsqueeze(dim=0)
        batch_x = einops.rearrange(batch_x, 'b c l -> b l c')
        batch_y = einops.rearrange(batch_y, 'b c l -> b l c')
        feats = model.rc_step(batch_x)
        feats = [torch.from_numpy(feat).float().to(device) for feat in feats]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred_choice = -1
        label_choice = -1
        #
        for t in range(batch_x.shape[1]):
            if pred_choice >0 and label_choice>0:
                break

            if pred_choice==-1:
                pred = model.linear_process(t, feats[t])
                pred = torch.argmax(F.sigmoid(pred),dim=1).cpu().numpy()[0]
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

    pred_result = np.array(pred_result)
    label_result = np.array(label_result)

    return label_result, shape_num


if __name__ == "__main__":

    test_dataset = SeqDataSet(num=2000,mode=0)
    data_y,data_x = get_choice(test_dataset,bs=1)

    data = np.hstack([data_x, data_y.reshape(-1, 1)])  # 合并成 11维度的 [x, y]

    # 建表，分组
    columns = [f"x{i}" for i in range(10)] + ["y"]
    df = pd.DataFrame(data, columns=columns)

    df['group_key'] = df.iloc[:, :-1].apply(tuple, axis=1)

    grouped = df.groupby('group_key')

    x = []
    y = []
    # P -> Q
    # 论文公式(2)
    for group, sub_df in grouped:
        sub_df = sub_df.to_numpy()
        p = (np.sum(sub_df[:,10]==1))/sub_df.shape[0]
        q = -np.log10(1 / (p+1e-8) - 1 +1e-8 )
        x.append(sub_df[0, :10])
        y.append(q)
    x = np.array(x)
    y = np.array(y)

    import numpy as np
    import matplotlib.pyplot as plt

    # 回归得到模型所学习到的主观权重
    infer_weights = regression(x,y)

    x = real_weight
    y = infer_weights.copy()

    plt.scatter(x,y,c='green')
    plt.xticks(ticks=x, labels=[round(float(item),1) for item in x])
    plt.xlabel("Assigned logLR",fontsize=13)
    plt.ylabel("Subjective logLR",fontsize=13)
    plt.show()
    plt.savefig('figC.png',dpi=600)



