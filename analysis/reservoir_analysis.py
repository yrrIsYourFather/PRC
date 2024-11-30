import torch
from data.dataset import SeqDataSet
import einops
from tqdm import tqdm, trange
import statsmodels.api as sm
import numpy as np
from model.PRL import PRL_Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('myNet_1127_145306.pth').to(device)
real_weight = np.linspace(-0.9, 0.9, 10).round(1)

def regression(shape,state,end):
    l=end+3
    x1 = np.zeros(l)
    sum_logLR = 0.0
    x2 = np.zeros(l)
    shape_num = 0
    x3 = np.zeros(l)

    sig = [0, 0, 0,0]
    sig_unit = np.zeros(100)

    for i in range(100):

        y_ = state[i][:l]
        t_ = 0
        shape_idx = 0
        while t_ < l:
            if t_ < 3:
                t_ += 1
                continue
            if t_ > end:
                x1[t_:end + 3] = x1[t_ - 1]
                x2[t_:end + 3] = x2[t_ - 1]
                x3[t_:end + 3] = x3[t_ - 1]
                break

            sum_logLR += round(real_weight[shape[shape_idx]], 1)
            shape_idx += 1
            shape_num += 1
            x1[t_:t_ + 5] = sum_logLR
            x2[t_:t_ + 5] = abs(sum_logLR)
            x3[t_:t_ + 5] = shape_num
            t_ += 5

        X = np.vstack([x1, x2, x3]).T

        X_with_intercept = sm.add_constant(X)  # 自动在 X 的左边添加一列全为 1 的列

        # 构建回归模型
        regress_model = sm.OLS(y_, X_with_intercept)

        # 拟合模型
        results = regress_model.fit()

        # 提取 t 值和 p 值
        t_values = results.tvalues[1:]
        p_values = results.pvalues[1:]

        for j in range(3):
            if p_values[j] < 0.05:
                sig[j] += 1
        if np.sum(p_values<0.05)==3:
            sig[-1]+=1
            sig_unit[i]=1


    return sig,sig_unit

def cell_effect(t_dataset,unit_type=0,bs=1):

    x,y,end,t_shapes,labels,ends = t_dataset.x,t_dataset.y,t_dataset.shape_end,t_dataset.shapes,t_dataset.choices,t_dataset.shape_end

    len = t_dataset.get_len()

    sig1_rate = np.zeros(len)
    sig2_rate = np.zeros(len)
    sig3_rate = np.zeros(len)
    sigall_rate = np.zeros(len)

    sig_unit_s = np.zeros(100)

    for idx in trange(len):
        batch_x, batch_y, shape,label,end = x[idx],y[idx],t_shapes[idx],labels[idx],ends[idx]
        batch_x,batch_y = torch.from_numpy(batch_x).unsqueeze(dim=0),torch.from_numpy(batch_y).unsqueeze(dim=0)
        batch_x = einops.rearrange(batch_x, 'b c l -> b l c')
        batch_y = einops.rearrange(batch_y, 'b c l -> b l c')
        feats,state_matrix_dict = model.rc_step(batch_x,get_state=True)

        # 获取内部 reservoir 的隐藏状态
        state0 = state_matrix_dict[unit_type][0].T
        # 进行回归和t-检验
        # 论文公式（3）
        sig, sig_unit= regression(shape,state0,end)

        # 统计回归的三个系数，分别显著和都显著的比例
        # 100为一个 reservoir 内部cell的数量
        sig1_rate[idx], sig2_rate[idx], sig3_rate[idx],sigall_rate[idx] = sig[0] / 100, sig[1] / 100, sig[2] / 100,sig[-1]/100

        sig_unit_s+=sig_unit

    sigall_rate = np.average(sigall_rate)
    return np.average(sig1_rate),np.average(sig2_rate),np.average(sig3_rate),sigall_rate


if __name__ == "__main__":

    test_dataset = SeqDataSet(num=100,mode=0)
    # 6个并行reservoir
    avg_sigall=[]
    for u in range(6):
        sig1,sig2,sig3,sigall = cell_effect(test_dataset,unit_type=u)
        print(sigall)
        avg_sigall.append(sigall)
    print('平均显著比例:')
    print(np.average(avg_sigall))





