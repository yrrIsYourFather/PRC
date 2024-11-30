import torch
import torch.nn.functional as F
from data.dataset import SeqDataSet,generator_t
from torch.utils.data import DataLoader
# from model_revis import MyNet
import einops
from tqdm import trange
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from model.PRL import PRL_Net


# *********************
# 如果要复现 fig D 的结果，需要先在 dataset.py（代码91行）中对形状序列进行指定形状为：shapes = [5,5,5,5,2,2,5,5,5,7,7]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('myNet_1127_145306.pth').to(device)

weight = np.linspace(-0.9, 0.9, 10)

def draw_fig3(t_dataset,bs=1):
    model.eval()

    x,y,end,t_shapes,labels,ends = t_dataset.x,t_dataset.y,t_dataset.shape_end,t_dataset.shapes,t_dataset.choices,t_dataset.shape_end

    for idx in trange(t_dataset.get_len()):
        batch_x, batch_y, shape,label,end = x[idx],y[idx],t_shapes[idx],labels[idx],ends[idx]
        batch_x,batch_y = torch.from_numpy(batch_x).unsqueeze(dim=0),torch.from_numpy(batch_y).unsqueeze(dim=0)
        batch_x = einops.rearrange(batch_x, 'b c l -> b l c')
        batch_y = einops.rearrange(batch_y, 'b c l -> b l c')
        feats = model.rc_step(batch_x)
        feats = [torch.from_numpy(feat).float().to(device) for feat in feats]
        for feat in feats:
            feat.requires_grad = False
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        result = []

        for t in range(batch_x.shape[1]):
            pred = model.linear_process(t, feats[t])
            pred = F.sigmoid(pred)
            pred = pred.cpu().detach().numpy()

            result.append(pred)

        result = np.concatenate(result,axis=0)

        network_output = result.T

        # 画图得到模型输出中，对于左侧选择的反应，和对于右侧的反应（经过了sigmoid处理）

        x = [i for i in range(1, 61)]

        xvline = [22,32,47]
        xtext = [11,27,39,53]
        text = [5,2,5,7]

        plt.plot(x, network_output[1][:60],color = '#28bc4d')
        plt.ylabel('Response for choosing left target',fontsize=13)
        plt.xlabel('Time steps',fontsize=13)
        for xl in xvline:
            plt.axvline(x=xl, color='gray', linestyle='--', linewidth=2)
        for i,xt in enumerate(xtext):
            plt.text(xt, 1.01, str(text[i]), ha='center',fontsize=16,color = '#64a5d9')
        plt.axvline(x=end, color='#FDDB22', linestyle='--', linewidth=2,label='Shape sequence end')
        plt.yticks(ticks=[round(i / 10, 1) for i in range(0, 12, 2)])
        plt.ylim(-0.05,1.1)
        plt.show()

        plt.plot(x, network_output[2][:60],color='#64a5d9')
        plt.ylabel('Response for choosing right target',fontsize=13)
        plt.xlabel('Time steps',fontsize=13)
        for xl in xvline:
            plt.axvline(x=xl, color='gray', linestyle='--', linewidth=2)
        for i,xt in enumerate(xtext):
            plt.text(xt, 1.01, str(text[i]), ha='center',fontsize=16,color = '#28bc4d')
        plt.axvline(x=end, color='#FDDB22', linestyle='--', linewidth=2,label='Shape sequence end')
        plt.yticks(ticks=[round(i/10,1) for i in range(0,12,2)])
        plt.ylim(-0.05,1.1)
        plt.show()


if __name__ == "__main__":

    test_dataset = SeqDataSet(num=1,mode=0)
    draw_fig3(test_dataset,bs=1)


