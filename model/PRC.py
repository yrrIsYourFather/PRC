import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from tqdm import tqdm

class PRL_Net(nn.Module):

    def __init__(self,
                 n_internal_units=100, 
                 spectral_radius=0.99, 
                 leak=None,
                 connectivity=0.3, 
                 input_scaling=0.2, 
                 noise_level=0.0, 
                 circle=False,
                 n_blocks=6,
                 n_class=4,
                 T=70,
                 lamda=5):

        super(PRL_Net, self).__init__()

        # Initialize hyperparameters
        self.n_internal_units = n_internal_units
        self.input_scaling = input_scaling
        self.noise_level = noise_level
        self.leak = leak
        self.n_blocks = n_blocks
        self.n_class = n_class
        self.T = T
        self.lamda = lamda
        
        # 初始化内部权重 internal_weights
        self.internal_weights = []
        for i in range(self.n_blocks):
            if circle:
                internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units,
                    spectral_radius)
            else:
                internal_weights = self._initialize_internal_weights(
                    n_internal_units,
                    connectivity,
                    spectral_radius)
            self.internal_weights.append(internal_weights)

        self.input_weights = []

        self.linear = nn.ModuleList()
        for t in range(self.T):
            linear = nn.Linear(self.n_blocks * self.n_internal_units, self.n_class)
            self.linear.append(linear)

        # 用于对照试验
        # self.dim = 1024
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.n_blocks * self.n_internal_units, self.dim),
        #     nn.ReLU(),
        #     nn.Linear(self.dim,self.n_class)
        # )


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """Generate internal weights with circular topology.
        """
        
        # Construct reservoir with circular topology
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
            
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        """Generate internal weights with a sparse, uniformly random topology.
        """

        # Generate sparse, uniformly distributed weights.
        # 随机生成n_internal_units×n_internal_units的稀疏矩阵，并转换为常规的密集矩阵
        # 其非零元素的密度由connectivity参数控制
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        # 找到矩阵所有非零值-0.5，使其均匀分布在 [-0.5,0.5] 区间
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights) # 计算矩阵特征值
        e_max = np.max(np.abs(E)) # 特征值绝对值最大的一个
        internal_weights /= np.abs(e_max)/spectral_radius # 调整矩阵谱半径

        return internal_weights


    def rc_step(self, X, n_drop=0, end=None, get_state=False):

        N, T, V = X.shape

        # if T - n_drop > 0:
        #     window_size = T - n_drop
        # else:
        window_size = T

        previous_state_dict = []
        state_matrix_dict = []
        feat_dict = []

        for i in range(self.n_blocks):
            # 根据输入 X 的形状初始化 input_weights
            input_weights = (2.0 * np.random.binomial(1, 0.5, [self.n_internal_units, V]) - 1.0) * self.input_scaling
            self.input_weights.append(input_weights)

            # 初始化存储上一时刻状态量
            previous_state = np.zeros((N, self.n_internal_units), dtype=float)
            previous_state_dict.append(previous_state)

            # 初始化存储各时刻状态量
            state_matrix = np.empty((N, T, self.n_internal_units), dtype=float)
            state_matrix_dict.append(state_matrix)


        # 更新各 block 各时刻的特征值
        for i in range(self.n_blocks):
            for _t in range(T):
                current_input = X[:, _t, :]

                # 计算状态
                state_before_tanh = self.internal_weights[i].dot(previous_state_dict[i].T) + self.input_weights[i].dot(
                    current_input.T)

                # 增加噪声
                state_before_tanh += np.random.rand(self.n_internal_units, N) * self.noise_level

                if self.leak is None:
                    previous_state_dict[i] = np.tanh(state_before_tanh).T
                else:
                    previous_state_dict[i] = (1.0 - self.leak) * previous_state_dict[i] + np.tanh(state_before_tanh).T

                # 保存状态值
                if T - n_drop > 0 and _t > n_drop - 1:
                    state_matrix_dict[i][:, _t - n_drop, :] = previous_state_dict[i]
                elif T - n_drop <= 0:
                    state_matrix_dict[i][:, _t, :] = previous_state_dict[i]

        # plot_rc(state_matrix_dict[0],end)

        # 将各 block 同一时刻 t 的特征值 concat 成一个特征 feat
        for _t in range(T):
            for i in range(self.n_blocks):
                if i == 0:
                    feat = state_matrix_dict[i][:, _t, :]
                else:
                    feat = np.concatenate((feat, state_matrix_dict[i][:, _t, :]), axis=1)
            feat_dict.append(feat)

        if get_state:
            return feat_dict,state_matrix_dict
        else:
            return feat_dict


    def fit(self, X, Y, n_drop=0, n_epochs=400, device="cpu"):

        self.to(device)
        self.train()

        # 判断 train 样本什么时候做出决策
        y_train_argmax = torch.argmax(Y, dim=2)
        y_train_argmax = torch.transpose(y_train_argmax, 0, 1)

        # 计算 train 样本做出决策的时刻
        time_train = []

        N_train = X.shape[0]
        T_train = X.shape[1]

        # 先算 train 数据集上作出决策的时间
        for n in range(N_train):
            flag = False
            for t in range(T_train):
                if(y_train_argmax[t][n] == 1 or y_train_argmax[t][n] == 2):
                    time_train.append(t)
                    flag = True
                    break
            if flag==False:
                time_train.append(T_train-1)

        _feat_dict = self.rc_step(X)
        #print(len(_feat_dict))
        #print(_feat_dict[0].shape)
        feat_dict = []
        for feat in _feat_dict:
            feat = torch.from_numpy(feat).float().to(device)
            feat_dict.append(feat)
        #feat_dict: [T, N, n_block*n_internal_unit]

        #print(feat_dict)

        # 将各时刻的 feat 和对应类别标签 y 用于 linear 层训练
        for t in range(self.T):
            # FIXME: 这里的 loss 函数可以设计得更为复杂
            criterion = nn.CrossEntropyLoss()
            
            # 测试其他的损失函数/优化器
#            criterion = nn.BCEWithLogitsLoss()
#             optimizer = optim.SGD(self.linear[t].parameters(), lr=0.001)
#             scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

            optimizer = optim.AdamW(self.linear[t].parameters(), lr=0.001, weight_decay=0.01)

            X = X.to(device)
            Y = Y.to(device)

            # 使用tqdm创建进度条
            epoch_pbar = tqdm(range(n_epochs), desc=f'Epochs for time step {t}')
            for epoch in epoch_pbar:
                # 正向传播
                classes = self.linear[t](feat_dict[t].float())
                classes = classes.to(device)

                Y_class_indices = torch.argmax(Y, dim=2)  # 将Y转换为类别索引形式
                loss = criterion(classes, Y_class_indices[:, t].long())  # 使用类别索引计算损失

                # 决策时刻单独加上 loss 
                for n in range(N_train):
                    #if(time_train[n]==t):
                    pred_decisions = torch.argmax(classes, dim=1) # 模型预测整个训练集该时刻的决策 [N, 1]
                    pred_decision =  pred_decisions[n] #模型预测该时刻的决策
                    gt_decision = y_train_argmax[time_train[n]][n] # gt该时刻的决策
                    if(time_train[n]<=t and time_train[n]+2>=t):
                        # 该时刻决策差异带来的 loss
                        diff = np.abs(pred_decision.cpu() - gt_decision.cpu())
                        #diff = np.square(pred_decision.cpu() - gt_decision.cpu())
                        loss = loss + self.lamda * diff
                
                # 反向传播和优化
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()

                # 更新进度条的描述
                epoch_pbar.set_description(f'Epoch {epoch+1} for time step {t}, Loss: {loss.item():.4f}')


    def forward(self, X, n_drop=0, device='cpu'):

        _feat_dict = self.rc_step(X, n_drop)
        feat_dict = []
        for feat in _feat_dict:
            feat = torch.from_numpy(feat).float().to(device)
            feat_dict.append(feat)

        # 每一个时刻的 linear 层给出一个输出
        class_dict = []
        for t in range(self.T):
            self.linear[t].eval()

#             class_t = self.linear[t](torch.from_numpy(feat_dict[t]).float())
            class_t = self.linear[t](feat_dict[t])

            # 确定该时刻最终的类别
            class_t = torch.argmax(class_t, dim=1)

            class_dict.append(class_t)

        return class_dict
    

    def evaluate(self, X, y, device='cpu'):

        self.eval()
        
        y_test = torch.argmax(y, dim=2)
        y_test = torch.transpose(y_test, 0, 1).to(device)

        N_test = X.shape[0]
        T_test = X.shape[1]

        _feat_dict = self.rc_step(X)
        feat_dict = []
        for feat in _feat_dict:
            feat = torch.from_numpy(feat).float().to(device)
            feat_dict.append(feat)

        pred = self.forward(X, device=device)

        ##################################################

        # 比较单纯的准确率，只比较各时刻 y 的值是否相同
        cnt = 0
        for t in range(T_test):
            for n in range(N_test):
                if(y_test[t][n] == pred[t][n]):
                    cnt = cnt + 1

        acc = cnt / (N_test * T_test)
        print("各时刻y值准确率: ", acc)

        ##################################################

        # 比较作出决策的时刻是否相同
        time_test = []
        time_pred = []

        # 先算 test 数据集上作出决策的时间
        for n in range(N_test):
            flag = False
            for t in range(T_test):
                if(y_test[t][n] == 1 or y_test[t][n] == 2):
                    time_test.append(t)
                    flag = True
                    break
            if flag==False:
                time_test.append(T_test-1)
                

        # 模型预测的作出决策的时间
        for n in range(N_test):
            flag = False
            for t in range(T_test):
                if(pred[t][n] == 1 or pred[t][n] == 2):
                    time_pred.append(t)
                    flag = True
                    break
            if flag==False:
                time_pred.append(T_test-1)

        #print("各样本，真实vs模型做出决策的时刻如下：")
        cnt_left = 0
        cnt_right = 0
        same_cnt_left = 0
        same_cnt_right = 0
        same_cnt_left_forced = 0 #强制做的左侧决策做对的
        same_cnt_right_forced = 0 #强制做的右侧决策做对的
        no_decision_left = 0
        no_decision_right = 0

        for n in range(N_test):
            #(time_test[n], time_pred[n], y_test[time_test[n]][n], pred[time_pred[n]][n])
            if(y_test[time_test[n]][n]==1):
                cnt_left = cnt_left + 1
            else:
                cnt_right = cnt_right + 1
            # 如果做出的决策相同：
            if(y_test[time_test[n]][n]==pred[time_pred[n]][n]):
                if(pred[time_pred[n]][n]==1):
                    same_cnt_left = same_cnt_left + 1
                else:
                    same_cnt_right = same_cnt_right + 1
            # 如果模型不做决策，需要在最后一步强制决策
            if(time_pred[n]==self.T-1):
                if(y_test[time_test[n]][n]==1):
                    no_decision_left = no_decision_left + 1
                else:
                    no_decision_right = no_decision_right + 1

                # 最后一步强制决策
                pred_ = self.linear_process(self.T-1, feat_dict[self.T-1])# [1, 4]
                pred_choice = 0

                if pred_[0, 1] > pred_[0, 2]:
                    pred_choice = 1
                else:
                    pred_choice = 2

                # 强制做的决策与 gt 决策比较
                if pred_choice==y_test[time_test[n]][n]:
                    if pred_choice==1:
                        same_cnt_left_forced = same_cnt_left_forced + 1
                    else:
                        same_cnt_right_forced = same_cnt_right_forced + 1
                


        same_cnt = same_cnt_left + same_cnt_right
        acc_left = same_cnt_left / cnt_left
        acc_right = same_cnt_right / cnt_right
        acc_left_decided = same_cnt_left / (cnt_left - no_decision_left)
        acc_left_decided_with_forced = (same_cnt_left + same_cnt_left_forced) / cnt_left
        acc_right_decided_with_forced = (same_cnt_right + same_cnt_right_forced) / cnt_right
        acc_right_decided = same_cnt_right / (cnt_right - no_decision_right)
        acc = same_cnt / N_test
        acc_decided = same_cnt / (N_test - no_decision_left - no_decision_right)
        acc_decided_forced = (same_cnt + same_cnt_left_forced + same_cnt_right_forced) / N_test
        print("样本总数：", N_test)
        print("模型未做出决策：", no_decision_left+no_decision_right)
        print(f"左侧样本总数:{cnt_left}，模型左侧未作决策:{no_decision_left}，模型做出左侧正确决策:{same_cnt_left}，正确率:{acc_left}，决策正确率:{acc_left_decided}，强制决策后正确率:{acc_left_decided_with_forced}")
        print(f"右侧样本总数:{cnt_right}，模型右侧未作决策:{no_decision_right}，模型做出右侧正确决策:{same_cnt_right}，正确率:{acc_right}，决策正确率:{acc_right_decided}，强制决策后正确率:{acc_right_decided_with_forced}")
        print(f"总正确决策数:{same_cnt}，正确率:{acc}，决策正确率:{acc_decided}，强制决策后正确率:{acc_decided_forced}")


    def linear_process(self, t, feat):

        # return self.mlp(feat)

        return self.linear[t](feat)


def plot_rc(data,end):
    # RC内部动态的可视化
    # **结果未展示**
    data = data[0][:70, :]
    time_steps = np.arange(70)

    for i in range(100):  # 遍历每一列（100 条曲线）
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, data[:, i], label=f"Curve {i}" if i < 5 else "", alpha=0.7)

        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.title('Plot of 100 Curves over 140 Time Steps')

        plt.legend(loc='upper right', fontsize=8, ncol=2, frameon=False)

        if end is not None:
            plt.axvline(x=end, color='red', linestyle='--', linewidth=2)

            plt.axvline(x=end+2, color='green', linestyle='--', linewidth=2)
        plt.tight_layout()
        plt.show()

