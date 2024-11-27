import torch
from torch import nn
import torch.optim as optim
import numpy as np
from scipy import sparse


class MyNet(nn.Module):

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
                 T=70):

        super(MyNet, self).__init__()

        # Initialize hyperparameters
        self.n_internal_units = n_internal_units
        self.input_scaling = input_scaling
        self.noise_level = noise_level
        self.leak = leak
        self.n_blocks = n_blocks
        # output 分类数
        self.n_class = n_class

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
        self.T = T

        self.linear = nn.ModuleList()
        self.dim = 128

        for _ in range(T):
            mlp = nn.Sequential(
                nn.Linear(self.n_blocks * self.n_internal_units, self.dim),
                nn.Tanh(),
                nn.Linear(self.dim,self.n_class)
            )
            for p in mlp.parameters():
                p.requires_grad = True
            self.linear.append(mlp)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """Generate internal weights with circular topology.
        """

        # Construct reservoir with circular topology
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0, -1] = 1.0
        for i in range(n_internal_units - 1):
            internal_weights[i + 1, i] = 1.0

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max) / spectral_radius

        return internal_weights

    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        """Generate internal weights with a sparse, uniformly random topology.
        """

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max) / spectral_radius

        return internal_weights

    def rc_step(self, X, n_drop=0):

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

        # 定义线性层。线性层的个数与时间 T 有关，因此只能在 fit 这步定义。。
        # FIXME: 这里用的最简单的循环定义，后续可以改成 nn.ModuleList 定义

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

        # 将各 block 同一时刻 t 的特征值 concat 成一个特征 feat
        for _t in range(T):
            for i in range(self.n_blocks):
                if i == 0:
                    feat = state_matrix_dict[i][:, _t, :]
                else:
                    feat = np.concatenate((feat, state_matrix_dict[i][:, _t, :]), axis=1)
            feat_dict.append(feat)

        return feat_dict

    def forward(self, X, n_drop=0):

        T = X.shape[-1]
        feat_dict = self.rc_step(X, n_drop)
        pred = [self.linear_process(t,feat_dict[t]) for t in range(T)]

        return pred


    def linear_process(self,t,feat):

        result = self.linear[t](feat)

        return result