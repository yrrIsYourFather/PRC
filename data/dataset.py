import numpy as np

n_input = 20

B = 1.5
exp_decay = 0
lin_decay = 0.1
shape_Dur = 3  # period for shape presentation
shape_inter = 2  # period for shape interval
saccade_dur = 6
min_shapenum = 1  # minimun number of shapes in each trial
max_shapenum = 10  # maximun number of shapes in each trial
n_timepoint = 70  # should be >= 8 + max_shapenum*(shape_inter+shape_Dur)+3;


# block1: left is rewarded
prob_left = np.array([0.0056, 0.0166, 0.0480, 0.0835, 0.1771, 0.2229, 0.1665, 0.1520, 0.0834, 0.0444])
# block2: right is rewarded
prob_right = np.flip(prob_left)
# the assigned weight of each shape
#weight = np.log10(prob_left/prob_right)
weight = np.linspace(-0.9, 0.9, 10)


def bound_update(n_shape):
    # the boundary decaies after each shape appearance
    return (B * np.exp(-exp_decay * (n_shape - 1)) - lin_decay * (n_shape - 1)).round(4)


def decide(shapes):
    """
    Judge whether sampling should be stopped, according to the sum weight and
        boundary in the normal state
    Return 1. whether the decision is going to be made
           2. what the decision is
    """
    n_shape = len(shapes)
    sumweight = weight[shapes].sum().round(4)
    boundary = bound_update(n_shape)
    if np.abs(sumweight) >= boundary or n_shape >= max_shapenum:
        return True, sumweight >= 0
    else:
        return False, []


class generator_t:
    def __init__(self):
        """

        """
        self.setting = {'n_input': n_input, 'boundary': B,
                        'exp_decay': exp_decay, 'lin_decay': lin_decay,
                        'min_rt': min_shapenum, 'max_rt': max_shapenum}

        np.random.seed(123)

    def generate(self, numTrials=int(1e3), seed=123):
        '''
        Generate the shapes and choice according to DDM

        Then, arrange the data into time sequence
        fixation is one time step later than fixation point appear
        fixation stop is one time step later than fixation point disappear
        Time point: 1: fixation point appears; 2: acquire fixation 3 target on;
            4:5:5*shape_num-1:shape on; 7:5:5*shape_num+2 shape off;
        '''
        if seed:
            np.random.seed(int(seed))

        # prepare the trial condition, choice and reward

        # 1: left; 0:right is rewarded
        trialtype = np.random.choice(2, numTrials)
        # 1: choose left; 2:choose right
        choice = np.zeros(numTrials, )
        # the shapes are presented in each trial
        trialCondi = []
        for nTrial in range(numTrials):
            pdf = prob_left if trialtype[nTrial] else prob_right

            shapes = []
            while 1:
                shapes.extend(np.random.choice(10, 1, p=pdf).tolist())
                sampling_stop, decision = decide(shapes)
                if sampling_stop:
                    choice[nTrial] = decision
                    break

            # fig D 此处可以指定形状序列
            # shapes = [5,5,5,5,2,2,5,5,5,7,7]
            trialCondi.append(shapes)

        shape_num = np.array([len(condi) for condi in trialCondi])  # shape number for each trial
        # 1: choose left; 0:choose right
        choice = choice.astype(int)
        # reward is deliveried if choice match the pre-selected target
        reward = choice == trialtype

        shapes_end = (2 + shape_num * (shape_inter + shape_Dur)).astype(int)
        # the time point are included for calculating loss
        training_guide = np.zeros((3, numTrials))
        training_guide[0, :] = reward  # training or not
        training_guide[1, :] = 0  # first trianing time step
        training_guide[2, :] = shapes_end + saccade_dur  # last training time step
        training_guide = training_guide.T.astype(int).tolist()
        # arranage the data
        data_x = []
        data_y = []
        data = []

        for ntrial in range(numTrials):
            # the total length of current trial
            shapes_end_c = int(shapes_end[ntrial])
            inputs = np.zeros((n_input, n_timepoint))  # the NO. of inputs-by-time points

            # viusal stimuli, targets
            inputs[0, :shapes_end_c + 1] = 1  # fixation point appear
            # 原来的             inputs[1, 2:shapes_end_c + 1 + 1] = 1  # left target
            # 原来的             inputs[2, 2:shapes_end_c + 1 + 1] = 1  # right target
            inputs[1, 2:shapes_end_c + 1] = 1  # left target
            inputs[2, 2:shapes_end_c + 1] = 1  # right target

            # viusal stimuli, shapes
            for nth, nshape in enumerate(trialCondi[ntrial]):
                nth_input = 3 + nshape
                # time window
                ind_start = (nth + 1) * (shape_inter + shape_Dur) - 2
                ind_end = ind_start + shape_Dur

                inputs[nth_input, ind_start:ind_end] = 1

            # actions
            choice_ind = (2 - choice[ntrial]) + 14
            # fixate on FP
            inputs[14, 1:shapes_end_c] = 1
            # fixate on choice target
            inputs[choice_ind, shapes_end_c:shapes_end_c + 3] = 1
            # unchosen target disappear
            # 正在看形状，没有选择
            # 原来的             inputs[1 + choice[ntrial], shapes_end_c + 1] = 0

            # reward is deliverd one time step later than fixation point disappear
            inputs[18, shapes_end_c + 2:shapes_end_c + 4] = reward[ntrial]

            inputs[13, :] = 1 - np.any(inputs[0:13, :] != 0, axis=0)  # channel: no visual Input
            inputs[17, :] = 1 - np.any(inputs[14:17, :] != 0, axis=0)  # channel: fixate somewhere else
            inputs[19, :] = 1 - inputs[18, :] != 0  # channel: no reward

            # 上面的数据生成过程**基本上**是
            # Arecurrent neural network framework for flexible and adaptive decision making based on sequence learning》
            # 原论文中所使用的，我们稍作调整和截取

            # 进行截取
            # 输入：14通道
            # 输出：4分类
            # 建模为一个多分类任务
            x = inputs[0:14, :]
            y = inputs[14:18, :]

            data_x.append(x)
            data_y.append(y)
            data.append([inputs.T])

        return data_x, data_y,shapes_end, trialCondi, choice



from torch.utils.data import Dataset


class SeqDataSet(Dataset):
    def __init__(self, num,mode=None):

        Train_data = generator_t()
        x, y, end, trialShape, choices = Train_data.generate(numTrials=num)

        self.x = x
        self.y = y
        # 结束时间
        self.shape_end = end
        # 使用了哪些形状
        self.shapes = trialShape
        self.choices = choices

    def __len__(self):
        return len(self.x)

    def get_len(self):
        return self.__len__()

    def __getitem__(self, _idx):
        return {
            'x': self.x[_idx],
            'y': self.y[_idx],
            'end':self.shape_end[_idx],
            # 'shape':self.shapes[_idx],
            'choice':self.choices[_idx]
        }