import numpy as np
import math
### 省略了博弈部分

def get_curriculum_reward(exp_reward, sparse_reward, alpha, step):
    """
    EXPLORATION CURRICULUM 参考论文
    [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748)
    :param exp_reward: 稠密回报
    :param sparse_reward: 稀疏回报
    :param alpha: 衰减系数
    :param step: step in one episode
    :return:
    """
    if step >= 30:
        alpha = 0.0
    else:
        alpha = math.exp(-step*1.0)

    return alpha*exp_reward + (1.0-alpha)*sparse_reward