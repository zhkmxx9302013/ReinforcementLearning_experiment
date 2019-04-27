import numpy as np
import pickle
import os
import pandas as pd
import mpu.ml


observation_field = ['未开源']


action_field = ['未开源']

def merge_to_one_action(expert_action):
    """
    处理动作空间
    :param expert_action:
    :return:
    """
    aciton_no_range = 0
    expert_action[np.where(expert_action[:, 0] == 1), 1] += aciton_no_range
    return expert_action[:,1]

def parse_file():
    # max traj
    max_traj = 2000
    state_num = 27
    state_list = [] # 按照数据划分，每个文件中的state序列为一个元素
    action_list = [] # 按照数据划分，每个文件中的action序列为一个元素
    rootdir = '../traj/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            print(path)
            df = pd.read_csv(path)
            state = df[observation_field].as_matrix()
            action = merge_to_one_action(df[action_field].as_matrix())
            idx = len(state)

            state_makeup = []
            action_makeup = []
            for index in range(idx, max_traj):
                state_makeup.append([0.0 for i in range(state_num)])
                action_makeup.append(0)

            state = np.concatenate((state, np.array(state_makeup)))
            action = np.concatenate((action, np.array(action_makeup)))
            action = mpu.ml.indices2one_hot(action, nb_classes=40)
            state_list.append(state)
            action_list.append(action)
            # pd_list.append()
            print(i, np.array(state).shape)
    with open('../gen_traj/state.pickle', 'wb') as sp:
        pickle.dump(state_list, sp)
    with open('../gen_traj/action.pickle', 'wb') as ap:
        pickle.dump(action_list, ap)
    # return state_list, action_list

if __name__ == '__main__':
    parse_file()