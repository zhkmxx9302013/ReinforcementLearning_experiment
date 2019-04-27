
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
import numpy as np
import util_vae

import pandas as pd
from vae import Vae
import pickle
import random

def get_dataset():
    rootdir = './gen_traj/'
    with open(rootdir+'state.pickle', 'rb') as sp:
        state_list = pickle.load(sp)
    with open(rootdir+'action.pickle', 'rb') as ap:
        action_list = pickle.load(ap)
    return state_list, action_list

if __name__ == '__main__':
    states, actions = get_dataset()
    idx_list = [i - 1 for i, item in enumerate(states)]
    # for episode in range(1000000):
    rnd_idx = random.sample(idx_list, 32)
    # print(np.array(states).reshape())
    state_mb = np.array(states)[rnd_idx]
    action_mb = np.array(np.array(actions)[rnd_idx])
    print(state_mb.shape)
    # Reshape data to get 28 seq of 28 elements
    batch_state = state_mb.reshape((32, -1, 27))