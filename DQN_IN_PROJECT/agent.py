import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from util import TARGET_SCOPE, MAIN_SCOPE
from model import Network


class Agent:
    def __init__(self, session, action_dim, state_dim, batch_size=64,
                 initial_epsilon=0.5, final_epsilon=0.01, replay_size=50000, gamma=0.9, replace_target_freq=10,
                 doubleDQN=True, duelingDQN=True, store_ckpt=True, ckpt_interval=100,
                 ckpt_dir='./model'):
        """
        Init agent.
        :param session: tensorflow session.
        :param action_dim: action dimension.
        :param state_dim: state dimension.
        :param batch_size: batch size.
        :param initial_epsilon: decay of the epsilon for random selection.
        :param final_epsilon: decay of the epsilon for random selection.
        :param replay_size: replay buffer size.
        :param gamma: decay rate.
        :param replace_target_freq: target net soft update interval.
        :param doubleDQN: use double dqn or not.
        :param duelingDQN: use dueling dqn or not.
        :param store_ckpt: store checkpoint or not.
        :param ckpt_interval: the interval for storing the checkpoint.
        :param ckpt_dir: the directory of the checkpoint files.
        """
        self.session = session
        self.INITIAL_EPSILON = initial_epsilon
        self.FINAL_EPSILON = final_epsilon
        self.REPLAY_SIZE = replay_size
        self.DOUBLE_DQN = doubleDQN
        self.DUELING_DQN = duelingDQN
        self.GAMMA = gamma
        self.REPLACE_TARGET_FREQ = replace_target_freq
        self.BATCH_SIZE = batch_size
        self.action_dim = action_dim
        self.state_dim = state_dim

        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = self.INITIAL_EPSILON
        self.store_ckpt = store_ckpt
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval

        self.main_net = Network(self.state_dim, self.action_dim, self.DUELING_DQN, MAIN_SCOPE)
        self.target_net = Network(self.state_dim, self.action_dim, self.DUELING_DQN, TARGET_SCOPE)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # saver after defining variables
        self.saver = tf.train.Saver()
        self.create_training_method()
        self.soft_update()
        self.init_checkpoint()
        session.run(tf.global_variables_initializer())
        pass



    def init_checkpoint(self):
        """
        Init checkpoint directory.
        :return:
        """
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def create_training_method(self):
        """
        Define cost and optimizer
        :return:
        """
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.main_net.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def soft_update(self):
        """
        soft update
        :return:
        """
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TARGET_SCOPE)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=MAIN_SCOPE)

        # soft update 更新 target net
        with tf.variable_scope('soft_update'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def perceive(self, state, action, reward, next_state, done, episode):
        """
        Replay buffer
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param episode:
        :return:
        """
        # 对action 进行one-hot存储，方便网络进行处理
        # [0,0,0,0,1,0,0,0,0] action=5
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        # 存入replay_buffer
        # self.replay_buffer = deque()
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

        # 溢出出队
        if len(self.replay_buffer) > self.REPLAY_SIZE:
            self.replay_buffer.popleft()

        # 可进行训练条件
        if len(self.replay_buffer) > self.BATCH_SIZE:
            self.train_Q_network(episode)

    def train_Q_network(self, episode):
        """
        Q网络训练
        :return:
        """
        self.time_step += 1
        # 从 replay buffer D中随机选取 batch size N条数据<s_j,a_j,r_j,s_j+1,done>$  D_selected
        minibatch = random.sample(self.replay_buffer, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 计算目标Q值y
        y_batch = []
        QTarget_value_batch = self.target_net.Q_value.eval(feed_dict={self.target_net.state_input: next_state_batch})
        Q_value_batch = self.main_net.Q_value.eval(feed_dict={self.main_net.state_input: next_state_batch})
        for i in range(0, self.BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                #################用target Q(Q)#######################
                if self.DOUBLE_DQN:
                    selected_q_next = QTarget_value_batch[i][np.argmax(Q_value_batch[i])]
                #################用target Q(target Q)################
                else:
                    selected_q_next = np.max(QTarget_value_batch[i])

                y_batch.append(reward_batch[i] + self.GAMMA * selected_q_next)

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.main_net.state_input: state_batch
        })

        if episode % self.ckpt_interval and self.store_ckpt == 0:
            self.global_step.assign(episode).eval()
            self.saver.save(self.session, self.ckpt_dir + "/model.ckpt", global_step=self.global_step)

    def egreedy_action(self, state):
        """
        epsilon-greedy策略
        :param state:
        :return:
        """
        Q_value = self.main_net.Q_value.eval(feed_dict={
            self.main_net.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.main_net.Q_value.eval(feed_dict={
            self.main_net.state_input: [state]
        })[0])

    def update_target_q_network(self, episode):
        # update target Q netowrk
        if episode % self.REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)
            # print('episode '+str(episode) +', target Q network params replaced!')
