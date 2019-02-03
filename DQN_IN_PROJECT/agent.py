import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from util import TARGET_SCOPE, MAIN_SCOPE
from model import Network

# from tensorflow.python import debug as tf_debug

class Agent:
    def __init__(self, session, action_dim, state_dim, batch_size=64,  nsteps=1, trace_length=10,hidden_size=10,
                 initial_epsilon=0.5, final_epsilon=0.01, replay_size=50000, gamma=0.9, replace_target_freq=10,
                 doubleDQN=False, duelingDQN=False,DRQN=False, store_ckpt=True, ckpt_interval=100,
                 ckpt_dir='./model'):
        """
        Init agent.
        :param session: tensorflow session.
        :param action_dim: action dimension.
        :param state_dim: state dimension.
        :param batch_size: batch size.
        :param nsteps: nsteps λ
        :param trace_length: LSTM trace length.
        :param hidden_size: Fully connected layer to LSTM layer size.
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
        self.DRQN = DRQN
        self.GAMMA = gamma
        self.REPLACE_TARGET_FREQ = replace_target_freq
        self.BATCH_SIZE = batch_size
        self.hidden_size = hidden_size
        self.trace_length = trace_length
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.nsteps = nsteps
        self.nstep_buffer = []
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = self.INITIAL_EPSILON
        self.store_ckpt = store_ckpt
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval

        self.main_net = Network(self.state_dim, self.action_dim, self.DUELING_DQN, self.DRQN, hidden_size, MAIN_SCOPE)
        self.target_net = Network(self.state_dim, self.action_dim, self.DUELING_DQN, self.DRQN, hidden_size, TARGET_SCOPE)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))
        # saver after defining variables
        self.saver = tf.train.Saver()
        self.create_training_method()
        self.soft_update()
        self.init_checkpoint()
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        writer = tf.summary.FileWriter("./model/", self.session.graph)
        writer.close()
        pass

    def init_checkpoint(self):
        """
        Init checkpoint directory.
        :return:
        """
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def visulize_network_structure(self, writer):
        summary_writer = tf.summary.FileWriter(writer.log_dir, graph=self.session.graph)
        summary_writer.close()

    def create_training_method(self):
        """
        Define cost and optimizer
        :return:
        """
        self.action_input = tf.placeholder("float", [None, self.action_dim], name="action_input")  # one hot presentation
        self.y_input = tf.placeholder("float", [None], name="y_input")
        Q_action = tf.reduce_sum(tf.multiply(self.main_net.Q_value, self.action_input), reduction_indices=1)
        # self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.loss = tf.losses.mean_squared_error(self.y_input, Q_action)
        self.optimizer = tf.train.RMSPropOptimizer(0.0011).minimize(self.loss)

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

        self.nstep_buffer.append((state, one_hot_action, reward, next_state))

        if (len(self.nstep_buffer) >= self.nsteps):
            R = sum([self.nstep_buffer[i][2] * (self.GAMMA ** i) for i in range(self.nsteps)])
            state, action, _, _ = self.nstep_buffer.pop(0)
            if done:
                self.nstep_buffer.clear()
            # 存入replay_buffer
            # self.replay_buffer = deque()
            self.replay_buffer.append((state, one_hot_action, R, next_state, done))

            # 溢出出队
            if len(self.replay_buffer) > self.REPLAY_SIZE:
                self.replay_buffer.popleft()
        # 可进行训练条件
        if len(self.replay_buffer) > self.BATCH_SIZE:
            self.train_Q_network(episode)

    def get_transition(self):
        """
        Get transition index.
        @mode random
        :return:
        """
        indexes = []
        for _ in range(self.BATCH_SIZE):
            accepted = False
            while not accepted:
                point = np.random.randint(0, len(self.replay_buffer) - self.trace_length)
                accepted = True
                for i in range(self.trace_length-1):
                    if self.replay_buffer[point+i][-1] > 0:
                        accepted = False
                        break
                if accepted:
                    for i in range(self.trace_length):
                        indexes.append(point+i)

        return np.array([self.replay_buffer[idx] for idx in indexes])

    def train_Q_network(self, episode):
        """
        Q网络训练
        :return:
        """
        state_in = (np.zeros([self.BATCH_SIZE, self.hidden_size]), np.zeros([self.BATCH_SIZE, self.hidden_size]))
        self.time_step += 1
        # 从 replay buffer D中随机选取 batch size N条数据<s_j,a_j,r_j,s_j+1,done>$  D_selected
        minibatch = self.get_transition()
        state_batch = minibatch[:,0].tolist()#[data[0] for data in minibatch]
        action_batch = minibatch[:,1].tolist()#[data[1] for data in minibatch]
        reward_batch = minibatch[:,2].tolist()#[data[2] for data in minibatch]
        next_state_batch = minibatch[:,3].tolist()#[data[3] for data in minibatch]

        # 计算目标Q值y
        y_batch = []
        QTarget_value_batch = self.target_net.Q_value.eval(feed_dict={self.target_net.state_input: next_state_batch,
                                                                      self.target_net.batch_size: self.BATCH_SIZE,
                                                                      self.target_net.train_length: self.trace_length
                                                                      })
        Q_value_batch = self.main_net.Q_value.eval(feed_dict={self.main_net.state_input: next_state_batch,
                                                              self.main_net.batch_size: self.BATCH_SIZE,
                                                              self.main_net.train_length: self.trace_length
                                                              })
        for i in range(0, len(minibatch)):
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
            self.main_net.state_input: state_batch,
            self.main_net.state_in: state_in,
            self.main_net.batch_size: self.BATCH_SIZE,
            self.main_net.train_length: self.trace_length
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
            self.main_net.state_input: [state],
            self.main_net.batch_size: 1,
            self.main_net.train_length: 1
        })[0]

        self.state_in = self.session.run(self.main_net.rnn_state, feed_dict={
            self.main_net.state_input: [state],
            self.main_net.state_in: self.state_in,
            self.main_net.batch_size: 1,
            self.main_net.train_length: 1
        })

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

    def reset_cell_state(self):
        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))
