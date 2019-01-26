import gym
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Environment
from tensorboardX import SummaryWriter

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 50000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network
DOUBLE_DQN = True

class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        """
        Q net 网络定义
        :return:
        """
        # 输入状态 placeholder
        self.state_input = tf.placeholder("float", [None, self.state_dim])

        # Q 网络结构 两层全连接
        with tf.variable_scope('current_net'):
            W1 = self.weight_variable([self.state_dim, 100])
            b1 = self.bias_variable([100])
            W2 = self.weight_variable([100, self.action_dim])
            b2 = self.bias_variable([self.action_dim])
            h_layer = tf.nn.tanh(tf.matmul(self.state_input, W1) + b1)
            # Q Value
            self.Q_value = tf.matmul(h_layer, W2) + b2

        # Target Net 结构与 Q相同，可以用tf的reuse实现
        with tf.variable_scope('target_net'):
            W1t = self.weight_variable([self.state_dim, 100])
            b1t = self.bias_variable([100])
            W2t = self.weight_variable([100, self.action_dim])
            b2t = self.bias_variable([self.action_dim])
            h_layer_t = tf.nn.tanh(tf.matmul(self.state_input, W1t) + b1t)
            # target Q Value
            self.target_Q_value = tf.matmul(h_layer_t, W2t) + b2t

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        # soft update 更新 target net
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def weight_variable(self, shape):
        """
        初始化网络权值(随机, truncated_normal)
        :param shape:
        :return:
        """
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """
        初始化bias(const)
        :param shape:
        :return:
        """
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        """
        Replay buffer
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
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
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        # 可进行训练条件
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        """
        Q网络训练
        :return:
        """
        self.time_step += 1
        # 从 replay buffer D中随机选取 batch size N条数据<s_j,a_j,r_j,s_j+1,done>$  D_selected
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 计算目标Q值y
        y_batch = []
        QTarget_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                #################用target Q(Q)#######################
                if DOUBLE_DQN:
                    selected_q_next = QTarget_value_batch[i][np.argmax(Q_value_batch[i])]
                #################用target Q(target Q)################
                else:
                    selected_q_next = np.max(QTarget_value_batch[i])

                y_batch.append(reward_batch[i] + GAMMA * selected_q_next)

        feed_dict = {
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        }

        self.optimizer.run(feed_dict=feed_dict)

        print("loss: ", self.cost.eval(feed_dict, self.session))


    def egreedy_action(self, state):
        """
        epsilon-greedy策略
        :param state:
        :return:
        """
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def update_target_q_network(self, episode):
        # update target Q netowrk
        if episode % REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)
            # print('episode '+str(episode) +', target Q network params replaced!')




# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1000  # Episode limitation



def main():
    # initialize OpenAI Gym env and dqn agent
    env = Environment()
    env = gym.make(ENV_NAME)
    writer = SummaryWriter()
    agent = DQN(env)
    score = []
    mean = []
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        total_reward = 0

        step = 0
        # Train
        # for step in range(STEP):
        while True:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)

            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                total_reward = total_reward
                total_reward = total_reward + reward
                break
            total_reward += reward
            step += 1


        print(total_reward)
        score.append(total_reward)
        writer.add_scalar('total_reward', total_reward, episode)
        mean_reward = sum(score[-100:]) / 100
        mean.append(mean_reward)
        writer.add_scalar('mean_reward', mean_reward, episode)
        agent.update_target_q_network(episode)


if __name__ == '__main__':
    main()
