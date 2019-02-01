import tensorflow as tf
from util import TARGET_SCOPE, MAIN_SCOPE
import numpy as np
import random
from collections import deque
import tensorflow.contrib.slim as slim

class Network:
    def __init__(self, state_dim, action_dim, duelingDQN, scope_name):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.create_Q_network(duelingDQN, scope_name)
        pass

    def create_Q_network(self, DUELING_DQN=True, scope_name=''):
        """
        Q net 网络定义
        :return:
        """
        # 输入状态 placeholder
        self.state_input = tf.placeholder("float", [None, self.state_dim])

        # Q 网络结构 两层全连接
        with tf.variable_scope(scope_name):
            layer1 = slim.fully_connected(self.state_input,
                                          100,
                                          activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                          biases_initializer=tf.constant_initializer(0.01)
                                          )
            if DUELING_DQN:
                with tf.variable_scope(scope_name + '_value'):
                    self.V = slim.fully_connected(layer1,
                                                  1,
                                                  activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                  biases_initializer=tf.constant_initializer(0.01)
                                                  )

                with tf.variable_scope(scope_name + '_advantage'):
                    self.A = slim.fully_connected(layer1,
                                                  self.action_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                  biases_initializer=tf.constant_initializer(0.01)
                                                  )

                with tf.variable_scope(scope_name + '_Q'):
                    # Q Value # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
                    self.Q_value = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope(scope_name + '_Q'):
                    self.Q_value = slim.fully_connected(layer1,
                                                        self.action_dim,
                                                        activation_fn=None,
                                                        weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                        biases_initializer=tf.constant_initializer(0.01)
                                                        )





    def __weight_variable(self, shape):
        """
        初始化网络权值(随机, truncated_normal)
        :param shape:
        :return:
        """
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        """
        初始化bias(const)
        :param shape:
        :return:
        """
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
