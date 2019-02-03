import tensorflow as tf
import tensorflow.contrib.slim as slim


class Network:
    def __init__(self, state_dim, action_dim, duelingDQN, DRQN, hidden_size, scope_name):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
        self.train_length = tf.placeholder(dtype=tf.int32, shape=[], name="train_length")
        self.hidden_size = hidden_size
        self.create_Q_network(duelingDQN, DRQN, scope_name)
        pass

    def create_Q_network(self, DUELING_DQN=False, DRQN=False,  scope_name=''):
        """
        Q net 网络定义
        :return:
        """
        # 输入状态 placeholder
        self.state_input = tf.placeholder("float", [None, self.state_dim], name="state_input")

        # Q 网络结构 两层全连接
        with tf.variable_scope(scope_name ):
            with tf.variable_scope(scope_name + '_FC1'):
                layer1 = slim.fully_connected(self.state_input,
                                          self.hidden_size,
                                          activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                          biases_initializer=tf.constant_initializer(0.01)
                                          )
                self.last_layer = layer1

            if DRQN:
                self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size,
                                                     state_is_tuple=True
                                                     )
                self.fc_reshape = tf.reshape(layer1,
                                         [self.batch_size, self.train_length, self.hidden_size]
                                         )
                self.state_in = self.cell.zero_state(self.batch_size,
                                                 tf.float32
                                                 )
                self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.fc_reshape,
                                                         cell=self.cell,
                                                         dtype=tf.float32,
                                                         initial_state=self.state_in,
                                                         scope=scope_name + '_LSTM'
                                                         )
                self.rnn = tf.reshape(self.rnn,
                                  shape=[-1, self.hidden_size]
                                  )



            if DUELING_DQN:
                self.last_layer = slim.fully_connected(self.rnn,
                                                       self.hidden_size,
                                                       activation_fn=None,
                                                       weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                       biases_initializer=tf.constant_initializer(0.01)
                                                       )
                with tf.variable_scope(scope_name + '_value'):
                    self.V = slim.fully_connected(self.last_layer,
                                                  1,
                                                  activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                  biases_initializer=tf.constant_initializer(0.01)
                                                  )

                with tf.variable_scope(scope_name + '_advantage'):
                    self.A = slim.fully_connected(self.last_layer,
                                                  self.action_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                                  biases_initializer=tf.constant_initializer(0.01)
                                                  )

                with tf.variable_scope(scope_name + '_Q'):
                    # Q Value # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
                    self.Q_value = self.V + (
                            self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope(scope_name + '_Q'):
                    self.Q_value = slim.fully_connected(self.rnn,
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
