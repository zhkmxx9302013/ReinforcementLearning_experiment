import tensorflow as tf
import tensorflow.contrib.slim as slim

class Decoder:
    def __init__(self, hidden_size, z_dim, time_steps, lstm_unit_size, action_num, state_num):
        """

        :param hidden_size:
        :param z_dim:
        """
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.timesteps = time_steps
        self.lstm_unit_size = lstm_unit_size
        self.action_num = action_num
        self.state_num = state_num

        self.A_W1 = tf.Variable(self.xavier_init([self.z_dim, self.hidden_size]))
        self.A_b1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))
        self.A_W2 = tf.Variable(self.xavier_init([self.hidden_size, self.timesteps * self.action_num]))
        self.A_b2 = tf.Variable(tf.zeros(shape=[self.timesteps * self.action_num]))

        self.S_W1 = tf.Variable(self.xavier_init([self.z_dim, self.hidden_size]))
        self.S_b1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))
        self.S_W2 = tf.Variable(self.xavier_init([self.hidden_size, self.timesteps * self.state_num]))
        self.S_b2 = tf.Variable(tf.zeros(shape=[self.timesteps * self.state_num]))
        pass

    def create_network(self, z, is_sample):
        self.action_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.action_num], name='decoder_input_action')
        with tf.name_scope('decoder') as scope:
            """
            处理latent space 输入
            """
            if is_sample: # 来自 encoder μ，σ采样
                z_name = '_input_z'
                self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='decoder_input_z')
            else: # 与latent space 相同维度的高斯噪声
                z_name = '_sample_z'
                with tf.variable_scope('decoder_sample_z'):
                    self.z = z
            self.z = tf.reshape(self.z, [-1, self.z_dim*self.timesteps])
            self.z = slim.fully_connected(self.z,
                                          self.z_dim,
                                          activation_fn=tf.nn.leaky_relu,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0))
            """
            构建 action decoder
            """
            with tf.variable_scope('decoder'+z_name+'_action_layer_1'):
                action_layer1 = tf.nn.relu(tf.matmul(self.z, self.A_W1) + self.A_b1)
            with tf.variable_scope('decoder'+z_name+'_action_logits'):
                action_logits = tf.matmul(action_layer1, self.A_W2) + self.A_b2
            with tf.variable_scope('decoder'+z_name+'_prob'):
                action_prob = tf.nn.sigmoid(action_logits)

            """
            构建 state decoder
            """
            with tf.variable_scope('decoder' + z_name + '_state_layer_1'):
                state_layer1 = tf.nn.relu(tf.matmul(self.z, self.S_W1) + self.S_b1)
            with tf.variable_scope('decoder' + z_name + '_state_logits'):
                state_logits = tf.matmul(state_layer1, self.S_W2) + self.S_b2
            with tf.variable_scope('decoder' + z_name + '_state_prob'):
                state_prob = tf.nn.sigmoid(state_logits)

            return action_logits, action_prob, state_logits, state_prob

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def get_model_param_list(self):
        return [variable for variable in tf.trainable_variables('decoder')]

# if __name__ == '__main__':
#     e =Decoder(10,3,2000,10,40,27)
#     e.create_network()
#     # e.sample_z(1,1)
#     print(e.state_in, e.z_var, e.z_mu, e.latent)