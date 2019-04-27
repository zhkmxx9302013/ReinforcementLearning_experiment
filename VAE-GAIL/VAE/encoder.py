import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn

class Encoder:
    def __init__(self, hidden_size, z_dim, time_steps, lstm_unit_size, action_num, state_num):

        self.h_dim = hidden_size
        self.z_dim = z_dim
        self.state_num = state_num

        self.time_steps = time_steps
        self.lstm_unit_size = lstm_unit_size
        self.action_num = action_num
        pass

    def create_network(self):
        with tf.name_scope('encoder') as scope:
            self.state_in = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.state_num], name='encoder_input_x')
            # Unstack to get a list of 'time_steps' tensors of shape (batch_size, num_input)
            # unstack_state = tf.unstack(self.state_in, self.time_steps, 1)

            # 前向 cell
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_unit_size, forget_bias=1.0)
            # 反向 cell
            lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_unit_size, forget_bias=1.0)

            with tf.variable_scope('encoder_bi_lstm'):
                # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, unstack_state, dtype=tf.float32)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.state_in, dtype=tf.float32)  # [batch_szie, max_time, depth]




            with tf.variable_scope('encoder_lstm_output_avg'):
                self.layer_avg = tf.reduce_mean([outputs[0]], 0)

            with tf.variable_scope('encoder_lstm_linear'):
                self.layer_after_avg = slim.fully_connected(self.layer_avg,
                                          self.h_dim,
                                          activation_fn=tf.nn.leaky_relu,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0)
                                          )
            with tf.variable_scope('encoder_latent_mu'):
                self.z_mu  = slim.fully_connected(self.layer_after_avg,
                                          self.z_dim,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0)
                                          )
            with tf.variable_scope('encoder_latent_var'):
                self.z_var = slim.fully_connected(self.layer_after_avg,
                                          self.z_dim,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0)
                                          )



    def sample_z(self, mu, var):
        eps = tf.random_normal(shape=tf.shape(mu))
        self.latent = mu + tf.exp(var / 2) * eps
        return self.latent

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def get_model_param_list(self):
        return [variable for variable in tf.trainable_variables('encoder')]

# if __name__ == '__main__':
#     e =Encoder(1,1,1)
#     e.create_network()
#     e.sample_z(1,1)
#     print(e.state_in, e.z_var, e.z_mu, e.latent)

