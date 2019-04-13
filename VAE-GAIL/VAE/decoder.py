import tensorflow as tf
import tensorflow.contrib.slim as slim

class Decoder:
    def __init__(self, x_dim, hidden_size, z_dim):
        self.X_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        pass

    def create_network(self, z, is_sample):
        with tf.name_scope('decoder') as scope:
            z_name = ''
            if is_sample:
                z_name = '_imput_z'
                self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='decoder_imput_z')
            else:
                z_name = '_sample_z'
                with tf.variable_scope('decoder_sample_z'):
                    self.z = z


            P_W1 = tf.Variable(self.xavier_init([self.z_dim, self.hidden_size]))
            P_b1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))

            P_W2 = tf.Variable(self.xavier_init([self.hidden_size, self.X_dim]))
            P_b2 = tf.Variable(tf.zeros(shape=[self.X_dim]))

            with tf.variable_scope('decoder'+z_name+'_layer_1'):
                h = tf.nn.relu(tf.matmul(self.z, P_W1) + P_b1)
            with tf.variable_scope('decoder'+z_name+'_logits'):
                logits = tf.matmul(h, P_W2) + P_b2
            with tf.variable_scope('decoder'+z_name+'_prob'):
                prob = tf.nn.sigmoid(logits)

            return logits, prob

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def get_model_param_list(self):
        return [variable for variable in tf.trainable_variables('decoder')]