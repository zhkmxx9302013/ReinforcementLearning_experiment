import tensorflow as tf
import tensorflow.contrib.slim as slim

class Encoder:
    def __init__(self, x_dim, hidden_size, z_dim):
        self.X_dim = x_dim
        self.h_dim = hidden_size
        self.z_dim = z_dim
        pass

    def create_network(self):
        with tf.name_scope('encoder') as scope:
            self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name='encoder_input_x')
            with tf.variable_scope('encoder_layer_1'):
                self.layer1 = slim.fully_connected(self.X,
                                          self.h_dim,
                                          activation_fn=tf.nn.relu,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0)
                                          )

            with tf.variable_scope('encoder_latent_mu'):
                self.z_mu  = slim.fully_connected(self.layer1,
                                          self.z_dim,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0)
                                          )
            with tf.variable_scope('encoder_latent_var'):
                self.z_var = slim.fully_connected(self.layer1,
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

if __name__ == '__main__':
    e =Encoder(1,1,1)
    e.create_network()
    e.sample_z(1,1)
    print(e.X, e.z_var, e.z_mu, e.latent)
