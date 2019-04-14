from encoder import Encoder
from decoder import Decoder
import tensorflow as tf


class Vae:
    def __init__(self, X_dim, h_dim, z_dim, time_steps, lstm_unit_size, action_num, state_num):
        self.time_steps = time_steps
        self.X_dim = X_dim
        self.state_num = state_num
        self.encoder = Encoder(X_dim, h_dim, z_dim, time_steps, lstm_unit_size, action_num, state_num)
        self.encoder.create_network()
        self.decoder = Decoder(X_dim, h_dim, z_dim)
        self.train_ops()
        pass

    def train_ops(self):
        z_sample = self.encoder.sample_z(self.encoder.z_mu, self.encoder.z_var)
        self.logits, _ = self.decoder.create_network(z_sample, False) #用于VAE训练
        _, self.prob = self.decoder.create_network(None, True) #用于从正态分布噪声 decoder出结果
        # self.logits = tf.reshape(self.logits, [-1, self.time_steps, self.state_num])
        x_for_loss = tf.reshape(self.encoder.X, [-1, self.X_dim])

        # reconstruction loss ::  E[log P(X|z)] 标签为输入样本X，拟合为decoder logits
        self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=x_for_loss), 1)

        # KL loss  ::  D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.encoder.z_var) + self.encoder.z_mu ** 2 - 1. - self.encoder.z_var, 1)

        # VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
        self.solver = tf.train.AdamOptimizer().minimize(self.vae_loss)
        pass

    def get_loss(self, X_mb):
        """
        获取loss
        :param X_mb: 样本batch序列
        :return:
        """
        _, loss = tf.get_default_session().run([self.solver, self.vae_loss], feed_dict={self.encoder.X: X_mb})
        return loss


    def get_sample(self, z):
        """
        获取decoder出的samples
        :param z: 输入与latent space 相同维度的高斯噪声
        :return:
        """
        samples =  tf.get_default_session().run(self.prob, feed_dict={self.decoder.z: z})
        return samples

# if __name__ == '__main__':
#     e = Vae(10,10,10)
#     print(e.encoder.get_model_param_list())
#     print(e.decoder.get_model_param_list())