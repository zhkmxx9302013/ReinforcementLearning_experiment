from encoder import Encoder
from decoder import Decoder
import tensorflow as tf


class Vae:
    def __init__(self, h_dim, z_dim, time_steps, lstm_unit_size, action_num, state_num):
        """
        init
        :param h_dim: hidden size
        :param z_dim: latent space size
        :param time_steps: timesteps
        :param lstm_unit_size: lstm unit size
        :param action_num: action space size
        :param state_num: state space size
        """
        self.time_steps = time_steps
        self.state_num = state_num
        self.action_num =action_num
        self.encoder = Encoder(h_dim, z_dim, time_steps, lstm_unit_size, action_num, state_num)
        self.encoder.create_network()
        self.decoder = Decoder(h_dim, z_dim, time_steps, lstm_unit_size, action_num, state_num)
        self.train_ops()
        pass

    def train_ops(self):
        z_sample = self.encoder.sample_z(self.encoder.z_mu, self.encoder.z_var)  # (batch, timestep, latent size)

        # 用于VAE训练
        self.action_logits, _, self.state_logits, _ = self.decoder.create_network(z_sample, False)

        # 用于从正态分布噪声 decoder出结果
        _, self.action_prob, _, self.state_prob = self.decoder.create_network(None, True)
        # self.logits = tf.reshape(self.logits, [-1, self.time_steps, self.state_num])
        # x_for_loss = tf.reshape(self.encoder.X, [-1, self.X_dim])

        # 用于计算 state部分的loss
        state_for_loss = tf.reshape(self.encoder.state_in, [-1, self.time_steps*self.state_num])

        # 用于计算 action 部分的loss
        action_for_loss = tf.reshape(self.decoder.action_in, [-1, self.time_steps*self.action_num])

        print(self.encoder.z_var,self.state_logits, state_for_loss)
        print(self.action_logits, action_for_loss)
        # reconstruction loss ::  E[log P(X|z)] 标签为输入样本X，拟合为decoder logits
        # self.recon_loss = None
        # for i in range(self.time_steps):
        self.recon_loss = (tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.state_logits, labels=state_for_loss), 1) +
                          tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.action_logits, labels=action_for_loss), 1))

        # self.recon_loss = tf.reduce_sum(self.recon_loss)
        # KL loss  ::  D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        self.kl_loss = tf.reduce_sum(0.5 * tf.reduce_sum(tf.exp(self.encoder.z_var) + self.encoder.z_mu ** 2 - 1. - self.encoder.z_var, 1),1)
        print(self.recon_loss, self.kl_loss)
        # VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
        self.solver = tf.train.AdamOptimizer().minimize(self.vae_loss)
        pass

    def get_loss(self, state_mb, action_mb):
        """
        获取loss
        :param state_mb: 样本state batch序列
        :param action_mb: action batch序列
        :return:
        """
        _, loss = tf.get_default_session().run([self.solver, self.vae_loss], feed_dict={self.encoder.state_in: state_mb, self.decoder.action_in: action_mb})
        return loss


    def get_sample(self, z, action_mb):
        """
        获取decoder出的samples
        :param z: 输入与latent space 相同维度的高斯噪声
        :param action_mb: action batch 序列
        :return:
        """
        action, state =  tf.get_default_session().run([self.action_prob, self.state_prob], feed_dict={self.decoder.z: z, self.decoder.action_in: action_mb})
        return action, state

# if __name__ == '__main__':
#     e = Vae(10,10,10)
#     print(e.encoder.get_model_param_list())
#     print(e.decoder.get_model_param_list())