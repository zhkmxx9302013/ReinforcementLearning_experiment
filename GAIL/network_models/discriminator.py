import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import pdb


class Discriminator:
    def __init__(self, env, batch_size=32, logger=None, args=None):
        """
        Discriminator 网络(WGAN-GP,替代GAIL中的primitive GAN)
        * 使用Gradient penalty 代替 weight clipping 来满足Lipschitz 限制
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        self.logger = logger
        self.batch_size = batch_size
        self.args = args

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name # D的scope
            """
                [EXPERT] 参数定义
            """
            # expert 状态空间 placeholder
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=(None, env.observation_space))
            # expert 动作空间 placeholder
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            # expert 动作空间 onehot
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space)
            # @todo: 给动作空间onehot加噪，便于训练稳定
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.1, stddev=0.1, dtype=tf.float32) / 1.2 # mean = 0.1
            # concat状态空间与动作空间作为真实样本
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)



            """
                [AGENT] 参数定义
            """
            # agent 状态空间 placeholder
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=(None, env.observation_space))
            # agent 动作空间 placeholder
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            # agent 动作空间 onehot
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space)
            # @todo: 给动作空间onehot加噪，便于训练稳定
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.1, stddev=0.1, dtype=tf.float32) / 1.2
            # concat状态空间与动作空间作为生成样本
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            # [GP] 先随机采一对真假样本，还有一个0-1的随机数:
            epsilon = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            # [GP] 在x_r和x_g的连线上进行随机线性插值采样:
            X_hat_State = self.expert_s + epsilon * (self.agent_s - self.expert_s)
            X_hat_Action = expert_a_one_hot + epsilon * (agent_a_one_hot - expert_a_one_hot)
            X_hat_s_a = tf.concat([X_hat_State, X_hat_Action], axis=1)

            with tf.variable_scope('network') as network_scope:
                crit_e = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                crit_A = self.construct_network(input=agent_s_a)
                network_scope.reuse_variables()
                X_hat_crit = self.construct_network(input=X_hat_s_a)


            with tf.variable_scope('Discriminator_loss'):
                wasserstein = tf.reduce_mean(crit_A) - tf.reduce_mean(crit_e)  # Wasserstein 距离
                grad_D_X_hat = tf.gradients(X_hat_crit, [X_hat_s_a])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=[1]))  # reduction_indices=range(1, X_hat_s_a.shape.ndims)
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)  # λ E[(||D(x_hat)'||_2 - 1)^2]
                loss = wasserstein + self.args.lambda_gp * gradient_penalty
                tf.summary.scalar('discriminator', loss)


            # optimizer = tf.train.AdamOptimizer(learning_rate=self.args.discriminator_lr)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.discriminator_lr)
            self.train_op = optimizer.minimize(loss)
            self.rewards = tf.exp(crit_A)
            self.rewards_e = tf.exp(crit_e)
            self.WGAN = loss



    def construct_network(self, input):
        """
        * 共享权值下构建网络  [variable_scope('network') 该域下共享权值]
        * 用于WGAN，去掉最后的sigmoid层，做回归拟合，近似拟合Wasserstein距离
        * 视情况弱化Discriminator网络
        :param input:
        :return:
        """
        layer_1 = tf.layers.dense(inputs=input, units=100, activation=tf.nn.tanh, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=100, activation=tf.nn.tanh, name='layer2')
        # layer_3 = tf.layers.dense(inputs=layer_2, units=50, activation=tf.nn.tanh, name='layer3')
        prob = tf.layers.dense(inputs=layer_2, units=1, activation=None, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_rewards_e(self, expert_s, expert_a):
        return tf.get_default_session().run(self.rewards_e, feed_dict={self.expert_s: expert_s,
                                                                       self.expert_a: expert_a})

    def get_wgan(self, expert_s, expert_a, agent_s, agent_a):
        """
        获取WGAN-GP Loss
        :param expert_s:
        :param expert_a:
        :param agent_s:
        :param agent_a:
        :return:
        """
        return tf.get_default_session().run(self.WGAN, feed_dict={self.expert_s: expert_s,
                                                                  self.expert_a: expert_a,
                                                                  self.agent_s: agent_s,
                                                                  self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def log_parameter(self, expert_s, expert_a, agent_s, agent_a):
        wgan, re, r = tf.get_default_session().run([self.WGAN, self.rewards_e, self.rewards], feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})
        log_dict = {
            'WGAN_Loss': wgan,
            'Expert_Reward': np.mean(re, axis=0),
            'Agent_Reward': np.mean(r, axis=0)
        }

        self.logger.log_parameter(log_dict)