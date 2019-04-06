import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01, logger=None, args=None):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        :param logger: hyper-parameter Saver
        :param is_log: wheter save the hyper-parameter
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma
        self.logger = logger
        self.args = args

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # agent通过新策略选择action的概率 probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # agent通过旧策略选择action的概率 probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('PPO_loss'):
            """
                策略目标函数
            """
            #
            # ratios = tf.divide(act_probs, act_probs_old)
            # r_t(θ) = π/πold 为了防止除数为0，这里截取一下值，然后使用e(log减法)来代替直接除法
            ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0)) - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
            # L_CLIP 裁剪优势函数值
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            self.loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            self.loss_clip = tf.reduce_mean(self.loss_clip)

            """
                策略模型的熵
            """
            # 计算新策略πθ的熵 S = -p log(p) 这里裁剪防止p=0
            self.entropy = -tf.reduce_sum(self.Policy.act_probs * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            self.entropy = tf.reduce_mean(self.entropy, axis=0)  # mean of entropy of pi(obs)

            """
                值目标函数
            """
            # L_vf = [(r+γV(π(st+1))) - (V(π(st)))]^2
            v_preds = self.Policy.v_preds
            self.loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            self.loss_vf = tf.reduce_mean(self.loss_vf)

            # construct computation graph for loss
            # L(θ) = E_hat[L_CLIP(θ) - c1 L_VF(θ) + c2 S[πθ](s)]
            # L = 策略目标函数 + 值目标函数 + 策略模型的熵
            self.loss = self.loss_clip - c_1 * self.loss_vf + c_2 * self.entropy
            # minimize -loss == maximize loss
            self.loss = -self.loss

        optimizer = tf.train.RMSPropOptimizer(learning_rate=args.ppo_lr, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(self.loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(self.loss, var_list=pi_trainable)


    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def log_parameter(self, obs, actions, gaes, rewards, v_preds_next):
        lc, ent, lvf, loss = tf.get_default_session().run([self.loss_clip, self.entropy, self.loss_vf, self.loss], feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes})

        log_dict = {
                'ppo_loss_clip': lc,
                'ppo_entropy': ent,
                'ppo_value_difference': lvf,
                'ppo_total = (Lclip+Lvf+S)': loss
            }

        self.logger.log_parameter(log_dict)

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t)
        :param v_preds: v(st)
        :param v_preds_next: v(st+1)
        :return:
        """
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]

        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})
