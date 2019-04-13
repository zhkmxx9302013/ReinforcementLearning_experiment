#!/usr/bin/python3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
from ResultLogger import ResultLogger
import gym
from environment import Environment
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
import pandas as pd
import utils
from tensorboardX import SummaryWriter


def argparser():
    """
    参数解析器
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail/Oldreward2Net')
    parser.add_argument('--savedir', help='save directory', default='trained_models/Oldreward2Net')
    parser.add_argument('--expertdir', help='expert trajectory directory', default='trajectory/expert_traj.csv')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--lambda_gp', help='gradient penalty lambda', default=6)
    parser.add_argument('--discriminator_lr', help='discriminator learning rate', default=1e-4)
    parser.add_argument('--ppo_lr', help='ppo learning rate', default=1e-4)
    parser.add_argument('--batchsize', default=64)
    parser.add_argument('--episode', default=int(10e4))
    return parser.parse_args()

def train_discriminator(expert_observations, expert_actions, observations, actions, D, batch_size, episode, logger):
    """
    训练discriminator
    :param expert_observations:
    :param expert_actions:
    :param observations:
    :param actions:
    :param D:
    :param batch_size:
    :param episode:
    :return:
    """
    inp_d = [expert_observations, expert_actions, observations, actions]
    for i in range(2):
        sample_indices_d = np.random.randint(low=0, high=observations.shape[0], size=batch_size)  # indices are in [low, high)(当observation数量小于batchsize的时候，采样会有重复)
        sampled_inp_d = [np.take(a=a, indices=sample_indices_d, axis=0) for a in inp_d]  # sample training data(有重复)
        D.train(expert_s=sampled_inp_d[0], expert_a=sampled_inp_d[1], agent_s=sampled_inp_d[2], agent_a=sampled_inp_d[3])
        wgan_loss = D.get_wgan(expert_s=sampled_inp_d[0], expert_a=sampled_inp_d[1], agent_s=sampled_inp_d[2], agent_a=sampled_inp_d[3])

        r_agent = D.get_rewards(agent_s=sampled_inp_d[2], agent_a=sampled_inp_d[3])
        r_expert = D.get_rewards_e(expert_s=sampled_inp_d[0], expert_a=sampled_inp_d[1])

    # output of this discriminator is reward
    if episode % 100 == 0:
        log_dict = {
            'Expert Reward': np.average(np.reshape(r_expert, newshape=(-1,))),
            'WGAN_gp_LOSS': wgan_loss,
            'Agent Reward':np.average(np.reshape(r_agent, newshape=(-1,)))
        }
        D.log_parameter(sampled_inp_d[0], sampled_inp_d[1], sampled_inp_d[2], sampled_inp_d[3])
        logger.log_info(log_dict, episode)
    d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
    d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
    return d_rewards


def train_PPO(PPO, observations, actions, d_rewards, v_preds, v_preds_next, batch_size, episode, logger):
    """
    训练PPO, 使用discriminator reward
    :param PPO:
    :param observations:
    :param actions:
    :param d_rewards:
    :param v_preds:
    :param v_preds_next:
    :param batch_size:
    :param episode:
    :return:
    """
    # GAE
    gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
    gaes = np.array(gaes).astype(dtype=np.float32)
    logger.log_gaes(gaes.mean(), episode)
    # gaes = (gaes - gaes.mean()) / gaes.std()
    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

    # PPO train policy
    inp = [observations, actions, gaes, d_rewards, v_preds_next]

    PPO.assign_policy_parameters()
    for epoch in range(5):
        sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                           size=batch_size)  # indices are in [low, high)
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data

        PPO.train(obs=sampled_inp[0], actions=sampled_inp[1], gaes=sampled_inp[2], rewards=sampled_inp[3], v_preds_next=sampled_inp[4])

    if episode % 100 == 0:
        PPO.log_parameter(sampled_inp[0], sampled_inp[1], sampled_inp[2], sampled_inp[3], sampled_inp[4])


def main(args):
    env = Environment()
    batch_size = args.batchsize
    writer = SummaryWriter(args.logdir)
    logger = ResultLogger(writer)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, logger=logger, args=args)
    D = Discriminator(env, batch_size, logger=logger, args=args)

    expert_ds = pd.read_csv(args.expertdir)
    expert_observations = expert_ds[utils.observation_field].as_matrix() # 筛选obs特征
    expert_actions = utils.merge_to_one_action(expert_ds[utils.action_field].as_matrix()) # 映射action空间，与具体环境相关，这里省略

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        obs = env.reset()
        reward = 0  # do NOT use rewards to update policy

        for episode in range(args.episode):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0

            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                next_obs, reward,sparse_rew , done, info = env.step(act)
                reward = utils.get_curriculum_reward(reward, sparse_rew, 1.0, run_policy_steps)

                if done:
                    total_reward = sum(rewards)
                    total_reward /= run_policy_steps
                    total_reward += reward

                    print("[episode]: ", episode)
                    print('[Policy Reward]: ', total_reward)

                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                  
                    reward = 0
                    break
                else:
                    obs = next_obs

                if episode % 100 == 0:
                    winnum = 0
                    drawnum = 0

            logger.log_result(total_reward, winnum, drawnum, episode)
            if episode % 1000 == 0:
                saver.save(sess, args.savedir + '/model.ckpt')

            observations = np.reshape(observations, newshape=(-1, ob_space))
            actions = np.array(actions).astype(dtype=np.int32)


            # 训练 Discriminator
            d_rewards = train_discriminator(expert_observations, expert_actions, observations, actions, D, batch_size, episode, logger)
            # 训练 PPO
            train_PPO(PPO, observations, actions, d_rewards, v_preds, v_preds_next, batch_size, episode, logger)



if __name__ == '__main__':
    args = argparser()
    main(args)
