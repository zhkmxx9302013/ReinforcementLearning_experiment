#!/usr/bin/python3
import argparse
import gym
import numpy as np
import utils
import tensorflow as tf
from environment import Environment
from network.policy_net import Policy_net
from algo.ppo import PPOTrain
from tensorboardX import SummaryWriter
from ResultLogger import ResultLogger


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo_curriculum_without_misslerew')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo_curriculum_without_misslerew')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--ppo_lr', help='ppo learning rate', default=1e-4)
    parser.add_argument('--episode', default=int(10e4), type=int)
    parser.add_argument('--continue_train',default=False, type=bool, help='whether continue training on the previous model.')
    parser.add_argument('--continue_meta', type=str, default='./trained_models/ppo_curriculum/model.ckpt.meta',
                        help='meta file trained by the previous model.')
    parser.add_argument('--continue_modeldir',  type=str, default='./trained_models/ppo_curriculum/',
                        help='trained models dirctory trained by the previous model.')
    return parser.parse_args()


def main(args):
    writer = SummaryWriter(args.logdir)
    logger = ResultLogger(writer)

    env = Environment()  # 自定义环境
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, args=args, logger=logger)
    saver = tf.train.Saver()

    if args.continue_train:
        tf.reset_default_graph()
        tf.train.import_meta_graph(args.continue_meta)



    with tf.Session() as sess:
        if args.continue_train:
            saver.restore(sess, args.continue_modeldir)
        sess.run(tf.global_variables_initializer())
        reward = 0
        winnum = 0
        drawnum = 0
        for episode in range(args.episode):

            observations = []
            actions = []
            v_preds = []
            rewards = []

            run_policy_steps = 0

            total_reward = 0
            obs = env.reset()
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1

                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, sparse_rew, done, info = env.step(act)   
                if reward < -1000:
                    reward = -10

                reward = utils.get_curriculum_reward(reward, sparse_rew, 1.0, run_policy_steps)
                # if episode==1:
                #     print(reward)


                obs = next_obs
                if done:
                    total_reward = sum(rewards)
                    total_reward /= run_policy_steps
                    total_reward += reward
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value

                    reward = -1
                    if info == 3:
                        winnum += 1
                    if info == 2:
                        drawnum += 1

                    break

            if episode % 100 == 0:
                winnum = 0
                drawnum = 0

            logger.log_result(total_reward, winnum, drawnum, episode)
            print(episode, total_reward)
            if episode % 1000 == 0:
                saver.save(sess, args.savedir + '/model.ckpt')

            ####
            ##  GAE
            ####
            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # 把list 转成 适应于tf.placeholder 的numpy array
            observations = np.reshape(observations,  newshape=(-1, ob_space))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            logger.log_gaes(gaes.mean(), episode)
            PPO.log_parameter(observations, actions, gaes, rewards, v_preds_next)
            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(2):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])



if __name__ == '__main__':
    args = argparser()
    main(args)
