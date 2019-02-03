"""
    @author: zhkmxx9302013
    @date: 2019年2月1日21:39:07
"""
import gym
import agent as algo_agent
import tensorflow as tf
# from environment import Environment
from tensorboardX import SummaryWriter

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 100000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
TRACE_LENTH = 10
HIDDEN_SIZE = 100
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network
STORE_CKPT = True
CKPT_INTERVAL = 1000
CKPT_DIR = './model/DuelingDDQN'
ENV_NAME = 'Acrobot-v1'
EPISODE = 3000  # Episode limitation

####################
####################
# Algo Modules
N_STEPS = 4         # multisteps
DOUBLE_DQN = True   # double dqn
DUELING_DQN = True  # dueling dqn
DRQN = True         # drqn
####################
####################

def main():
    # initialize OpenAI Gym env and dqn agent
    # env = Environment()
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    session = tf.InteractiveSession()

    agent = algo_agent.Agent(session=session,
                             action_dim=action_dim,
                             state_dim=state_dim,
                             batch_size=BATCH_SIZE,
                             nsteps=N_STEPS,
                             trace_length=TRACE_LENTH,
                             hidden_size=HIDDEN_SIZE,
                             initial_epsilon=INITIAL_EPSILON,
                             final_epsilon=FINAL_EPSILON,
                             replay_size=REPLAY_SIZE,
                             gamma=GAMMA,
                             replace_target_freq=REPLACE_TARGET_FREQ,
                             doubleDQN=DOUBLE_DQN,
                             duelingDQN=DUELING_DQN,
                             DRQN=DRQN,
                             store_ckpt=STORE_CKPT,
                             ckpt_interval=CKPT_INTERVAL,
                             ckpt_dir=CKPT_DIR
                             )
    writer = SummaryWriter()
    # print(writer.log_dir)
    # agent.visulize_network_structure(writer)

    score = []
    mean = []
    # summary_writer = tf.summary.FileWriter(writer.log_dir, graph=session.graph)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        total_reward = 0
        step = 0
        # Train
        while True:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)

            agent.perceive(state, action, reward, next_state, done, episode)
            state = next_state
            if done:
                total_reward = total_reward
                total_reward = total_reward + reward
                agent.reset_cell_state()
                break
            total_reward += reward
            step += 1

        print(total_reward)
        score.append(total_reward)

        writer.add_scalar('total_reward', total_reward, episode)
        mean_reward = sum(score[-100:]) / 100
        mean.append(mean_reward)
        writer.add_scalar('mean_reward', mean_reward, episode)
        agent.update_target_q_network(episode)

    # summary_writer.close()
if __name__ == '__main__':
    main()
