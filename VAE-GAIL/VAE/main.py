import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import util_vae
from tensorboardX import SummaryWriter
import pandas as pd
from vae import Vae
import pickle
import random



def get_dataset():
    rootdir = './gen_traj/'
    with open(rootdir+'state.pickle', 'rb') as sp:
        state_list = pickle.load(sp)
    with open(rootdir+'action.pickle', 'rb') as ap:
        action_list = pickle.load(ap)
    return state_list, action_list

def train():
    mb_size = 4                        # minibatch size  64条轨迹
    z_dim = 30                         # latent space size
    h_dim = 10                         # hidden layer size
    timesteps = 2000                    # timesteps 每个轨迹采样2000个时间点, 动态lstm，带padding
    state_num = 27                      # state space size
    action_num = 40                     # action space size

    writer = SummaryWriter()
    vae_obj = Vae(h_dim, z_dim,  timesteps, lstm_unit_size=10, action_num=action_num, state_num=state_num)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_vae = tf.train.Saver()

        if not os.path.exists('out/'):
            os.makedirs('out/')


        states, actions = get_dataset()
        idx_list = [i-1 for i ,item in enumerate(states)]
        for episode in range(1000000):

            rnd_idx = random.sample(idx_list, mb_size)
            state_mb = np.array(np.array(states)[rnd_idx])
            action_mb = np.array(np.array(actions)[rnd_idx])
            # Reshape data to get 28 seq of 28 elements
            batch_state= state_mb.reshape((mb_size, timesteps, state_num))
            batch_action = action_mb.reshape((mb_size, timesteps, action_num))
            loss = vae_obj.get_loss(batch_state, batch_action)

            print(episode)
            print('Iter: {}'.format(episode))
            print('Loss: {:.4}'.format(loss))
            print()


            if episode % 1000 == 0:
                saver_vae.save(sess, "./checkpoint_dir/vae_bilstm/vae", global_step=episode, write_meta_graph=True)

                # samples = vae_obj.get_sample(np.random.randn(16, z_dim), action_mb)

            writer.add_scalar('loss', loss, episode)
            # writer.add_graph()
                # fig = plot(samples)
                # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                # i += 1
                # plt.close(fig)

def test():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./checkpoint_dir/vae/vae-10000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir/vae'))

        mb_size = 64
        # mnist = input_data.read_data_sets('../../MNIST_data')
        # (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        # x_test = mnist.test.images
        # y_test_ = mnist.test.labels

        #
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("encoder/encoder_input_x:0")
        latent = graph.get_tensor_by_name("add:0")
        # var = graph.get_tensor_by_name("encoder/encoder_latent_var/fully_connected/BiasAdd:0")
        # l = sess.run(latent, feed_dict={input:x_test})

        # print(l)

        # pic01=plt.figure(figsize=(6, 6))
        # plt.scatter(l[:,0], l[:,1], c=y_test_)
        # plt.plot()
        # plt.colorbar()
        # plt.show()
        # pic01.savefig('temp.png')
        # print(sess.run('encoder_layer_1/fully_connected/weights:0'))


# def plot(samples):
#     fig = plt.figure(figsize=(4, 4))
#     gs = gridspec.GridSpec(4, 4)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#     return fig


if __name__ == '__main__':
    train()
    # test()
    # print(get_dataset())


