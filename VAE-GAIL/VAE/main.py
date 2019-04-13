import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from vae import Vae
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train():
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    mb_size = 64
    z_dim = 2
    X_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    h_dim = 128
    c = 0
    lr = 1e-3


    vae_obj =  Vae(X_dim, h_dim, z_dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_vae = tf.train.Saver()
        # saver_encoder = tf.train.Saver(vae_obj.encoder.get_model_param_list())
        # saver_decoder = tf.train.Saver(vae_obj.decoder.get_model_param_list())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0
        for it in range(1000000):
            X_mb, _ = mnist.train.next_batch(mb_size)
            loss = vae_obj.get_loss(X_mb)
            if it % 1000 == 0:
                if it % 5000 == 0:
                    saver_vae.save(sess, "./checkpoint_dir/vae/vae", global_step=it, write_meta_graph=True)
                print('Iter: {}'.format(it))
                print('Loss: {:.4}'.format(loss))
                print()
                samples = vae_obj.get_sample(np.random.randn(16, z_dim))
                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

def test():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./checkpoint_dir/vae/vae-10000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir/vae'))

        mb_size = 64
        mnist = input_data.read_data_sets('../../MNIST_data')
        # (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
        x_test = mnist.test.images
        y_test_ = mnist.test.labels

        #
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("encoder/encoder_input_x:0")
        latent = graph.get_tensor_by_name("add:0")
        # var = graph.get_tensor_by_name("encoder/encoder_latent_var/fully_connected/BiasAdd:0")
        l = sess.run(latent, feed_dict={input:x_test})

        print(l)

        pic01=plt.figure(figsize=(6, 6))
        plt.scatter(l[:,0], l[:,1], c=y_test_)
        plt.plot()
        plt.colorbar()
        plt.show()
        pic01.savefig('temp.png')
        # print(sess.run('encoder_layer_1/fully_connected/weights:0'))


if __name__ == '__main__':
    # train()
    test()