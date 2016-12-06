from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import datetime
import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import manifold
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
FLAGS = None

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)
    ########################
    # PUT YOUR CODE HERE  #
    ########################

    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    num_batches = int(FLAGS.max_steps)
#    num_batches = 100

    batch_size = FLAGS.batch_size
    x = tf.placeholder("float", [None,32,32,3])
    y = tf.placeholder("float", [None,10])
    with tf.variable_scope("training") as scope:
        conv = ConvNet()
        _,_,_,logits = conv.inference(x)
        #print(logits.get_shape())
        loss = conv.loss(logits, y)
        accuracy = conv.accuracy(logits,y)
        minimize = train_step(loss)
        merge_summaries = tf.merge_all_summaries()
        saver = tf.train.Saver()
	scope.reuse_variables()


        with tf.Session() as sess:
            st = str(datetime.datetime.now())
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir+ "/" + str(st) + "/train" , sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir+ "/" + str(st) +"/test", sess.graph)
            sess.run(tf.initialize_all_variables())

            x_test, y_test = cifar10.test.images, cifar10.test.labels
            for i in range(num_batches):
                x_in, y_in = cifar10.train.next_batch(FLAGS.batch_size)
                if i % FLAGS.print_freq == 0 or i == num_batches - 1 :
                    [summary,l,a,m] = sess.run([merge_summaries,loss,accuracy,minimize], {x:x_in,y:y_in})
                    train_writer.add_summary(summary,i)
                    print("Progress: ",(float(i)/num_batches) * 100)
		    print("Train Accuracy:", a)
		    print("Train Loss", l)
		    print("-------------")
                else:
                    [l,a,m] = sess.run([loss,accuracy,minimize], {x:x_in,y:y_in})

                if i % FLAGS.eval_freq == 0  or i==num_batches -1:
                    [summary,test_acc,test_loss] = sess.run([merge_summaries,loss,accuracy],{x:x_test, y:y_test})
                    test_writer.add_summary(summary,i)
                    print("===========")
                    print("Test accuracy:", test_acc)
                    print("Test Loss:", test_loss)
                    print("===========")
                if i%FLAGS.checkpoint_freq == 0 or i==num_batches-1:
                    save_path = saver.save(sess, FLAGS.checkpoint_dir + "/model_l2.ckpt")




            print("Model saved in %s" % save_path)
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x = tf.placeholder("float", [None,32,32,3])
    with tf.variable_scope("training") as scope:
        conv = ConvNet()
        #saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir + "/model.ckpt.meta")
        [_,fc1,fc2,_] = conv.inference(x)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_dir + "/model_l2.ckpt")
        x_in, y_in = cifar10.test.images, cifar10.test.labels
	# x_in = x_in[0:100,:]
	# y_in = y_in[0:100,:]
        X = sess.run([fc2],{x:x_in})
        X = np.asarray(X[0])
        print(X.shape)
        tsne = manifold.TSNE()
        plotting = tsne.fit_transform(X)
        print(plotting.shape)
        plt.scatter(plotting[:, 0], plotting[:, 1],c=np.argmax(y_in,1),marker='+')
        plt.savefig("t-sne.png")
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()

    if FLAGS.is_train =='True':
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = 'True',
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
