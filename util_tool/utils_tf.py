from distutils.version import LooseVersion
import math
import numpy as np
import random
import os
from six.moves import xrange
import tensorflow as tf
import time
import warnings
from itertools import combinations
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def get_adv(sess, x, y, adv, X_test=None, Y_test=None,
            feed=None, batch_size=128):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """

    assert batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    adv_x = np.ndarray(X_test.shape, dtype=X_test.dtype)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            # feed_dict = {x: X_cur, y: Y_cur}
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            adv_x[start:end] = sess.run(adv, feed_dict=feed_dict)[:cur_batch_size]
        assert end >= len(X_test)

    return adv_x


def model_eval(sess, x, y, predictions, X_test=None, Y_test=None,
               feed=None, batch_size=128):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """

    assert batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                 tf.argmax(predictions,
                                           axis=tf.rank(predictions) - 1))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

            accuracy += cur_corr_preds[:cur_batch_size].sum()

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy


def model_predicton(sess, x, y, predictions, X_test=None, Y_test=None,
                    feed=None, batch_size=128):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """

    assert batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Init result var
    accuracy = 0.0
    predict = np.ndarray([Y_test.shape[0], Y_test.shape[1]])

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_preds = predictions.eval(feed_dict=feed_dict)
            predict[start:end] = cur_preds[:cur_batch_size]
        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return predict


def get_distances(sess, x, predictions1, predictions2, X_test=None,
                  feed=None, batch_size=128):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """

    assert batch_size, "Batch size was not given in args dict"
    if X_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    if (FLAGS.similarity_type == 'l1'):
        correct_preds1 = tf.reduce_sum(tf.abs(predictions1 - predictions2), axis=-1)
    elif (FLAGS.similarity_type == 'cos'):
        predictions2 = tf.concat([predictions2[:, 10:], predictions2[:, :10]], axis=-1)
        correct_preds1 = 1. - tf.div(tf.reduce_sum(predictions1 * predictions2, axis=-1),
                                     tf.sqrt(tf.reduce_sum(predictions1 * predictions1, axis=-1)) * tf.sqrt(
                                         tf.reduce_sum(predictions2 * predictions2, axis=-1)) + 1e-8)

    elif (FLAGS.similarity_type == 'l2'):
        correct_preds1 = tf.linalg.norm(predictions1 - predictions2, axis=-1)
    else:
        print("unknown similarity!")

    # correct_preds1 = tf.reduce_max(tf.abs(predictions1 - predictions2), reduction_indices=[-1])
    # Init result var
    accuracy = 0.0
    L1_dis = np.ndarray(X_test.shape[0], dtype=np.float32)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            feed_dict = {x: X_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_corr_preds = correct_preds1.eval(feed_dict=feed_dict)
            L1_dis[start:end] = cur_corr_preds[:cur_batch_size]
        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return L1_dis
