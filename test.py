"""
adversiral detection
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.keras import backend as K

sys.path.append("attack/cleverhans")
sys.path.append("external/EvadeML")
sys.path.append("external/LID")
sys.path.append("external/EvadeML/externals/MagNet")

from cleverhans.utils_keras import KerasModelWrapper

# Create TF session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
from dataset.datasets import data_fmnist, data_cifar10, data_mnist, data_svhn, data_gtsrb
from util_tool.utils import get_model, get_detection, get_first_n_examples_id_each_class, \
    get_correct_prediction_idx, \
    get_match_pred_vec, calculate_accuracy, get_umatrix
from util_tool.utils_tf import model_eval, model_predicton
from attack.attacks import get_adv_examples

FLAGS = flags.FLAGS


def detection(nb_epochs=60, batch_size=128,
              data_name='mnist', attack_type='fgsm'):
    """
    detect adversarial examples
    :param nb_epochs: number of iterations of model
    :param batch_size: batch size during training
    :param data_name:  dataset type used(mnist,fmnist,svhn,cifar10)
    :param attack_type: attack for generating adversarial examples
    :return
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    import hashlib
    attack_string_hash = hashlib.sha1(
        (FLAGS.attack_type + FLAGS.dataset + FLAGS.detection_type + FLAGS.label_type).encode('utf-8')).hexdigest()[:5]

    # get data
    dataset_dir = {"mnist": data_mnist, "cifar10": data_cifar10, "fmnist": data_fmnist, 'svhn': data_svhn,
                   "gtsrb": data_gtsrb}
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_dir[data_name]()

    if (FLAGS.detection_type == 'negative'):
        x_train = np.append(x_train, 1. - x_train, axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    if (FLAGS.label_type != 'type1' and FLAGS.detection_type == 'negative'):
        if (FLAGS.label_type == 'type2'):
            P = 1.
        elif (FLAGS.label_type == 'type3'):
            P = 0.75
        y_train_temp1 = np.copy(y_train[:y_train.shape[0] // 2])
        y_train_temp1[y_train_temp1 == 1] = P
        y_train_temp2 = np.copy(y_train[:y_train.shape[0] // 2])
        y_train_temp2[y_train_temp2 == 1] = 1. - P
        y_train = np.concatenate((np.concatenate((y_train_temp1, y_train_temp2), axis=-1),
                                  np.concatenate((y_train_temp2, y_train_temp1), axis=-1)), axis=0)
        y_test_temp1 = np.copy(y_test[:y_test.shape[0]])
        y_test_temp1[y_test_temp1 == 1] = P
        y_test_temp2 = np.copy(y_test[:y_test.shape[0]])
        y_test_temp2[y_test_temp2 == 1] = 1. - P
        y_test = np.concatenate((y_test_temp1, y_test_temp2), axis=-1)

        y_valid_temp1 = np.copy(y_valid[:y_valid.shape[0]])
        y_valid_temp1[y_valid_temp1 == 1] = P
        y_valid_temp2 = np.copy(y_valid[:y_valid.shape[0]])
        y_valid_temp2[y_valid_temp2 == 1] = 1. - P
        y_valid = np.concatenate((y_valid_temp1, y_valid_temp2), axis=-1)

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    FLAGS.nb_classes = nb_classes

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define model and Train model
    Model = get_model(data_name, nb_classes)
    Model.train(x_train, y_train, x_valid, y_valid, batch_size, nb_epochs, FLAGS.is_train)
    model = Model.model
    acc = model_eval(sess, x, y, model(x), x_test, y_test, batch_size=batch_size)
    print('Test accuracy on legitimate examples: %.4f' % acc)

    # Get select samples which are classifying correctly
    Y_pred_all = model_predicton(sess, x, y, model(x), x_test, y_test, batch_size=batch_size)
    correct_idx = get_correct_prediction_idx(Y_pred_all, y_test)
    if (FLAGS.dataset == 'gtsrb'):
        correct_and_selected_idx = get_first_n_examples_id_each_class(y_test[correct_idx], n=50)
    else:
        correct_and_selected_idx = get_first_n_examples_id_each_class(y_test[correct_idx], n=200)
    # correct_and_selected_idx = correct_idx
    if (FLAGS.use_cache == True):
        selected_idx = np.load("cache/%s_select.npy" % attack_string_hash)
    else:
        selected_idx = [correct_idx[i] for i in correct_and_selected_idx]
        # selected_idx = correct_idx
        # np.random.shuffle(selected_idx)
        # selected_idx = selected_idx[:2000]
    x_select, y_select, y_select_pred = x_test[selected_idx], y_test[selected_idx], Y_pred_all[selected_idx]
    accuracy_selected = calculate_accuracy(y_select_pred, y_select)
    print('Test accuracy on selected legitimate examples(should be 1.0000): %.4f' % (accuracy_selected))

    wrap = KerasModelWrapper(model)

    # use test mode，it may cause error when detection is lid or kd
    # K.set_learning_phase(0)

    cache_name = {"negative": 'negative', 'fs': 'origin', 'lid': 'origin', 'bu': 'origin', 'mlloo': 'origin'}
    # Get adversarial Examples
    if (FLAGS.use_adv_cache == True):
        x_adv_test = np.load(
            "cache/%s_%s_%s.npy" % (FLAGS.dataset, FLAGS.attack_type, cache_name[FLAGS.detection_type]))
        x_adv_test1 = np.load(
            "cache/%s_%s_%s.npy" % (FLAGS.dataset, 'fgsm', cache_name[FLAGS.detection_type]))
    else:
        x_adv_test = get_adv_examples(sess, wrap, attack_type, x_select, y_select)
    FLAGS.stdevs = np.std(x_adv_test1.reshape(len(x_adv_test1), -1) - x_select.reshape(len(x_adv_test1), -1),
                          axis=-1).mean()
    acc = model_eval(sess, x, y, model(x), x_adv_test, y_select, batch_size=128)
    print('Test accuracy on adversarial examples: %.4f' % acc)

    FLAGS.attack_type = 'fgsm'
    attack_type = 'fgsm'

    distortion = 0
    for i in range(len(x_adv_test)):
        distortion += np.linalg.norm(x_adv_test[i] - x_select[i])
    print("L2 norm: ", distortion / len(x_adv_test))

    # Detect adversarial Examples
    result_folder_detection = os.path.join(FLAGS.result_folder, data_name)
    csv_fname = "%s_attacks_%s_detection.csv" % (attack_type, attack_string_hash)
    de = get_detection(FLAGS.detection_type, model, sess, result_folder_detection, csv_fname, data_name)
    y_pred = model_predicton(sess, x, y, model(x), x_test, y_test)
    y_select_adv_pred = model_predicton(sess, x, y, model(x), x_adv_test, y_select)
    if (FLAGS.detection_type == 'negative'):
        de.build_detection_dataset(x_test, y_test, y_pred, selected_idx, x_adv_test,
                                   y_select_adv_pred, attack_type, attack_string_hash)
    elif (FLAGS.detection_type == 'fs'):
        de.build_detection_dataset(x_test, y_test, y_pred, selected_idx, [x_adv_test],
                                   [y_select_adv_pred], [attack_type], attack_string_hash)
    elif (FLAGS.detection_type == 'bu' or FLAGS.detection_type == 'lid' or FLAGS.detection_type == 'kd'):
        de.build_detection_dataset(model, x_train, y_train, x_test, y_test, x_adv_test, selected_idx)
    de.evaluate_detections(data_name)
    np.save("cache/%s_adv.npy" % attack_string_hash, x_adv_test)
    np.save("cache/%s_select.npy" % attack_string_hash, selected_idx)


def main(argv=None):
    detection(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size, data_name=FLAGS.dataset,
              attack_type=FLAGS.attack_type,
              )


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64,
                         'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 10,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128,
                         'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10,
                         'Number of classes')
    flags.DEFINE_float('train_fpr', 0.05,
                       'faes rate to decide threshold')
    flags.DEFINE_string('dataset', 'mnist', 'train dataset name')
    flags.DEFINE_string("attack_type", "fgsm", 'attack to detect')
    flags.DEFINE_string("test_attack_type", "fgsm", 'attack to detect')
    flags.DEFINE_bool("is_train", False, "train online or load from file")
    flags.DEFINE_string("detection_type", "negative", "which detection type to use")
    flags.DEFINE_string('result_folder', "results", 'The output folder for results.')
    flags.DEFINE_bool('use_cache', False, 'use history cache or get adversarial examples online')
    flags.DEFINE_bool('use_adv_cache', True, 'use history cache or get adversarial examples online')
    flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')
    flags.DEFINE_float('stdevs', 0.05,
                       'L-2 perturbation size is equal to that of the adversarial samples')
    flags.DEFINE_string('similarity_type', "cos", 'similarity index')
    flags.DEFINE_string('label_type', "type1", 'label assignment')
    tf.app.run()
