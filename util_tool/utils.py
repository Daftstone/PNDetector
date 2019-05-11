import numpy as np
from functools import reduce

from model.mnist_model import MNIST_model
from model.cifar10_model import CIFAR10_model
from model.fmnist_model import FMNIST_model
from model.svhn_model import SVHN_model
from detection.negtive_detection import DetectionEvaluator as neg_detection

from detection.fs_detection import DetectionEvaluator as fs_detection
from detection.lid_detection import DetectionEvaluator as lid_detection
from detection.bu_detection import DetectionEvaluator as bu_detection
from external.LID.util import get_model as get_lid_model


def get_umatrix(data):
    shape = data.shape
    u_matrix = np.zeros(shape, dtype=np.float)
    u_count = np.zeros(shape)
    u_matrix[:, 1:, 1:, :] = np.abs(data[:, 1:, 1:, :] - data[:, :-1, :-1, :])
    u_count[:, 1:, 1:, :] += 1
    u_matrix[:, 1:, :, :] = np.abs(data[:, 1:, :, :] - data[:, :-1, :, :])
    u_count[:, 1:, :, :] += 1
    u_matrix[:, 1:, :-1, :] = np.abs(data[:, 1:, :-1, :] - data[:, :-1, 1:, :])
    u_count[:, 1:, :-1, :] += 1
    u_matrix[:, :, 1:, :] = np.abs(data[:, :, 1:, :] - data[:, :, :-1, :])
    u_count[:, :, 1:, :] += 1
    u_matrix[:, :, :-1, :] = np.abs(data[:, :, :-1, :] - data[:, :, 1:, :])
    u_count[:, :, :-1, :] += 1
    u_matrix[:, :-1, 1:, :] = np.abs(data[:, :-1, 1:, :] - data[:, 1:, :-1, :])
    u_count[:, :-1, 1:, :] += 1
    u_matrix[:, :-1, :, :] = np.abs(data[:, :-1, :, :] - data[:, 1:, :, :])
    u_count[:, :-1, :, :] += 1
    u_matrix[:, :-1, :-1, :] = np.abs(data[:, :-1, :-1, :] - data[:, 1:, 1:, :])
    u_count[:, :-1, :-1, :] += 1
    temp = u_matrix / u_count
    for i in range(len(temp)):
        temp[i] = temp[i] / np.max(temp[i])
    return temp


def get_model(dataname,nb_classes):
    if (dataname == "mnist"):
        model = MNIST_model(nb_classes=nb_classes)
    elif (dataname == 'cifar10'):
        model = CIFAR10_model(nb_classes=nb_classes)
    elif (dataname == 'fmnist'):
        model = FMNIST_model(nb_classes=nb_classes)
    elif (dataname == 'svhn'):
        model = SVHN_model(nb_classes=nb_classes)
    else:
        model = None
        print("unknown model!!!")
    return model


def get_detection(detection_name, model, sess, result_folder_detection, csv_fname, data_name, attack_string_hash=None):
    if (detection_name == 'negative'):
        Detection = neg_detection(model, sess, result_folder_detection, csv_fname, data_name)
    elif (detection_name == 'fs'):
        Detection = fs_detection(model, attack_string_hash, data_name)
    elif (detection_name == 'lid'):
        Detection = lid_detection(model, attack_string_hash, data_name)
    elif (detection_name == 'bu'):
        Detection = bu_detection(model, attack_string_hash, data_name)
    else:
        Detection = None
    return Detection


def get_first_n_examples_id_each_class(Y_test, n=1):
    """
    Only return the classes with samples.
    """
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)

    selected_idx = []
    for i in range(num_classes):
        loc = np.where(Y_test_labels == i)[0]
        if len(loc) > 0:
            selected_idx.append(list(loc[:n]))

    selected_idx = reduce(lambda x, y: x + y, zip(*selected_idx))

    return np.array(selected_idx)


def get_correct_prediction_idx(Y_pred, Y_label):
    """
    Get the index of the correct predicted samples.
    :param Y_pred: softmax output, probability matrix.
    :param Y_label: groundtruth classes in shape (#samples, #classes)
    :return: the index of samples being corrected predicted.
    """
    pred_classes = np.argmax(Y_pred, axis=1)
    labels_classes = np.argmax(Y_label, axis=1)

    return np.where(pred_classes == labels_classes)[0]


def get_match_pred_vec(Y_pred, Y_label):
    assert len(Y_pred) == len(Y_label)
    Y_pred_class = np.argmax(Y_pred, axis=1)
    Y_label_class = np.argmax(Y_label, axis=1)
    return Y_pred_class == Y_label_class


def calculate_accuracy(Y_pred, Y_label):
    match_pred_vec = get_match_pred_vec(Y_pred, Y_label)

    accuracy = np.sum(match_pred_vec) / float(len(Y_label))
    # pdb.set_trace()
    return accuracy


def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]


def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]
