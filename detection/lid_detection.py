import os
import warnings
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
from external.LID.util import (get_data, get_noisy_samples, get_mc_predictions, compute_roc,
                               get_deep_representations, score_samples, normalize, train_lr,
                               get_lids_random_batch, get_kmeans_random_batch, block_split)

BANDWIDTHS = {'mnist': 3.7926, 'cifar10': 0.26, 'svhn': 1.00}
batch_size = 128


def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv, dataset='mnist'):
    """
    Get kernel density scores
    :param model:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param X_test_noisy:
    :param X_test_adv:
    :return: artifacts: positive and negative examples with kd values,
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                     batch_size=batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=batch_size)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    print('bandwidth %.4f for %s' % (BANDWIDTHS[dataset], dataset))
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[dataset]) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model.predict_classes(X_test, verbose=0,
                                              batch_size=batch_size)
    preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
                                             batch_size=batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    print("densities_normal:", densities_normal.shape)
    print("densities_adv:", densities_adv.shape)
    print("densities_noisy:", densities_noisy.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
    #     densities_normal,
    #     densities_adv,
    #     densities_noisy
    # )

    densities_pos = densities_adv
    densities_neg = np.concatenate((densities_normal, densities_noisy))
    artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)

    return artifacts, labels


def get_bu(model, X_test, X_test_noisy, X_test_adv, batch_size=100):
    """
    Get Bayesian uncertainty scores
    :param model:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param X_test_noisy:
    :param X_test_adv:
    :return: artifacts: positive and negative examples with bu values,
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test,
                                        batch_size=batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                       batch_size=batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv,
                                     batch_size=batch_size) \
        .var(axis=0).mean(axis=1)

    print("uncerts_normal:", uncerts_normal.shape)
    print("uncerts_noisy:", uncerts_noisy.shape)
    print("uncerts_adv:", uncerts_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
    #     uncerts_normal,
    #     uncerts_adv,
    #     uncerts_noisy
    # )

    uncerts_pos = uncerts_adv
    uncerts_neg = np.concatenate((uncerts_normal, uncerts_noisy))
    artifacts, labels = merge_and_generate_labels(uncerts_pos, uncerts_neg)

    return artifacts, labels


def get_lid(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Get local intrinsic dimensionality
    :param model:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param X_test_noisy:
    :param X_test_adv:
    :return: artifacts: positive and negative examples with lid values,
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, dataset, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # lids_normal_z, lids_adv_z, lids_noisy_z = normalize(
    #     lids_normal,
    #     lids_adv,
    #     lids_noisy
    # )

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels


class DetectionEvaluator:
    def __init__(self, model, attack_string_hash, dataset_name):
        pass

    def build_detection_dataset(self, model, X_train, Y_train, X_test, Y_test, X_test_adv, inds_correct):
        k_nearest = 20

        X_test = X_test[inds_correct]
        Y_test = Y_test[inds_correct]
        pred = np.argmax(model.predict(X_test_adv), axis=1)
        X_test_adv = X_test_adv[pred != np.argmax(Y_test, axis=1)]
        X_test = X_test[pred != np.argmax(Y_test, axis=1)]
        X_test_noisy = get_noisy_samples(X_test, X_test_adv, FLAGS.dataset, FLAGS.attack_type)
        print("Number of correctly predict images: %s" % (len(inds_correct)))

        self.start=time.clock()
        # extract local intrinsic dimensionality
        if (FLAGS.detection_type == 'lid'):
            characteristics, labels = get_lid(model, X_test, X_test_noisy, X_test_adv,
                                              k_nearest, batch_size, FLAGS.dataset)
        elif (FLAGS.detection_type == 'bu'):
            characteristics, labels = get_bu(model, X_test, X_test_noisy, X_test_adv, batch_size)
        elif (FLAGS.detection_type == 'kd'):
            characteristics, labels = get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv, FLAGS.dataset)
        print("LID: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        self.data = np.concatenate((characteristics, labels), axis=1)
        print("hello")
        print(len(self.data))

    def evaluate_detections(self, dataname):
        X, Y = self.load_characteristics()

        # standarization
        scaler = MinMaxScaler().fit(X)
        X = scaler.transform(X)
        # X = scale(X) # Z-norm

        # test attack is the same as training attack
        X_train, Y_train, X_test, Y_test = block_split(X, Y)

        print("Train data size: ", X_train.shape)
        print("Test data size: ", X_test.shape)

        ## Build detector
        print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
              (FLAGS.dataset, FLAGS.attack_type, FLAGS.test_attack_type))
        lr = train_lr(X_train, Y_train)

        ## Evaluate detector
        y_pred = lr.predict_proba(X_test)[:, 1]
        y_label_pred = lr.predict(X_test)

        # AUC
        _, _, auc_score = compute_roc(Y_test, y_pred, plot=False)
        precision = precision_score(Y_test, y_label_pred)
        recall = recall_score(Y_test, y_label_pred)

        y_label_pred = lr.predict(X_test)
        self.end=time.clock()
        acc = accuracy_score(Y_test, y_label_pred)
        print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (
            auc_score, acc, precision, recall))
        print("time: %.5f" % ((self.end - self.start) / X_test.shape[0]))
        return lr, auc_score, scaler

    def load_characteristics(self):
        """
        Load multiple characteristics for one dataset and one attack.
        :param dataset:
        :param attack:
        :param characteristics:
        :return:
        """
        data = self.data
        X = data[:, :-1]
        Y = data[:, -1]
        return X, Y
