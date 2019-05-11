import os
import warnings
import time
import numpy as np
from sklearn.neighbors import KernelDensity
from tensorflow.python.platform import flags
from util_tool.utils_bu import (get_data, get_noisy_samples, get_mc_predictions,
                           get_deep_representations, score_samples, normalize,
                           train_lr, compute_roc)

FLAGS = flags.FLAGS
BANDWIDTHS = {'mnist': 0.05, 'cifar10': 0.26, 'svhn': 1.00, 'fmnist': 0.05}


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
        self.X_test = X_test
        X_test_noisy = get_noisy_samples(X_test, X_test_adv, FLAGS.dataset, FLAGS.attack_type)
        print("Number of correctly predict images: %s" % (len(inds_correct)))

        self.start = time.clock()
        ## Get Bayesian uncertainty scores
        print('Getting Monte Carlo dropout variance predictions...')
        uncerts_normal = get_mc_predictions(model, X_test,
                                            batch_size=FLAGS.batch_size) \
            .var(axis=0).mean(axis=1)
        uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                           batch_size=FLAGS.batch_size) \
            .var(axis=0).mean(axis=1)
        uncerts_adv = get_mc_predictions(model, X_test_adv,
                                         batch_size=FLAGS.batch_size) \
            .var(axis=0).mean(axis=1)

        ## Get KDE scores
        # Get deep feature representations
        print('Getting deep feature representations...')
        X_train_features = get_deep_representations(model, X_train,
                                                    batch_size=FLAGS.batch_size)
        X_test_normal_features = get_deep_representations(model, X_test,
                                                          batch_size=FLAGS.batch_size)
        X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                         batch_size=FLAGS.batch_size)
        X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                       batch_size=FLAGS.batch_size)

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
        for i in range(Y_train.shape[1]):
            kdes[i] = KernelDensity(kernel='gaussian',
                                    bandwidth=BANDWIDTHS[FLAGS.dataset]) \
                .fit(X_train_features[class_inds[i]])
        # Get model predictions
        print('Computing model predictions...')
        # preds_test_normal = model.predict_classes(X_test, verbose=0,
        #                                           batch_size=FLAGS.batch_size)
        # preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
        #                                          batch_size=FLAGS.batch_size)
        # preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
        #                                        batch_size=FLAGS.batch_size)
        preds_test_normal = np.argmax(model.predict(X_test, verbose=0,
                                                    batch_size=FLAGS.batch_size), axis=1)
        preds_test_noisy = np.argmax(model.predict(X_test_noisy, verbose=0,
                                                   batch_size=FLAGS.batch_size), axis=1)
        preds_test_adv = np.argmax(model.predict(X_test_adv, verbose=0,
                                                 batch_size=FLAGS.batch_size), axis=1)

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

        ## Z-score the uncertainty and density values
        uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
            uncerts_normal,
            uncerts_adv,
            uncerts_noisy
        )
        densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
            densities_normal,
            densities_adv,
            densities_noisy
        )

        ## Build detector
        self.values, labels, self.lr = train_lr(
            densities_pos=densities_adv_z,
            densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
            uncerts_pos=uncerts_adv_z,
            uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
        )

    def evaluate_detections(self, dataname):
        ## Evaluate detector
        # Compute logistic regression model predictions
        probs = self.lr.predict_proba(self.values)[:, 1]
        # Compute AUC
        n_samples = len(self.X_test)
        # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
        # and the last 1/3 is the positive class (adversarial samples).
        _, _, auc_score = compute_roc(
            probs_neg=probs[:2 * n_samples],
            probs_pos=probs[2 * n_samples:]
        )
        self.end = time.clock()
        print('Detector ROC-AUC score: %0.4f' % auc_score)
        print("time: %.5f" % ((self.end - self.start) / (n_samples * 3)))

    def load_characteristics(self, dataset, attack, characteristics):
        """
        Load multiple characteristics for one dataset and one attack.
        :param dataset:
        :param attack:
        :param characteristics:
        :return:
        """
        X, Y = None, None
        for characteristic in characteristics:
            data = self.data
            if X is None:
                X = data[:, :-1]
            else:
                X = np.concatenate((X, data[:, :-1]), axis=1)
            if Y is None:
                Y = data[:, -1]  # labels only need to load once

        return X, Y
