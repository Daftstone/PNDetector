import os
import warnings
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.platform import flags
import tensorflow as tf
import math
import tensorflow.keras.backend as K

FLAGS = flags.FLAGS
from external.LID.util import (get_data, get_noisy_samples, get_mc_predictions, compute_roc,
                               get_deep_representations, score_samples, normalize, train_lr,
                               get_lids_random_batch, get_kmeans_random_batch, block_split)

BANDWIDTHS = {'mnist': 3.7926, 'cifar10': 0.26, 'svhn': 1.00}
batch_size = 128

from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew

def con(score):
  # score (n, d)
  score = score.reshape(len(score), -1)
  score_mean = np.mean(score, -1, keepdims = True)
  c_score = score - score_mean
  c_score = np.abs(c_score)
  return np.mean(c_score, axis = -1)

def mad(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    median = np.median(d)
    abs_dev = np.abs(d - median)
    med_abs_dev = np.median(abs_dev)
    pd.append(med_abs_dev)
  pd = np.array(pd)
  return pd

def med_pdist(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    k = np.median(pdist(d.reshape(-1,1)))
    pd.append(k)
  pd = np.array(pd)
  return pd

def pd(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    k = np.mean(pdist(d.reshape(-1,1)))
    pd.append(k)
  pd = np.array(pd)
  return pd

def neg_kurtosis(score):
  k = []
  for i in range(len(score)):
    di = score[i]
    ki = kurtosis(di, nan_policy = 'raise')
    k.append(ki)
  k = np.array(k)
  return -k

def quantile(score):
    # score (n, d)
  score = score.reshape(len(score), -1)
  score_75 = np.percentile(score, 75, -1)
  score_25 = np.percentile(score, 25, -1)
  score_qt = score_75 - score_25
  return score_qt

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

def collect_layers(model, interested_layers):
    #if model.framework == 'keras':
    outputs = [layer.output for layer in model.layers]

    outputs = [output for i, output in enumerate(outputs) if i in interested_layers]
    print(outputs)
    features = []
    for output in outputs:
        print(output)
        if len(output.get_shape())== 4:
            features.append(
                tf.reduce_mean(output, axis = (1, 2))
            )
        else:
            features.append(output)
    return features

def calculate(score, stat_name):
    if stat_name == 'variance':
        results = np.var(score, axis = -1)
    elif stat_name == 'std':
        results = np.std(score, axis = -1)
    elif stat_name == 'pdist':
        results = pd(score)
    elif stat_name == 'con':
        results = con(score)
    elif stat_name == 'med_pdist':
        results = med_pdist(score)
    elif stat_name == 'kurtosis':
        results = neg_kurtosis(score)
    elif stat_name == 'skewness':
        results = -skew(score, axis = -1)
    elif stat_name == 'quantile':
        results = quantile(score)
    elif stat_name == 'mad':
        results = mad(score)
    #print('results.shape', results.shape)
    return results

def evaluate_features(x, model, features):
    x = np.array(x)
    if len(x.shape) == 3:
        _x = np.expand_dims(x, 0)
    else:
        _x = x

    batch_size = 500
    num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))

    outs = []
    for i in range(num_iters):
        x_batch = _x[i * batch_size: (i+1) * batch_size]
        sess = K.get_session()
        out = sess.run(features,
            feed_dict = {model.input: x_batch})

        outs.append(out)

    num_layers = len(outs[0])
    outputs = []
    for l in range(num_layers):
        outputs.append(np.concatenate([outs[s][l] for s in range(len(outs))]))

    # (3073, 64)
    # (3073, 64)
    # (3073, 128)
    # (3073, 128)
    # (3073, 256)
    # (3073, 256)
    # (3073, 10)
    # (3073, 1)
    outputs = np.concatenate(outputs, axis = 1)
    #prob = outputs[:,-model.num_classes:]
    prob = outputs[:, -10:]
    label = np.argmax(prob[-1])
    #print('outputs', outputs.shape)
    #print('prob[:, label]', np.expand_dims(prob[:, label], axis = 1).shape)
    outputs = np.concatenate([outputs, np.expand_dims(prob[:, label], axis = 1)], axis = 1)

    return outputs

def loo_ml_instance(batch_sample, reference, model, features):
    n_feed, h, w, c = batch_sample.shape
    batch_sample = batch_sample.reshape(n_feed, -1)
    reference = reference.reshape(-1)

    batch_data = []
    st = time.time()
    # positions = np.ones((h * w * c + 1, h * w * c), dtype=np.bool)
    # for i in range(h * w * c):
    #     positions[i, i] = False
    positions = np.logical_not(np.eye(h * w * c + 1, h * w * c, dtype=bool))
    batch_positions = np.repeat(np.expand_dims(positions, axis=0), n_feed, axis=0)
    batch_sample = np.repeat(np.expand_dims(batch_sample, axis=1), h * w * c + 1, axis=1)
    batch_data = np.where(batch_positions, batch_sample, reference)
    batch_data = batch_data.reshape((n_feed, -1, h, w, c))

    batch_features_val = []
    for i in range(n_feed):
        features_val = evaluate_features(batch_data[i], model, features)  # (3072+1, 906+1)
        batch_features_val.append(features_val)
    st1 = time.time()
    batch_features_val = np.asarray(batch_features_val)
    return batch_features_val

def generate_ml_loo_features(model, X_test, X_test_noisy, X_test_adv, reference, interested_layers, batch_size):
    features = collect_layers(model, interested_layers)

    stat_name = 'quantile'

    x = {'original': X_test, 'noisy': X_test_noisy, 'adv': X_test_adv}
    combined_features = {}
    for data_type in ['original', 'noisy', 'adv']:
        all_features = []
        xs = x[data_type]
        n_batches = int(np.ceil(xs.shape[0] / float(batch_size)))
        for i_batch in range(n_batches):

            start = i_batch * batch_size
            end = np.minimum(len(xs), (i_batch + 1) * batch_size)
            n_feed = end - start
            batch_sample = xs[start:end]
            #print('Generating ML-LOO for {}th sample...'.format(i))
            batch_features_val = loo_ml_instance(batch_sample, reference, model, features)

            # (3073, 907)
            #print('features_val.shape', features_val.shape)
            batch_features_val = np.transpose(batch_features_val, (0, 2, 1))[:, :, :-1]
            #print('features_val.shape', features_val.shape)
            # (906, 3073)

            batch_feature = []
            for features_val in batch_features_val:
            #print('stat_name', stat_name)
                batch_feature.append(calculate(features_val, stat_name))

            batch_feature = np.array(batch_feature)
            #print('single_feature', single_feature.shape)
            # (k, 906)
            all_features.extend(batch_feature)
        #print('all_features', np.array(all_features).shape)
        #IQR value
        all_features = np.array(all_features)
        combined_features[data_type] = all_features

    loo_normal = combined_features['original']
    loo_noisy = combined_features['noisy']
    loo_adv = combined_features['adv']
    return loo_normal, loo_noisy, loo_adv


def get_loo(model, X_test, X_test_noisy, X_test_adv, batch_size=100, dataset='mnist'):
    # all interested_layers in random select
    if dataset == 'mnist':
        # 7
        # "activation_1/Relu:0", "activation_3/Relu:0", "activation_5/Relu:0", "activation_6/Softmax:0"
        interested_layers = [3, 8, 14, 16]
    elif dataset == 'cifar10':
        # vgg
        # "activation_3/Relu:0", activation_6/Relu:0", "activation_9/Relu:0", "activation_12/Relu:0", "max_pooling2d_4/MaxPool:0"
        interested_layers = [13, 25, 37, 49, 51]
    elif dataset == 'svhn' or dataset == 'fmnist':
        # 8
        # "conv2d_1_2/Relu:0", "conv2d_3_2/Relu:0", "conv2d_5_1/Relu:0". "conv2d_5_1/Relu:0". "activation_1/Softmax:0"
        interested_layers = [1, 4, 7, 13]

    x_train_mean = np.zeros(X_test.shape[1:])
    print('extracting layers ', interested_layers)
    reference = -x_train_mean

    loo_normal, loo_noisy, loo_adv  = generate_ml_loo_features(model, X_test, X_test_noisy, X_test_adv, reference, interested_layers, batch_size)
    labels = np.ones(len(loo_normal)+len(loo_noisy)+len(loo_adv))
    labels[0:len(loo_normal)+len(loo_noisy)] = 0
    labels = labels.reshape(-1, 1)
    characteristics = np.concatenate((loo_normal, loo_noisy, loo_adv))
    return characteristics, labels

class DetectionEvaluator:
    def __init__(self, model, attack_string_hash, dataset_name):
        pass

    def build_detection_dataset(self, model, X_train, Y_train, X_test, Y_test, X_test_adv, inds_correct):
        X_test = X_test[inds_correct]
        Y_test = Y_test[inds_correct]
        pred = np.argmax(model.predict(X_test_adv), axis=1)
        X_test_adv = X_test_adv[pred != np.argmax(Y_test, axis=1)]
        X_test = X_test[pred != np.argmax(Y_test, axis=1)]
        X_test_noisy = get_noisy_samples(X_test, X_test_adv, FLAGS.dataset, FLAGS.attack_type)
        print("Number of correctly predict images: %s" % (len(inds_correct)))

        self.start=time.clock()
        # extract local intrinsic dimensionality
        characteristics, labels = get_loo(model, X_test, X_test_noisy, X_test_adv, batch_size=100, dataset='mnist')
        print("ML-LOO: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        self.data = np.concatenate((characteristics, labels), axis=1)
        print("hello")
        print(len(self.data))

    def evaluate_detections(self, dataname):
        X, Y = self.load_characteristics()

        #standarization
        scaler = MinMaxScaler().fit(X)
        X = scaler.transform(X)
        X = scale(X) # Z-norm

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
        return lr, auc_score

    def evaluate_detections_fgsm(self, dataname):
        from  sklearn.externals import  joblib

        X, Y = self.load_characteristics()

        #standarization
        scaler = MinMaxScaler().fit(X)
        X = scaler.transform(X)
        X = scale(X) # Z-norm

        np.save('./weights/{}/mlloo_{}_lr_x_norm.npy'.format(dataname, FLAGS.attack_type), X)
        np.save('./weights/{}/mlloo_{}_lr_y.npy'.format(dataname, FLAGS.attack_type), Y)
        # test attack is the same as training attack
        X_train, Y_train, X_test, Y_test = block_split(X, Y)

        print("Train data size: ", X_train.shape)
        print("Test data size: ", X_test.shape)

        ## Build detector
        print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
              (FLAGS.dataset, FLAGS.attack_type, FLAGS.test_attack_type))

        fgsm_lr_model_path = './weights/{}/mlloo_fgsm_lr.model' .format(dataname)
        if os.path.exists(fgsm_lr_model_path):
            print('load: {}'.format(fgsm_lr_model_path))
            lr = joblib.load(fgsm_lr_model_path)
        elif FLAGS.attack_type == 'fgsm':
            lr = train_lr(X_train, Y_train)
            print('save: {}'.format(fgsm_lr_model_path))
            joblib.dump(lr, fgsm_lr_model_path)

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
        return lr, auc_score

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
