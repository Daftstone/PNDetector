import tensorflow as tf
import numpy as np
from keras import backend as K
from util_tool.utils_tf import get_adv
from tensorflow.python.platform import flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import LBFGS
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import SPSA

FLAGS = flags.FLAGS


def get_adv_examples(sess, wrap, attack_type, X, Y):
    """
        detect adversarial examples
        :param sess: target model session
        :param wrap: wrap model
        :param attack_type:  attack for generating adversarial examples
        :param X: examples to be attacked
        :param Y: correct label of the examples
        :return: x_adv: adversarial examples
    """
    x = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2],
                                          X.shape[3]))
    y = tf.placeholder(tf.float32, shape=(None, Y.shape[1]))
    adv_label = np.copy(Y)
    batch_size = 128

    # Define attack method parameters
    if (attack_type == 'fgsm'):
        attack_params = {
            'eps': 0.1,
            'clip_min': 0.,
            'clip_max': 1.
        }
        attack_object = FastGradientMethod(wrap, sess=sess)
    elif (attack_type == 'jsma'):
        attack_params = {
            'theta': 1., 'gamma': 0.1,
            'clip_min': 0., 'clip_max': 1.,
            'y_target': None
        }
        attack_object = SaliencyMapMethod(wrap, sess=sess)
        batch_size = 32
    elif (attack_type == 'cw'):
        attack_params = {
            'binary_search_steps': 1,
            'y': y,
            'max_iterations': 100,
            'learning_rate': .2,
            'batch_size': 128,
            'initial_const': 10
        }
        attack_object = CarliniWagnerL2(wrap, sess=sess)
    elif (attack_type == 'mim'):
        attack_object = MomentumIterativeMethod(wrap, back='tf', sess=sess)
        attack_params = {'clip_min': 0., 'clip_max': 1., 'eps': 0.1}
    elif (attack_type == 'df'):
        attack_params = {
            'max_iterations': 50,
            'clip_min': 0., 'clip_max': 1.,
            'overshoot': 0.02
        }
        attack_object = DeepFool(wrap, sess=sess)
        batch_size = 64
    elif (attack_type == 'bim'):
        attack_object = BasicIterativeMethod(wrap, back='tf', sess=sess)
        attack_params = {'eps': 0.1, 'eps_iter': 0.05,
                         'nb_iter': 10, 'clip_min': 0.,
                         'clip_max': 1.
                         }
    elif (attack_type == 'vam'):
        attack_object = VirtualAdversarialMethod(wrap, back='tf', sess=sess)
        attack_params = {'clip_min': 0., 'clip_max': 1., 'nb_iter': 100, 'eps': 2, 'xi': 1e-6}
    elif (attack_type == 'enm'):
        attack_object = ElasticNetMethod(wrap, back='tf', sess=sess)
        attack_params = {'y': y, 'max_iterations': 10, 'batch_size': 128}
    elif (attack_type == 'spsa'):
        attack_object = SPSA(wrap, sess=sess)
        adv_x = attack_object.generate(x=x, y=y, eps=0.1, clip_min=0., clip_max=1., nb_iter=100,
                                       early_stop_loss_threshold=-5.)
        batch_size = 1
    elif (attack_type == 'lbfgs'):
        attack_object = LBFGS(wrap, sess=sess)
        attack_params = {'clip_min': 0, 'clip_max': 1., 'batch_size': 128,
                         'max_iterations': 10, "y_target": y}
        true_label = np.argmax(Y, axis=-1)
        for i in range(len(Y)):
            ind = (true_label[i] + 1) % FLAGS.nb_classes
            adv_label[i] = np.zeros([FLAGS.nb_classes])
            adv_label[i, ind] = 1
    if (attack_type != 'spsa'):
        adv_x = attack_object.generate(x, **attack_params)

    # Get adversarial examples
    if (attack_type == 'lbfgs'):
        x_adv = get_adv(sess, x, y, adv_x, X, adv_label, batch_size=batch_size)
    else:
        x_adv = get_adv(sess, x, y, adv_x, X, Y, batch_size=batch_size)
    return x_adv
