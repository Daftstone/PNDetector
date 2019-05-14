import os

from tensorflow.python.platform import flags


def get_file(detection, dataset):
    index = 0
    while (os.path.exists("results/%s_%s_%d.txt" % (detection, dataset, index))):
        index += 1
    file = open("results/%s_%s_%d.txt" % (detection, dataset, index), 'w')
    return file


def write_file(file, mgs):
    for line in mgs.split('\n'):
        file.write(line + "\n")


# define parameter
flags.DEFINE_string("dataset", 'mnist', 'test dataset')
flags.DEFINE_string("detection_type", 'negative', 'detection method')
flags.DEFINE_float("train_fpr", 0.05, 'to calculate threshold')
flags.DEFINE_string("label_type", 'type1', 'label assignment')

attack_method = ['fgsm', 'lbfgs', 'df', 'enm', "vam", 'cw', 'spsa', 'jsma']
# attack_method = ['fgsm',  'lbfgs', 'df', 'enm', "vam", 'cw', 'spsa']
file = get_file(flags.FLAGS.detection_type, flags.FLAGS.dataset)
for attack in attack_method:
    msg = os.popen(
        "python main.py --attack_type %s --dataset %s --train_fpr %f --detection_type %s --label_type %s" % (
            attack, flags.FLAGS.dataset, flags.FLAGS.train_fpr, flags.FLAGS.detection_type,
            flags.FLAGS.label_type)).read()
    file.write(attack + ":\n")
    write_file(file, msg)
file.close()
