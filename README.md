# Detecting Adversarial Examples by Positive and Negative Representations

This project is for the paper "Detecting Adversarial Examples by Positive and 
Negative Representations". Some codes are from EvadeML, LID, BU 
and cleverhans.

The code was developed on Python 3.6.8


## 1. Install dependencies.
Our experiment runs on GPU,, install this list:
```bash
pip install -r requirements_gpu.txt
```

## 2. Download dataset and pre-trained models.
Download [dataset](https://drive.google.com/file/d/1gyBeIpy4WzO17_7hjZKR6flP9oPhHVfU/view) 
and extract it to the root of the program, which contains the MNIST, FMNIST, and CIFAR-10 dataset.

Download [pre-trained models](https://drive.google.com/open?id=1fld7SpilHwqnH_Lj7fxR5Au61OAU8qUs)
and extract it to the root of the program. 
## 3. Usage of `python main.py`
```
usage: python main.py [--dataset DATASET_NAME] [--attack_type ATTACK_NAME]
               [--detection_type DETECTION_NAME] [--is_train [IS_TRAIN]] [--use_cache [USE_CACHE]]
               [--nb_epochs EPOCHS_NUMBER] [--train_fpr FPR] 
               [-similarity_type SIMILARITY_INDEX]

optional arguments:
  --dataset DATASET_NAME
                        Supported: mnist, fmnist, svhn, cifar-10.
  --attack_type ATTACK_NAME
                        Supported: fgsm, lbfgs, df, enm, vam, cw, spsa, jsma.
  ----detection_type DETECTION_NAME
                        Supported: negative, lid, bu, fs.
  --is_train [IS_TRAIN]
                        User this parameter to train online, otherwise remove the parameter.
  --use_cache [USE_CACHE]
                        User this parameter to load adversarial examples from caches, otherwise remove the parameter.
  ----nb_epochs EPOCHS_NUMBER
                        Number of epochs the classifier is trained.
  --train_fpr FPR
                        set FPR
  --similarity_type
                        Supported: cos, l1, l2
```

### 4. Example.
Use pre-trained model.
```bash
python main.py --dataset mnist --attack_type fgsm \
--detection_type negative --train_fpr 0.05 --similarity_type cos
```
Train model online.
```bash
python main.py --dataset mnist --attack_type fgsm \
--nb_epochs 100 --detection_type negative \
--train_fpr 0.05 --is_train --similarity_type l1
```


## 5. Usage of `python run.py`
```
usage: python run.py [--dataset DATASET_NAME] [--detection_type DETECTION_NAME]
               [--train_fpr FPR] [-similarity_type SIMILARITY_INDEX]

optional arguments:
  --dataset DATASET_NAME
                        Supported: mnist, fmnist, svhn, cifar-10.
  ----detection_type DETECTION_NAME
                        Supported: negative, lid, bu, fs.
  --train_fpr FPR
                        set FPR
  --similarity_type
                        Supported: cos, l1, l2
```

### 6. Example.
```bash
python run.py --dataset mnist --detection_type negative \
--train_fpr 0.05 --similarity_type cos
```


## Cite this work

You are encouraged to cite the following paper if you use `PNDetector` for academic research.

```
pass
```
