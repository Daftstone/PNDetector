# Detecting Adversarial Examples by Positive and Negative Representations

This project is for the paper "Detecting Adversarial Examples by Positive and 
Negative Representations". Some codes are from EvadeML, LID, BU 
and cleverhans.

The code was developed on Python 3.6


## 1. Install dependencies.
Our experiment runs on GPU,, install this list:
```bash
pip install -r requirements_gpu.txt
```

## 2. Download dataset and pre-trained models.
Download [dataset](https://drive.google.com/file/d/1gyBeIpy4WzO17_7hjZKR6flP9oPhHVfU/view) 
and extract it to the root of the program, which contains the MNIST, FMNIST, and SVHN dataset.

Download [pre-trained models](https://drive.google.com/open?id=1c1rKZzbYP5AELJhl5Uxs1bFzHU04sRnx)
and extract it to the root of the program. 
## 3. Usage of `python main.py`
```
usage: python main.py [--dataset DATASET_NAME] [--attack_type ATTACK_NAME]
               [--detection_type DETECTION_NAME] [--is_train [IS_TRAIN]] [--use_cache [USE_CACHE]]
               [--nb_epochs EPOCHS_NUMBER] [--train_fpr FPR] 
               [-label_type LABEL_ASSIGNMENT]

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
  --label_type
                        Supported: type1, type2, type3
```

### 4. Example.
Use pre-trained model.
```bash
python main.py --dataset mnist --attack_type fgsm \
--detection_type negative --train_fpr 0.05 --label_type type1
```
Train model online.
```bash
python main.py --dataset mnist --attack_type fgsm \
--nb_epochs 100 --detection_type negative \
--train_fpr 0.05 --is_train --label_type type1
```


## 5. Usage of `python run.py`
```
usage: python run.py [--dataset DATASET_NAME] [--detection_type DETECTION_NAME]
               [--train_fpr FPR] [-label_type LABEL ASSIGNMENT]

optional arguments:
  --dataset DATASET_NAME
                        Supported: mnist, fmnist, svhn, cifar-10.
  ----detection_type DETECTION_NAME
                        Supported: negative, lid, bu, fs.
  --train_fpr FPR
                        set FPR
  --label_type
                        Supported: type1, type2, type3
```

### 6. Example.
```bash
python run.py --dataset mnist --detection_type negative \
--train_fpr 0.05 --label_type type1
```


