# Efficient Spiking Neural Network Design via Neural Architecture Search

## Introduction
This is the TensorFlow implementation code for [Efficient Spiking Neural Network Design via Neural Architecture Search].

## Requirements
* Python 3.9    
* TensorFlow-gpu 2.5.1    
* NVIDIA GPU (>= 12GB) + CUDA    

## Dataset Preparation
The CIFAR-10 is avaliable at [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Employ NAS Search 
We provide example code to train and test on CIFAR10 dataset for reproducing and evaluating our method. Code for all datasets will be publicly available after the paper is accepted.

Reference the following steps to run this code:

### Search on CIFAR10:
```
python search_main.py --dataset_name='CIFAR10' --data_dir='./cifar-10-python/' --mode='train_and_eval' --repeat_num=9 --loss_lambda=0.1 --warmup_steps=10000 --train_steps=40000 --num_train_images=50000 --num_eval_images=10000 --num_label_classes=10 
```

The model results obtained during the search phase are saved at [ESNN_Search/NAS_search_results](/ESNN_Search/).


### Train the searched SNN on CIFAR10
```
python main.py --dataset_name='CIFAR10' --data_dir='./cifar-10-python/' --mode='train_and_eval' --train_steps=700000 --num_train_images=50000 --num_eval_images=10000 --num_label_classes=10 
```
Checkpoint data obtained during the search phase needs to be put in [ESNN_Retrain_CIFAR10/NAS_search_checkpoints](/ESNN_Retrain_CIFAR10/). 

events.out.tfevents obtained during the search phase needs to be put in [ESNN_Retrain_CIFAR10/NAS_search_results](/ESNN_Retrain_CIFAR10/). 


### Evaluate the searched SNN on CIFAR10
```
python main.py --dataset_name='CIFAR10' --data_dir='./cifar-10-python/' --mode='eval' --train_steps=700000 --num_train_images=50000 --num_eval_images=10000 --num_label_classes=10 
```
