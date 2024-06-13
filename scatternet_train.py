import argparse
import os

import torch
import torch.nn as nn
from opacus import PrivacyEngine

from utils import get_device, train, test, CheXpert_train, CheXpert_test
from data import get_data, get_scatter_transform, create_scattered_features
from _models.scatternet_model import CNNS, get_num_params, ScatterLinear
from data_normalization import ORDERS, data_normalization

from torchvision import models
from opacus.validators import ModuleValidator
import time

from libauc.optimizers import PESG, Adam
from torch_ema import ExponentialMovingAverage

from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import TorchTensorField, IntField, NDArrayField
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder
from ffcv.transforms import ToTensor

from torch.multiprocessing import Pool, Process, set_start_method

import numpy as np
from sklearn.model_selection import ParameterGrid
import math
from torch.utils.data import random_split
from sklearn.linear_model import LogisticRegression

def store_scatter_features(dataset, augment=False, size=None,
        minibatch_size=256, sample_batches=False,
        lr=1, optim="Adam", momentum=0.9, nesterov=False,
        clip_norm=0.1, epochs=100,
        norm_flag=None, group_norm_groups=None, data_norm_sigma=None,
        epsilon=None, logdir=None,
        max_physical_batch_size=128, privacy=True,
        weight_standardization=True, ema_flag=True, write_file=None, 
        grad_sample_mode="no_op", log_file=None,
        aug_multiplicity=False, n_augs=8, device="cpu", 
        result_file_csv=None, output_dim=5, layer='cnn'):

    print("CUDA: ", torch.cuda.is_available())

    train_data, test_data = get_data(dataset, augment=augment)
    print("Size of train data and test data: ", len(train_data), len(test_data))

    scattering, K, _ = get_scatter_transform(dataset)
    scattering.to(device)
    print("K: ", K)


    # SET UP LOADER
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=minibatch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=minibatch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Create features
    create_scattered_features(train_loader, scattering, device, aug_multiplicity, n_augs, dataset=dataset, data_type='train')
    create_scattered_features(test_loader, scattering, device, aug_multiplicity, n_augs, dataset=dataset, data_type='test')

def main(dataset, augment=False, size=None,
        minibatch_size=256, sample_batches=False,
        lr=1, optim="SGD", momentum=0.9, nesterov=False,
        clip_norm=0.1, epochs=100,
        norm_flag=None, group_norm_groups=None, data_norm_sigma=None,
        epsilon=None, logdir=None,
        max_physical_batch_size=128, privacy=False,
        weight_standardization=True, ema_flag=True,
        grad_sample_mode="no_op", log_file=None,
        checkpoin_dir=None, aug_multiplicity=False, n_augs=8,
        device="cpu", result_file_csv=None, output_dim=5, layer='cnn'):
   
    print("CUDA:", torch.cuda.is_available())

    train_data, test_data = get_data(dataset, augment=augment)
    print("Non-private. Data size of tensor T: ", train_data[0][0].shape, train_data[0][1].dtype)


    scattering, K, (h, w) = None, 243, (56, 56)


    # SET UP LOADER
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=minibatch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=minibatch_size, shuffle=False, num_workers=0, pin_memory=False)


    rdp_norm = 0
    if norm_flag == "DataNorm":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = data_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=data_norm_sigma,
                                                   orders=ORDERS,
                                                   save_dir=save_dir)
        if layer == 'cnn':
            model = CNNS[dataset](K, input_norm="DataNorm", bn_stats=bn_stats, size=size, weight_standardization=weight_standardization)
        else:
            model = ScatterLinear(K, (h, w), input_norm="DataNorm", classes = output_dim, bn_stats=bn_stats)
    else:
        if layer == 'cnn':
            model = CNNS[dataset](K, input_norm=norm_flag, num_groups=group_norm_groups, size=size, weight_standardization=weight_standardization)
        else:
            model = ScatterLinear(K, (h, w), input_norm=norm_flag, classes=output_dim, num_groups=group_norm_groups)

    model.to(device)

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ema_flag:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    else:    
        ema = None

    log_file.write(f'Minibatch size: {minibatch_size}\nLearning Rate: {lr}\nOptimizer: {optim}\n')

    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")
        
        if dataset == 'chexpert_tensors':
            train_loss, train_acc, train_val_auc_mean = CheXpert_train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = CheXpert_test(model, test_loader, ema)
        elif dataset == 'eyepacs_tensors':
            train_loss, train_acc, train_val_auc_mean = train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = test(model, test_loader, ema)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_file = f'/u2/s4mokhta/checkpoints/cnn/{dataset}_epochs{epochs}_privacy{privacy}_Optim{optim}_LR{lr}_BatchSize{minibatch_size}_checkpoint_{epoch}.pth'
        torch.save(checkpoint, checkpoint_file)
        
        log_file.write(f'Train set: Average loss: {train_loss}, Accuracy: {train_acc}, AUC: {train_val_auc_mean}\n')
        log_file.write(f'Test set: Average loss: {test_loss}, Accuracy: {test_acc}, AUC: {test_val_auc_mean}\n')

def private_main(dataset, augment=False, size=None,
        minibatch_size=256, sample_batches=False,
        lr=1, optim="Adam", momentum=0.9, nesterov=False,
        clip_norm=0.1, epochs=100,
        norm_flag=None, group_norm_groups=None, data_norm_sigma=None,
        epsilon=None, logdir=None,
        max_physical_batch_size=128, privacy=True,
        weight_standardization=True, ema_flag=True, write_file=None, 
        grad_sample_mode="no_op", log_file=None,
        aug_multiplicity=False, n_augs=8, device="cpu", 
        result_file_csv=None, output_dim=5, layer='cnn'):

    print("CUDA: ", torch.cuda.is_available())

    train_data, test_data = get_data(dataset, augment=augment)
    print("Size of train data and test data: ", len(train_data), len(test_data))


    n = len(train_data)
    DELTA = 10 ** -(math.ceil(math.log10(n)))
    print("DELTA: ", DELTA)

    scattering, K, (h, w) = None, 243, (56, 56)


    # SET UP LOADER
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=minibatch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=minibatch_size, shuffle=False, num_workers=0, pin_memory=False)


    rdp_norm = 0
    if norm_flag == "DataNorm":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = data_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=data_norm_sigma,
                                                   orders=ORDERS,
                                                   save_dir=save_dir,
                                                   grad_sample_mode=grad_sample_mode)
        if layer == 'cnn':
            model = CNNS[dataset](K, input_norm="DataNorm", bn_stats=bn_stats, size=size, weight_standardization=weight_standardization, grad_sample_mode=grad_sample_mode)
        else:
            model = ScatterLinear(K, (h, w), input_norm="DataNorm", classes=output_dim, bn_stats=bn_stats)
    else:
        if layer == 'cnn':
            model = CNNS[dataset](K, input_norm=norm_flag, num_groups=group_norm_groups, size=size, weight_standardization=weight_standardization, grad_sample_mode=grad_sample_mode)
        else:
            model = ScatterLinear(K, (h, w), input_norm=norm_flag, classes=output_dim, num_groups=group_norm_groups)

    model.to(device)

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        print(model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ema_flag:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    else:
        ema = None

    privacy_engine = PrivacyEngine(accountant='prv')

    if grad_sample_mode == "no_op":
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=DELTA,
                max_grad_norm=clip_norm,
                clipping="flat",
                grad_sample_mode="no_op",
            )
    else:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=DELTA,
            max_grad_norm=clip_norm,
        )

    noise_multiplier = optimizer.noise_multiplier # change it to for private experiments optimizer.noise_multiplier o/w 0
    print(f"Using sigma={optimizer.noise_multiplier} and C={clip_norm}")


    log_file.write(f'Minibatch size: {minibatch_size}\nLearning Rate: {lr}\nOptimizer: {optim}\n')
    log_file.write(f'Epochs: {epochs}\nPrivacy: {privacy}\nMaximum Epsilon: {epsilon}\nDelta: {DELTA}\n'
                      f'Clip norm: {clip_norm}\n, Noise Multiplier: {noise_multiplier}\n, Max Physical Batch Size: {max_physical_batch_size}\n')

    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        if dataset == 'chexpert_tensors':
            train_loss, train_acc, train_val_auc_mean = CheXpert_train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = CheXpert_test(model, test_loader, ema)
        elif dataset == 'eyepacs_tensors':
            train_loss, train_acc, train_val_auc_mean = train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = test(model, test_loader, ema)

        if noise_multiplier > 0:
            
            print(f"privacy engine: {privacy_engine.accountant.history}")
            epsilon_calculated = privacy_engine.get_epsilon(delta=DELTA)
            print(f"ε = {epsilon_calculated:.2f}")

            if epsilon is not None and epsilon_calculated >= epsilon:
                return
        else:
            epsilon_calculated = None

        log_file.write(f'Train set: Average loss: {train_loss}, Accuracy: {train_acc}, AUC: {train_val_auc_mean}\n')
        log_file.write(f'Test set: Average loss: {test_loss}, Accuracy: {test_acc}, AUC: {test_val_auc_mean}\n')
        log_file.write(f'EMA Test set: Average loss: {ema_test_loss}, Accuracy: {ema_test_acc}, AUC: {ema_test_val_auc_mean}\n')
        log_file.write(f'Epsilon without considering DataNorm: {epsilon_calculated}\n')

    if norm_flag == "DataNorm":  # add DataNorm privacy step for DataNorm
        privacy_engine.accountant.step(noise_multiplier=data_norm_sigma, sample_rate=1)
        privacy_engine.accountant.step(noise_multiplier=data_norm_sigma, sample_rate=1)
        log_file.write(f'Final Epsilon Value with DataNorm: {privacy_engine.get_epsilon(delta=DELTA)}\n')

def train_scatternet(params):
    reps = params['reps']
    model = params['model']
    privacy = params['privacy']

    params['grad_sample_mode'] = "no_op" if params['aug_multiplicity'] else None
    params['layer'] = params['model']

    del params['baseline']
    del params['reps']
    del params['model']

    # create ScatterNet tensors
    store_scatter_features(**params)    # features are now stored at ./data/dataset/. Remove this if you already have the tensors stored.

    # point to ScatterNet tensors
    params['dataset'] = f"{params['dataset']}_tensors"

    for rep in range(reps):
        if privacy:
            print(f"Running private {model}")
            private_main(**params)
        else:
            print(f"Running non-private {model}")
            main(**params)
    params['log_file'].close()
    params['result_file_csv'].close()