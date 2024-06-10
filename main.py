import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchmetrics.functional.classification import multilabel_accuracy
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, confusion_matrix
from libauc.losses import AUCMLoss, CrossEntropyLoss
from opacus import PrivacyEngine
from data import get_data
from numpy.linalg import norm
from torch_ema import ExponentialMovingAverage
from data_normalization import data_normalization, standardize
from opacus.grad_sample.functorch import make_functional
from torch.func import grad, grad_and_value, vmap
from opacus.grad_sample import GradSampleModule
from models import LogisticRegresion, CNN, TwoLayer
import config   
import time
params = config.params
if params['dataset'].startswith('chexpert') or params['dataset'].startswith('eyepacs'):
    params['output_dim'] = 5
elif params['dataset'].startswith('cifar10'):
    params['output_dim'] = 10
else:
    raise ValueError('dataset not implemented')

result_dir = config.RESULTS_DIR
time_stamp_now = time.strftime("%Y%m%d-%H%M%S")

result_file_excel = open(f'result_dir/{time_stamp_now}_{params['dataset']}_{params['n_augs']}_{params['model']}_epochs{params['epochs']}_\
                         privacy{params['privacy']}_reps{params['reps']}_ema{params['ema_flag']}_norm{params['norm_flag']}\
                            {params['group_norm_groups']}_batchsize{params['minibatch_size']}_excel.csv', 'w')
log_file = open(f'result_dir/{time_stamp_now}_{params['dataset']}_{params['n_augs']}_{params['model']}_epochs{params['epochs']}_\
                         privacy{params['privacy']}_reps{params['reps']}_ema{params['ema_flag']}_norm{params['norm_flag']}\
                            {params['group_norm_groups']}_batchsize{params['minibatch_size']}_excel.txt', 'w')

params['result_file_excel'] = result_file_excel
params['log_file'] = log_file

result_file_excel.write('name,epochs,privacy,epsilon,delta,max_grad_norm,optim,LR,WS,EMA,BatchSize,input_norm,train_auc,train_auc_std_test_auc,test_auc_std, test_acc\n')