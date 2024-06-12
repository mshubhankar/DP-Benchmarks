import torch

RESULTS_DIR='results'
DATA_DIR='data'

use_cuda = torch.cuda.is_available()
# assert use_cuda
device = torch.device("cuda" if use_cuda else "cpu")

params = {
    'dataset' : 'chexpert',  # dataset name. Options: cifar10, chexpert, eyepacs_complete
    'minibatch_size' : 64, # mini batch size for dpsgd
    'max_physical_batch_size': 16,
    'aug_multiplicity' : False, # if augmentation multiplicity turned on
    'n_augs' : 8, # number of augmentations
    'baseline' : 'wrn', # baseline model. Options: clip_g14, clip_g16, wrn, scatternet
    'mode' : 'full', #wrn training mode. Options: 'full', 'final', 'scratch'
    'model' : 'lr', # model. Options: lr, cnn, tlnn
    'epochs' : 3, # number of epochs
    'lr' : 1e-3,
    'privacy' : True, # if privacy turned on
    'epsilon' : 1.0, # epsilon for privacy
    'clip_norm' : 1.0, # clip norm for privacy
    'reps' : 3, # number of repetitions for each experiment
    'ema_flag' : False, # if exponential parameter averaging turned on
    'norm_flag' : 'GroupNorm', # None, GroupNorm, DataNorm
    'group_norm_groups' : 8, # number of groups for GroupNorm
    'data_norm_sigma' : 8.0, # sigma for DataNorm
    'device' : device, # device
}
    


variable_parameters_dict = {
    'dataset' : ['cifar10', 'chexpert', 'eyepacs_complete'],
    'mode' : ['full', 'final', 'scratch'],
}
