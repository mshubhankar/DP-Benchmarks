import torch

RESULTS_DIR='results'
DATA_DIR='data'

use_cuda = torch.cuda.is_available()
# assert use_cuda
device = torch.device("cuda" if use_cuda else "cpu")

params = {
    'dataset' : 'cifar10',  # dataset name. Options: cifar10, chexpert, eyepacs
    'mini_batch_size' : 64, # mini batch size for dpsgd
    'aug_multiplicity' : True, # if augmentation multiplicity turned on
    'n_augs' : 8, # number of augmentations
    'baseline' : 'clip_g14', # baseline model. Options: clip_g14, clip_g16, wrn, scatternet
    'model' : 'lr', # model. Options: lr, cnn, tlnn
    'epochs' : 20, # number of epochs
    'privacy' : True, # if privacy turned on
    'epsilon' : 1.0, # epsilon for privacy
    'clip_norm' : 1.0, # clip norm for privacy
    'reps' : 3, # number of repetitions for each experiment
    'ema_flag' : False, # if exponential parameter averaging turned on
    'norm_flag' : 'GroupNorm', # None, GroupNorm, DataNorm
    'group_norm_groups' : 8, # number of groups for GroupNorm
    'data_norm_sigma' : 8.0, # sigma for DataNorm
}
    


variable_parameter_dicts = {

}