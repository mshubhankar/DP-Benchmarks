import torch
import os
from PIL import Image
import open_clip
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
from tqdm import tqdm
from sys import getsizeof
import numpy as np
from torchmetrics.functional.classification import multilabel_accuracy
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from data import get_data, get_scatter_transform, get_scattered_loader


class CrossEntropyLossChexpert(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLossChexpert, self).__init__()
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true, reduction='mean'):  # TODO: handle the tensor shapes
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        return self.criterion(y_pred, y_true, reduction=reduction)
    

def print_metrics(pred, true, multi_label=False):
    if multi_label:
        acc = multilabel_accuracy(pred, true, num_labels=true.size(dim=1), average=None).cpu()
        auc = roc_auc_score(true.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted', multi_class='ovr')
        print(f'Acc: {acc} AUC: {auc}')
    else:
        pred_prob = torch.nn.functional.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
        acc_label = torch.argmax(true, dim=1)
        acc = torch.sum(pred_label == acc_label).item() / len(true)
        auc = roc_auc_score(true.cpu().detach().numpy(), pred_prob.cpu().detach().numpy(), average='weighted', multi_class='ovr')            
        print(f'Acc: {acc} AUC: {auc}')
    return acc, auc



def make_features(params, feature_path):
    if 'g_14' in params['baseline']:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    elif 'b_16' in params['baseline']:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-16')

    if params['dataset'] == 'chexpert':
        text = tokenizer(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']) #chexpert
    if params['dataset'] == 'eyepacs_complete':
        text = tokenizer(['No Diabetic Retinopathy', 'Mild Diabetic Retinopathy', 'Moderate Diabetic Retinopathy', 'Severe Diabetic Retinopathy', 'Proliferate Diabetic Retinopathy']) #eyepacs
    if params['dataset'] == 'cifar10':
        text = tokenizer(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']) #cifar10

    train_data, test_data = get_data(params['dataset'], preprocess=preprocess, augment=False)
    
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=params['mini_batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=params['mini_batch_size'], shuffle=False, num_workers=1, pin_memory=True)


    if params['aug_multiplicity']:
        aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect')])
    
    tmp_path = '/tmp/'
    with torch.no_grad():
        
        for i, rows in enumerate(tqdm(train_loader)):

            data, target = rows
            data, target = data.to(params['device']), target.to(params['device'])
            model = model.to(params['device'])
            # text = text.to(params['device'])
            image_features, text_features, _ = model(data, text)
            
            if params['aug_multiplicity']:
                for j in range(params['n_augs']-1):
                    data_aug = aug(data)
                    with torch.no_grad():
                        image_features_aug, _, _ = model(data_aug, text)
                    
                    # append image_features_aug to image_features so that [64,512] becomes [64,2,512]
                    if len(image_features.shape) == 2:
                        image_features = image_features.unsqueeze(1)
                    image_features_aug = image_features_aug.unsqueeze(1)
                    image_features = torch.cat((image_features, image_features_aug), dim=1)
                    
                    del data_aug, image_features_aug
            
                
            del data
            

            #store output in a numpy file
            if params['aug_multiplicity']:
            
                torch.save(image_features.cpu().detach(), f'{tmp_path}/{params['dataset']}_aug_{params['n_augs']}_train_X_part{i}.pt')
                torch.save( target.cpu().detach(), f'{tmp_path}/{params['dataset']}_aug_{params['n_augs']}_train_Y_part{i}.pt')
            else:
                torch.save(image_features.cpu().detach(), f'{tmp_path}/{params['dataset']}_train_X_part{i}.pt')
                torch.save( target.cpu().detach(), f'{tmp_path}/{params['dataset']}_train_Y_part{i}.pt')


        # if test_path does not exist
        if not os.path.exists(f'{feature_path}/{params['dataset']}_test_X.pt'):
            for i, rows in enumerate(tqdm(test_loader)):
                data, target = rows
                data, target = data.to(params['device']), target.to(params['device'])
                model = model.to(params['device'])
                text = text.to(params['device'])
                image_features, text_features, _ = model(data, text)

                
                #store output in a numpy file
                torch.save(image_features.cpu().detach(), f'{tmp_path}/{params['dataset']}_test_X_part{i}.pt')
                torch.save( target.cpu().detach(), f'{tmp_path}/{params['dataset']}_test_Y_part{i}.pt')

    print('Combining files')
    full_X = []
    full_Y = []
    # number of files that start with dataset_train_X_part
    n_train_files = len([name for name in os.listdir(tmp_path) if name.startswith(f'{params['dataset']}_train_X_part')])
    n_test_files = len([name for name in os.listdir(tmp_path) if name.startswith(f'{params['dataset']}_test_X_part')])

    for i in tqdm(range(n_train_files)):
        X = torch.load(tmp_path + f'{params['dataset']}_train_X_part{i}.pt')
        Y = torch.load(tmp_path + f'{params['dataset']}_train_Y_part{i}.pt')
        full_X.append(X)
        full_Y.append(Y)
    full_X = torch.cat(full_X)
    full_Y = torch.cat(full_Y)

    torch.save(full_X, feature_path + f'{params['dataset']}_train_X.pt')
    torch.save(full_Y, feature_path + f'{params['dataset']}_train_Y.pt')

    # if test_path does not exist
    if not os.path.exists(f'{feature_path}/{params['dataset']}_test_X.pt'):
        full_X = []
        full_Y = []
        for i in tqdm(range(n_test_files)):
            X = torch.load(tmp_path + f'{params['dataset']}_test_X_part{i}.pt')
            Y = torch.load(tmp_path + f'{params['dataset']}_test_Y_part{i}.pt')
            full_X.append(X)
            full_Y.append(Y)
        full_X = torch.cat(full_X)
        full_Y = torch.cat(full_Y)

        torch.save(full_X, feature_path + f'{params['dataset']}_test_X.pt')
        torch.save(full_Y, feature_path + f'{params['dataset']}_test_Y.pt')
