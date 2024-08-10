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
import open_clip
from torchvision import transforms
from tqdm import tqdm
import utils


class get_feature(Dataset):
    def __init__(self, params, feature_dir, aug=False, mode='train'):
        self.root_dir = feature_dir
        self.mode = mode
        if aug:
            self.aug = True
            dataset = params['dataset']
            augk = params['n_augs']
        else:
            self.aug = False
        
        if self.aug:
            x_train_path = os.path.join(self.root_dir, f'{params["dataset"]}_aug_{augk}_train_X.pt')
            y_train_path = os.path.join(self.root_dir, f'{params["dataset"]}_aug_{augk}_train_Y.pt')
        else:
            x_train_path = os.path.join(self.root_dir, f'{params["dataset"]}_train_X.pt')
            y_train_path = os.path.join(self.root_dir, f'{params["dataset"]}_train_Y.pt')
        x_test_path = os.path.join(self.root_dir, f'{params["dataset"]}_test_X.pt')
        y_test_path = os.path.join(self.root_dir, f'{params["dataset"]}_test_Y.pt')

        # make features if not already made
        if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
            print('Making features')
            utils.make_features(params, self.root_dir)

        if mode == 'test':
            self.X = torch.load(x_test_path)
            self.Y = torch.load(y_test_path)
        else:
            self.X = torch.load(x_train_path)
            self.Y = torch.load(y_train_path) 
                
        # convert to one hot encoding
        if len(self.Y.shape) == 1:
            self.Y = torch.nn.functional.one_hot(self.Y.squeeze(), num_classes=torch.unique(self.Y).shape[0])
        self.Y = self.Y.float()
        self.length = self.X.shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def train_clip(params):
    
    clip_features = config.DATA_DIR + '/clip_features'
    if not os.path.exists(clip_features):
        os.makedirs(clip_features)
    
    data_feature_path = clip_features + f'/{params["baseline"]}'
    if not os.path.exists(data_feature_path):
        os.makedirs(data_feature_path)
    
    train_set = get_feature(params, feature_dir=data_feature_path, aug=params['aug_multiplicity'], mode='train') 
    test_set = get_feature(params, feature_dir=data_feature_path,  mode='test')
    target_delta = float(f'1e-{len(str(len(train_set)))}')
    if 'chexpert' in params['dataset']:
            multi_label = True
            avg_acc = torch.zeros(params['output_dim']).cpu()
            avg_train_acc = torch.zeros(params['output_dim']).cpu()
    else:
        multi_label = False
        avg_acc = 0
        avg_train_acc = 0

    if 'g14' in params['baseline']:
        data_features = 1280
    elif 'b16' in params['baseline']:
        data_features = 512


    avg_auc = []
    avg_train_auc = []
    
    params['log_file'].write(f'Minibatch size: {{params["minibatch_size"]}}\nData feature: {{data_features}}\n\
                                 Output dim: {{params["output_dim"]}}\nClip norm: {{params["clip_norm"]}}\n\
                                 EMA: {{params["ema_flag"]}}\n Norm: {{params["norm_flag"]}}\n')
    params['log_file'].write(f'Epochs: {{params["epochs"]}} Privacy: {{params["privacy"]}} Epsilon: {{params["epsilon"]}}\
                                  Reps: {{params["reps"]}}\n')


    
    norm_stats = None
    if params['norm_flag'] == "DataNorm":
        save_dir = f"norm_stats/norm_stats_{params['baseline']}/{params['dataset']}"
        os.makedirs(save_dir, exist_ok=True)
        norm_loader = DataLoader(train_set, batch_size= params['minibatch_size'], shuffle=True)
        norm_stats, rdp = data_normalization(norm_loader, False, data_features, params['device'],\
                                              len(train_set), len(train_set), params['data_norm_sigma'], save_dir=save_dir)


    for rep in range(params['reps']):
        print(f'Repetition {rep+1}/{params["reps"]}')

        train_loader = DataLoader(train_set, batch_size=params['minibatch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        
        if params['aug_multiplicity']:
            train_set_without_aug = get_feature(params, feature_dir=data_feature_path, aug=False, mode='train')
            full_train_loader = DataLoader(train_set_without_aug, batch_size=params['minibatch_size'], shuffle=False)
        else:
            full_train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
        
        if params['model'] == 'lr':
            # train the logistic regression model using torch
            model = LogisticRegresion(data_features, params['output_dim'], norm=params['norm_flag'], norm_stats=norm_stats,\
                                        num_groups=params['group_norm_groups'])

        if params['model'] == 'tlnn':
            # train the logistic regression model using torch
            model = TwoLayer(data_features, params['output_dim'], norm=params['norm_flag'], norm_stats=norm_stats)
            
        if params['model'] == 'cnn':
            model = CNN(input_norm=params['norm_flag'], weight_standardization=False)
        # torch.nn.init.zeros_(model.linear.weight)
        
        model = model.to(params['device'])
        
        # criterion = CrossEntropyLoss_oh
        if params['dataset'].startswith('eyepacs') or params['dataset'].startswith('cifar'):
            criterion = torch.nn.CrossEntropyLoss()
        if params['dataset'].startswith('chexpert'):
            criterion = utils.CrossEntropyLossChexpert()

        if params['ema_flag']:
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hook_style = "hooks"
        if params['aug_multiplicity']:
            hook_style = "no_op"
        if params['privacy']:
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=params['epochs'],
                target_epsilon=params['epsilon'],
                target_delta=target_delta,
                max_grad_norm=params['clip_norm'],
                grad_sample_mode=hook_style,
                clipping="flat"
            )
        model.train()
        
        
        
        if params['aug_multiplicity'] :
            fmodel, _fparams = make_functional(model)
                            
            def compute_loss_stateless_model(m_params, sample, target):
                target = target.repeat(params['n_augs'], 1)
                predictions = fmodel(m_params, sample)
                loss = criterion(predictions, target)
                return loss

            ft_compute_grad = grad_and_value(compute_loss_stateless_model)
            ft_compute_sample_grad = vmap(ft_compute_grad, in_dims = (None, 0, 0))
            # Using model.parameters() instead of fparams
            # as fparams seems to not point to the dynamically updated parameters
            model_params = list(model.parameters())
            


        for epoch in range(params['epochs']):
            train_pred = []
            train_true = []
            train_loss = 0
            for i, (X, Y) in enumerate(train_loader):
                
                X, Y = X.to(params['device']), Y.to(params['device'])
                if params['aug_multiplicity']:

                    per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                        model_params, X, Y
                    )
                    per_sample_grads = [g.detach() for g in per_sample_grads]
                    
                    loss = torch.mean(per_sample_losses)
                    for p, g in zip(model_params, per_sample_grads):
                        p.grad_sample = g
                    
                else:
                    output = model(X)
                    loss = criterion(output, Y)
                    loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if params['ema_flag']:
                    ema.update()
                
                train_loss += loss.item()
            print(f"Epoch {epoch}/{params['epochs']}: Train loss: {train_loss/len(train_set)}", end=' ')
            del X, Y
            del train_pred, train_true

            if params['privacy']:
                epsilon = privacy_engine.get_epsilon(target_delta)
                print(f'Epsilon after epoch {epoch}: {epsilon}')


        # test the model
        with torch.no_grad():
            train_pred = []
            train_true = []
            train_pred = []
            for i, (X, Y) in enumerate(full_train_loader):
                X, Y = X.to(params['device']), Y.to(params['device'])

                model = model.to(params['device'])
                if params['ema_flag']:
                    with ema.average_parameters():
                        output = model(X)
                else:
                    output = model(X)
                loss = criterion(output, Y)
                train_pred.append(output)
                train_true.append(Y)
                del X, Y
            train_pred = torch.cat(train_pred)
            train_true = torch.cat(train_true)
            print('Train metrics:')
            acc, auc = utils.print_metrics(train_pred, train_true, multi_label)
            print('Train loss:', loss.item())
            avg_train_acc += acc
            avg_train_auc.append(auc)
            params['log_file'].write(f'Rep {rep}: Train ACC- {acc} Train AUC- {auc}\n')
            
            test_pred = []
            test_true = []
            for i, (X, Y) in enumerate(test_loader):
                X, Y = X.to(params['device']), Y.to(params['device'])
                model = model.to(params['device'])
                
                if params['ema_flag']:
                    with ema.average_parameters():
                        output = model(X)
                else:
                    output = model(X)
                
                loss = criterion(output, Y)
                test_pred.append(output)
                test_true.append(Y)
                del X, Y
            test_pred = torch.cat(test_pred)
            test_true = torch.cat(test_true)
            print('Test metrics:')
            acc, auc = utils.print_metrics(test_pred, test_true, multi_label)

            print('Test loss:', loss.item())
            avg_acc += acc
            avg_auc.append(auc)
            params['log_file'].write(f'Rep {rep}: Test ACC- {acc} Test AUC- {auc}\n')
            del test_pred, test_true
        
    avg_acc /= params['reps']
    std_dev_auc = torch.tensor(avg_auc).std()
    params['log_file'].write(f'Average Test ACC- {sum(avg_auc) / len(avg_auc)} AUC- {avg_auc}\n')
    print(f'Average Test ACC- {avg_acc} AUC- {sum(avg_auc) / len(avg_auc)}\n')

    avg_train_acc /= params['reps']
    std_train_auc = torch.tensor(avg_train_auc).std()
    params['log_file'].write(f'Average Train ACC- {sum(avg_train_auc) / len(avg_train_auc)} AUC- {avg_train_auc}\n')
    print(f'Average Train ACC- {avg_train_acc} AUC- {sum(avg_train_auc) / len(avg_train_auc)}\n')
    if params['privacy']:
        params['log_file'].write(f'Noise multiplier: {optimizer.noise_multiplier}\n\n')

    if params['privacy']:
        params['result_file_csv'].write(f"{params['dataset']},{params['epochs']},{params['privacy']},{epsilon},{target_delta},{params['clip_norm']},Adam,0.001,False,{params['ema_flag']},{params['minibatch_size']},{params['norm_flag']},{sum(avg_train_auc) / len(avg_train_auc)},{std_train_auc},{sum(avg_auc) / len(avg_auc)},{std_dev_auc},{avg_acc}\n")
    else:
        params['result_file_csv'].write(f"{params['dataset']},{params['epochs']},{params['privacy']},None,None,{params['clip_norm']},Adam,0.001,False,{params['ema_flag']},{params['minibatch_size']},{params['norm_flag']},{sum(avg_train_auc) / len(avg_train_auc)},{std_train_auc},{sum(avg_auc) / len(avg_auc)},{std_dev_auc},{avg_acc}\n")
    params['log_file'].close()
    params['result_file_csv'].close()
