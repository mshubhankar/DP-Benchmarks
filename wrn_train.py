import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import opacus
from data import get_data
from wrn_model import WideResNet
import math
import config

from sklearn import metrics
# from sklearn.metrics import roc_auc_score
from torchmetrics.functional.classification import multilabel_accuracy

import warnings
warnings.filterwarnings("ignore") 

params = config.params

MAX_GRAD_NORM = params['clip_norm']
train_data, test_data = get_data(params['dataset'], augment=False)
EPSILON = params['epsilon']
DELTA = 10 ** -(math.ceil(math.log10(len(train_data))))
EPOCHS = params['epochs']
LR = params['lr']

mode = params['mode']

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=params['minibatch_size'], shuffle=False, num_workers=1, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=params['minibatch_size'], shuffle=False, num_workers=1, pin_memory=True)


checkpoint = torch.load("WRN_28_10_model_best_1219.pth")
model = WideResNet(num_classes=1000, depth=28, width=10)
model = nn.DataParallel(model)

if mode != 'scratch':
    model.load_state_dict(checkpoint['state_dict'])

num_classes = 5

# Modify the final fully connected layer according to the number of classes
num_features = 64*model.module.width
model.module.Softmax = nn.Linear(num_features, num_classes)

if mode == 'final':
    for name, param in model.named_parameters():  
        if not 'Softmax' in name:
            param.requires_grad = False


from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)

if errors == []:
    print("Model valid!")
else:
    print("The following errors were found in the model:")
    print(errors)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)            

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true, reduction='mean'):
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        return self.criterion(y_pred, y_true, reduction=reduction)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.module.parameters(), lr=LR)

if params['dataset'] == 'chexpert':
    criterion = CrossEntropyLoss()

from opacus import PrivacyEngine
privacy_engine = PrivacyEngine(accountant='prv')

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs = EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager

def get_auc(pred, true, multi_label=False):
    if multi_label:
        auc = metrics.roc_auc_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted', multi_class='ovr')
        
    else:
        auc = metrics.roc_auc_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted', multi_class='ovr')
   
    return auc

def accuracy(preds, labels):
    return (preds == labels).mean()

def eye_train(model, train_loader, optimizer, epoch, device):
    print("In eye_train")
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    train_pred_cpu = []
    train_true_cpu = []

    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=params['max_physical_batch_size'], 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   

            if i>40:
                break

            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            
            loss = criterion(output, target)

            m = nn.Softmax(dim=1)
            output_ = m(output)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            
            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

            # calculate AUC
            train_pred_cpu.append(output_.cpu().detach().numpy())
            train_true_cpu.append(target.cpu().detach().numpy())

    train_pred_cpu = np.concatenate(train_pred_cpu)
    train_true_cpu = np.concatenate(train_true_cpu)
    auc = metrics.roc_auc_score(train_true_cpu, train_pred_cpu, average='weighted', multi_class='ovr')

    print("LOOPS: ", i, " AUC: ", auc)

def chex_train(model, train_loader, optimizer, EPOCHS, device):
    print("In chex_train")

    device = next(model.parameters()).device
    model.train()
    train_loss = 0
    num_examples = 0
    losses = []

    criterion = CrossEntropyLoss()
    train_pred = []
    train_true = []

    train_pred_cpu = []
    train_true_cpu = []

    print(f'Train Loader: {len(train_loader)}, {train_loader.batch_size}')


    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=params['max_physical_batch_size'],
            optimizer=optimizer
    ) as memory_safe_data_loader:
        for i, (data, target) in enumerate(memory_safe_data_loader):

            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)

            output = model(data)
            
            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()


            train_loss += criterion(output, target, reduction='sum').item()
            num_examples += len(data)

            # calculate multi-label accuracy
            train_pred.append(output)
            train_true.append(target)

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

            # Append to calculate AUC
            train_pred_cpu.append(output.cpu().detach().numpy())
            train_true_cpu.append(target.cpu().detach().numpy())


    train_loss /= num_examples
    
    train_pred = torch.cat(train_pred)
    train_true = torch.cat(train_true)
    multi_label_acc = multilabel_accuracy(train_pred, train_true, num_labels=train_true.size(dim=1), average=None).cpu()

    train_pred_cpu = np.concatenate(train_pred_cpu)
    train_true_cpu = np.concatenate(train_true_cpu)
    val_auc_mean = metrics.roc_auc_score(train_true_cpu, train_pred_cpu, average='weighted', multi_class='ovr')

    print("LOOPS: ", i ," ", f'Train set: Average loss: {train_loss:.4f}', f'Accuracy: {multi_label_acc}', f'AUC: {val_auc_mean:.4f}')

    return train_loss, multi_label_acc, val_auc_mean 


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    if params['dataset'] == 'chexpert':
        criterion = CrossEntropyLoss()

    losses = []
    top1_acc = []
    auc_list = []

    test_pred = []
    test_true = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            test_probs = torch.softmax(output, dim=1)

            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            test_pred.append(test_probs.detach().cpu().numpy())
            test_true.append(target.detach().cpu().numpy())

            losses.append(loss.item())
            top1_acc.append(acc)


    top1_avg = np.mean(top1_acc)
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc_mean = metrics.roc_auc_score(test_true, test_pred, average='weighted', multi_class='ovr')

    if params['dataset'] == 'chexpert':
        test_acc = multilabel_accuracy(torch.from_numpy(test_pred), torch.from_numpy(test_true), num_labels=torch.from_numpy(test_true).size(dim=1), average=None).cpu()
        print(
            f"\tTest set:   "
            f"Acc:  "
            f"{test_acc}"
            f"AUC: {val_auc_mean * 100:.6f} "
        )

        return test_acc


    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
        f"AUC: {val_auc_mean * 100:.6f} "
    )

    return np.mean(top1_acc)


def wrn_train(params, model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, device=device):
    EPOCHS = params['epochs']
    # import pdb; pdb.set_trace()


    print("Training...")
    for epoch in range(EPOCHS):
        if params['dataset'] == 'chexpert':
            chex_train(model, train_loader, optimizer, epoch + 1, device)
        elif params['dataset'] == 'eyepacs_complete':
            eye_train(model, train_loader, optimizer, epoch + 1, device)

    print("End of training. Testing...")
    test(model, test_loader, device)
