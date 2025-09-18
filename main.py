import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader import load_data
from loss import Loss, DECLoss
from metric import valid, H_valid, evaluate
from network import Network
from train import pretrain, contrastive_train
from utils import save_results

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='MNIST-USPS',
                    choices=['MSRCv1', 'MNIST-USPS', 'Hand', 'COIL20', 'Scene15',
                             'Fashion', 'BDGP', 'Hdigit', 'Caltech-5V', 'CCV',
                             'Synthetic3d', 'Prokaryotic', 'Cifar10', 'Cifar100'])
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=100)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--lamda1", default=0.01)
parser.add_argument("--lamda2", default=0.6)
parser.add_argument("--lamda3", default=0.05)
parser.add_argument("--lamda4", default=0.1)
parser.add_argument("--lamda5", default=1)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "MNIST-USPS":
    args.mse_epochs = 150
    seed = 10
if args.dataset == "Hdigit":
    args.batch_size = 128
    args.mse_epochs = 150
    args.con_epochs = 100
    seed = 10
if args.dataset == "BDGP":
    args.mse_epochs = 100
    seed = 10
if args.dataset == "Fashion":
    args.feature_dim = 64
    args.high_feature_dim = 16
    args.mse_epochs = 100
    seed = 10

if args.dataset == "Caltech-2V":
    args.batch_size = 128
    args.feature_dim = 256
    args.high_feature_dim = 64
    args.mse_epochs = 200
    seed = 10

if args.dataset == "Caltech-3V":
    args.batch_size = 128
    args.high_feature_dim = 64
    args.mse_epochs = 100
    seed = 10

if args.dataset == "Caltech-4V":
    args.batch_size = 128
    args.high_feature_dim = 64
    args.mse_epochs = 120
    seed = 10

if args.dataset == "Caltech-5V":
    args.batch_size = 128
    args.feature_dim = 256
    args.high_feature_dim = 64
    args.mse_epochs = 140
    args.con_epochs = 100
    seed = 5

if args.dataset == "Cifar10":
    args.mse_epochs = 120
    args.con_epochs = 15
    seed = 10

if args.dataset == "Cifar100":
    args.mse_epochs = 150
    args.con_epochs = 15
    seed = 10

if args.dataset == "Prokaryotic":
    args.mse_epochs = 150
    args.con_epochs = 10
    seed = 10

if args.dataset == "Synthetic3d":
    args.feature_dim = 64
    args.high_feature_dim = 16
    args.mse_epochs = 100
    args.con_epochs = 100
    seed = 100


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


print("==========\nArgs:{}\n==========".format(args))
setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
if not os.path.exists('./models'):
    os.makedirs('./models')

setup_seed(seed)

model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, data_size, device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
kl_loss = DECLoss().to(device)

best_acc, best_nmi, best_ari, best_pur, best_epoch = 0, 0, 0, 0, 0
Loss, ACC, NMI, PUR = [], [], [], []
train_time = 0

pretrain_time = pretrain(model, data_loader, device, view, args.mse_epochs, optimizer)

for epoch in range(0, args.con_epochs + 1):
    loss, time = contrastive_train(epoch, model, optimizer, criterion, kl_loss, device, data_loader, view, args)
    train_time += time

    if epoch >= 0 or epoch == args.con_epochs:
        acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num)

        Loss.append(loss)
        ACC.append(acc)
        NMI.append(nmi)
        PUR.append(pur)

        if acc > best_acc:
            best_acc, best_nmi, best_ari, best_pur, best_epoch = acc, nmi, ari, pur, epoch
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')

Loss_cpu = [loss.cpu().detach().numpy() for loss in Loss]
ACC_cpu = [acc for acc in ACC]
NMI_cpu = [nmi for nmi in NMI]
PUR_cpu = [pur for pur in PUR]

print('******** End, training time = {} s ********'.format(round(train_time + pretrain_time, 2)))

save_results(args, best_acc, best_nmi, best_ari, best_pur)
