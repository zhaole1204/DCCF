import torch
import torch.nn as nn
from scipy.stats import entropy

from dataloader import *
from utils import get_W


def pretrain(model, data_loader, device, view, epochs, optimizer):
    t = time.time()
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        tot_loss = 0.
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            _, _, xrs, _, _, _, _ = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            print('pretrain: Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    print("Pre-training finished..................................................................")
    print("Total time elapsed: {:.4f}s\n".format(time.time() - t))
    return time.time() - t


def contrastive_train(epoch, model, optimizer, criterion, kl_loss, device, data_loader, view, args):
    tot_loss = 0.
    time0 = time.time()
    mes = torch.nn.MSELoss()

    for batch_idx, (xs, label, sample_idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        hs, qs, xrs, zs, H, H_pre, pred = model(xs)

        qs, preds = model.forward_cluster(xs)
        pre = sum(qs) / view

        contrastive_loss1 = 0.
        contrastive_loss2 = 0.
        kl_div_loss = 0.

        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        kl_div_loss += args.lamda3 * kl_criterion(torch.log(pre + 1e-9), H_pre)
        kl_div_loss += args.lamda1 * kl_loss(H_pre)

        for v in range(view):
            contrastive_loss1 += args.lamda3 * criterion.forward_feature(hs[v], H)
            contrastive_loss1 += args.lamda4 * criterion.forward_label(qs[v], H_pre)
            contrastive_loss1 += args.lamda1 * criterion.forward_prob(qs[v], H_pre)
            contrastive_loss1 += mes(xs[v], xrs[v])

        for v in range(view):
            for w in range(v + 1, view):
                contrastive_loss2 += args.lamda3 * criterion.forward_feature(hs[v], hs[w])
                contrastive_loss2 += args.lamda2 * criterion.forward_label(qs[v], qs[w])
                contrastive_loss2 += args.lamda1 * criterion.forward_prob(qs[v], qs[w])

        loss = contrastive_loss1 + contrastive_loss2 + kl_div_loss
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    epoch_time = time.time() - time0

    return loss/ len(data_loader), epoch_time
