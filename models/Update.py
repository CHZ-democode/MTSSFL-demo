#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from consistency import softmax_mse_loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def local_update(args, net, data, target, tao=0.99):
    traindata = TensorDataset(data.clone().detach(), target.clone().detach().long())
    ldr_train = DataLoader(traindata, batch_size=args.local_bs, shuffle=True)

    net.train()
    # train and update
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            p_out, log_probs = net(images)
            pseudo_label = torch.softmax(p_out.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(tao).float()
            labels = targets_u
            loss = -(F.nll_loss(p_out, labels, reduction='none') * mask).mean()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(images), len(ldr_train.dataset),
                          100. * batch_idx / len(ldr_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def global_update(args, net, data, target):
    traindata = TensorDataset(data.clone().detach(), target.clone().detach().long())
    ldr_train = DataLoader(traindata, batch_size=256, shuffle=True)

    net.train()
    # train and update
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    epoch_loss = []
    for iter in range(100): # range default is 30
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            p_out, output = net(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            if args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(images), len(ldr_train.dataset),
                          100. * batch_idx / len(ldr_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def MT_update(args, net_t, net, data, target , tao=0.99):
    traindata = TensorDataset(data.clone().detach(), target.clone().detach().long())
    ldr_train = DataLoader(traindata, batch_size=args.local_bs, shuffle=True)
    consistency_criterion = softmax_mse_loss

    net.train()
    net_t.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            p_out, log_probs = net(images)
            pseudo_label = torch.softmax(p_out.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(tao).float()
            labels = targets_u
            logit_loss = -(F.nll_loss(p_out, labels, reduction='none') * mask).mean()
            #             print('loss', loss)
            p_out_t, _ = net_t(images)
            consistency_weight = 3
            consistency_loss = consistency_weight * consistency_criterion(p_out, p_out_t) / args.local_bs
            loss = logit_loss + args.consistency_rate * consistency_loss


            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(images), len(ldr_train.dataset),
                          100. * batch_idx / len(ldr_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

