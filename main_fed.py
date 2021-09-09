#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, GlobalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, AlexNetMnist, AlexNetCifar
from models.Fed import FedAvg, MeanTeacher
from models.test import test_img
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.mean_teacher)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # # split labeled data and unlabeled data
        # X_Y = DataLoader(dataset_train)

        X = dataset_train.data.to('cpu').numpy() # covert X, Y from torch to numpy
        Y = dataset_train.targets.to('cpu').numpy()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=3)
        x_train, x_unlabel, y_train, y_unlabel = train_test_split(x_train, y_train, test_size=0.8624, random_state=3)
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.5, random_state=3)
        assert len(x_train) == len(y_train)
        assert len(x_unlabel) == len(y_unlabel)

        dataset_unlabel = (x_unlabel, y_unlabel)
        dataset_test = (x_test, y_test)
        dataset_server = (x_train, y_train)


        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_unlabel, args.num_users)
            # print(len(dataset_unlabel))
        else:
            dict_users = mnist_noniid(dataset_unlabel, args.num_users)

    elif args.dataset == 'FashionMnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/FashionMnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/FashionMnist/', train=False, download=True, transform=trans_mnist)

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'alexnet' and args.dataset == 'cifar':
        net_glob = AlexNetCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'alexnet' and args.dataset == 'mnist':
        net_glob = AlexNetMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'FashionMnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'alexnet' and args.dataset == 'FashionMnist':
        net_glob = AlexNetMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    user_data = []
    user_target = []
    for i in range(args.num_users):
        image = dataset_unlabel[0][dict_users[i]]
        label = dataset_unlabel[1][dict_users[i]]
        user_data.append(image)
        user_target.append(label)
    w_glob = net_glob.state_dict()
    w_save = copy.deepcopy(w_glob)

    # training
    loss_train_list = []
    acc_train_list = []
    loss_test_list = []
    acc_test_list = []

    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    num_params = 0
    for param in net_glob.parameters():
        num_params += param.numel()
    print(num_params)
    
    # breakpoint()
    w_teacher_total = [] # For recording historical teacher's model parameters

    for iter in range(args.epochs):

        # train global model

        w_locals, loss_locals = [], []
        globe = GlobalUpdate(args=args, dataset=dataset_server[0],
                                target=dataset_server[1])
        w, loss = globe.train(net=copy.deepcopy(net_glob).to(args.device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))

        w_locals.append(copy.deepcopy(w_glob))

        # train local models
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=user_data[idx], target=user_target[idx], tao=0.99, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)


        # whether mean_teacher or not
        if args.mean_teacher == True:
            if iter != 0:
                w_teacher = MeanTeacher(w_teacher_total[iter-1], w_glob)
            else:
                w_teacher = copy.deepcopy(w_glob)

            # append w in historical list
            w_teacher_total.append(copy.deepcopy(w_teacher))

            # copy weight to net_glob
            net_glob.load_state_dict(w_teacher)

        else:
            net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_unlabel, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        loss_test_list.append(loss_test)
        acc_test_list.append(acc_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print()


    count = np.array([_ for _ in range(0,args.epochs)])
    plt.plot(count, loss_train_list, label='train loss')
    plt.plot(count, loss_test_list, label='test loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.savefig('val.pdf')
    plt.show()


