#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from utils.options import args_parser


args = args_parser()


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def MeanTeacher(previous, w_glob):
    for k in w_glob.keys():
        w_glob[k] = args.alpha*previous[k] + (1-args.alpha)*w_glob[k]
    w_MT = w_glob
    return w_MT
