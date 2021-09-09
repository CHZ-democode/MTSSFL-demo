#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021-03-03 16:27
# @Author  : Wizard Chenhan Zhang
# @FileName: consistency.py
# @Software: PyCharm

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

# def consistency_update(args, teacher_net, student_net, data, target):
#     traindata = TensorDataset(data.clone().detach(), target.clone().detach().long())
#     ldr_train = DataLoader(traindata, batch_size=256, shuffle=True)
#
#     teacher_net.train()
#     student_net.train()

