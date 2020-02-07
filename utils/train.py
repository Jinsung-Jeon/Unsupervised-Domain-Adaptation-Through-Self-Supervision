# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:43:06 2020

@author: Jinsung
"""

import torch
from utils.get_mmd import get_mmd

def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    
    return 1 - correct/total