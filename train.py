# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import torch
import torch.nn as nn


def train(loader, model, optimizer, epoch, cuda, log_interval, weight=None, verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (_, data, target) in enumerate(loader):
        criterion = nn.CrossEntropyLoss(weight=weight)
        if cuda:
            data, target = data.cuda(), target.cuda()
            criterion = criterion.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data.item()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.data.item()))
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, verbose=True, data_set='Test', save=None):
    model.eval()
    test_loss = 0
    correct = 0

    if save is not None:
        csv = open(save, 'wt')
        print('fname,label', file=csv)

    with torch.no_grad():
        for keys, data, target in loader:
            criterion = nn.CrossEntropyLoss(reduction='sum')
            if cuda:
                data, target = data.cuda(), target.cuda()
                criterion = criterion.cuda()
            output = model(data)
            test_loss += criterion(output, target).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if save is not None:
                for i, key in enumerate(keys):
                    print(key+'.wav,'+loader.dataset.classes[int(pred[i])], file=csv)

    test_loss /= len(loader.dataset)
    accuracy = float(correct) / len(loader.dataset)
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            data_set, test_loss, correct, len(loader.dataset), 100 * accuracy))
    return test_loss, accuracy
