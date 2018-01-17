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

'''Neural Nets for Speech Commands Recognition'''

from __future__ import print_function

import argparse
import os

import torch
import torch.optim as optim

from gcommands_loader import Loader
from model import TDNN, VGG, LeNet
from train import test, train


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
    parser.add_argument('--train_path', default='data/train_training', help='path to the train data folder')
    parser.add_argument('--test_path', default='data/train_testing', help='path to the test data folder')
    parser.add_argument('--valid_path', default='data/train_validation', help='path to the valid data folder')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
    parser.add_argument('--arc', default='VGG16', help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--patience', type=int, default=5, metavar='N', help='how many epochs of no loss improvement should we wait before stop training')
    parser.add_argument('--checkpoint', default='checkpoint', metavar='CHECKPOINT', help='checkpoints directory')
    parser.add_argument('--no-train', dest='train', action='store_false')
    # feature extraction options
    parser.add_argument('--window_size', type=float, default=.02, help='window size for the stft')
    parser.add_argument('--window_stride', type=float, default=.01, help='window stride for the stft')
    parser.add_argument('--window_type', default='hamming', help='window type for the stft')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='do not not to normalize the spect')
    parser.add_argument('--num_workers', type=int, default=0, help='int, how many subprocesses to use for data loading')

    args = parser.parse_args()
    print(args)

    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))

    # loading data
    if args.train:
        train_dataset = Loader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
            window_type=args.window_type, normalize=args.normalize)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.cuda, sampler=None)

        valid_dataset = Loader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
            window_type=args.window_type, normalize=args.normalize)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=None,
            num_workers=args.num_workers, pin_memory=args.cuda, sampler=None)

        # build model
        if args.arc.startswith('VGG'):
            model = VGG(args.arc)
        elif args.arc == 'TDNN':
            model = TDNN()
        else:
            model = LeNet()

        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        # define optimizer
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        best_valid_acc = 0
        iteration = 0
        epoch = 1

        # trainint with early stopping
        while (epoch < args.epochs + 1) and (iteration < args.patience):
            train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval,
                weight=train_dataset.weight)
            valid_loss, valid_acc = test(valid_loader, model, args.cuda, data_set='Validation')
            if not os.path.isdir(args.checkpoint):
                os.mkdir(args.checkpoint)
            torch.save(model.module if args.cuda else model,
                './{}/model{:03d}.t7'.format(args.checkpoint, epoch))
            if valid_acc <= best_valid_acc:
                iteration += 1
                print('Accuracy was not improved, iteration {0}'.format(str(iteration)))
            else:
                print('Saving state')
                iteration = 0
                best_valid_acc = valid_acc
                state = {
                    'valid_acc': valid_acc,
                    'valid_loss': valid_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.checkpoint):
                    os.mkdir(args.checkpoint)
                torch.save(state, './{}/ckpt.t7'.format(args.checkpoint))
            epoch += 1


    # test model
    test_dataset = Loader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
        window_type=args.window_type, normalize=args.normalize)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=None,
        num_workers=args.num_workers, pin_memory=args.cuda, sampler=None)

    state = torch.load('./{}/ckpt.t7'.format(args.checkpoint))
    epoch = state['epoch']
    print("Testing model {} (epoch {})".format(args.checkpoint, epoch))
    model = torch.load('./{}/model{:03d}.t7'.format(args.checkpoint, epoch))
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    results = './{}/{}.csv'.format(args.checkpoint, os.path.basename(args.test_path))
    print("Saving results in {}".format(results))
    test(test_loader, model, args.cuda, save=results)

if __name__ == '__main__':
    main()
