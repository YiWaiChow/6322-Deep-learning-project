from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging
import glob

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from dataset import make_data_loader
from segmentation import Segmentic_Pvt


from util import Evaluator
from utils.saver import Saver
from util import SegmentationLosses

from segmentation import Segmentic_Pvt


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


class Trainer(object):
    def __init__(self, no_validation):

        self.batch_size = 5
        self.lr_decay_gamma = 0.01
        self.weight_decay = 0.001
        self.no_val = no_validation

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        self.train_loader, self.val_loader, self.test_loader, self.num_class = make_data_loader(self.batch_size, "..\\ADEChallengeData2016\\images",
                                                                                                "..\\ADEChallengeData2016\\annotations")
        self.cuda = True
        # Define network
        blocks = [2, 4, 23, 3]
        pvt = Segmentic_Pvt(blocks, 150, channels=3,
                            height=512, width=512, batch_size=self.batch_size)
        # Define Optimizer
        self.lr = 0.1

        self.lr = self.lr * 0.1
        optimizer = torch.optim.Adam(
            pvt.parameters(), lr=self.lr, momentum=0, weight_decay=self.weight_decay)

        # Define Criterion
        weight = None
        self.criterion = SegmentationLosses(
            weight=weight, cuda=self.cuda).build_loss(mode='ce')

        self.model = pvt
        self.optimizer = optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.num_class)

        # Using cuda

        self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.lr_stage = [68, 93]
        self.lr_staget_ind = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        if self.lr_staget_ind > 1 and epoch % (self.lr_stage[self.lr_staget_ind]) == 0:
            adjust_learning_rate(self.optimizer, self.lr_decay_gamma)
            self.lr *= self.lr_decay_gamma
            self.lr_staget_ind += 1
        for iteration, batch in enumerate(self.train_loader):
            if self.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long())
            loss_val = loss.item()
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            train_loss += loss.item()

            if iteration % 10 == 0:
                print("Epoch[{}]({}/{}):Loss:{:.4f}, learning rate={}".format(epoch,
                                                                              iteration, len(self.train_loader), loss.data, self.lr))
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, iteration * self.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        for iter, batch in enumerate(self.val_loader):
            if self.args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            print('Test Loss:%.3f' % (test_loss/(iter+1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, iter * self.args.batch_size + image.shape[0]))
        print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(
            Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():

    trainer = Trainer()
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == '__main__':
    main()
