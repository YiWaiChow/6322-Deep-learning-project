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


from util import Evaluator
from util import Saver
from util import SegmentationLosses

from segmentation_model import Segmentic_Pvt
from torch.utils.tensorboard import SummaryWriter

# the following codes are modifiy from https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation
# using their implementation of FPN model, as this is not the main contribution of the PvT paper


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


class Trainer(object):
    def __init__(self, start_epoch, end_epoch, no_validation, load_check_point, check_point_path=None, lr=0.01):

        self.writer = None
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        # batch_size needs to be divisible by the len of the dataset, otherwise the model will mess up
        self.batch_size = 5
        self.lr_decay_gamma = 0.01
        self.weight_decay = 0.001
        self.no_val = no_validation
        self.lr = lr
        self.validation_itr = 0
        # Define Saver
        self.saver = Saver("./segementic_work_dir",
                           "ADE20K", "Sementic_Pvt", self.lr, start_epoch)
        self.saver.save_experiment_config()

        self.train_loader, self.val_loader, self.test_loader, self.num_class = make_data_loader(self.batch_size, "..\\ADEChallengeData2016\\images",
                                                                                                "..\\ADEChallengeData2016\\annotations")
        self.cuda = True
        # Define network
        blocks = [2, 4, 23, 3]
        pvt = Segmentic_Pvt(blocks, 150, channels=3,
                            height=512, width=512, batch_size=self.batch_size)

        # self.lr = self.lr * 0.1
        optimizer = torch.optim.AdamW(
            pvt.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Define Optimizer

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

        if(load_check_point):
            checkpoint = torch.load(check_point_path)
            print(list(checkpoint.keys()))
            pvt.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.saver.epochs = checkpoint["epoch"]
            self.best_pred = checkpoint['best_pred']
            print("loaded check point")

    def training(self, epoch):
        self.writer = SummaryWriter(
            'segmentation_runs/training_epoch_{}'.format(epoch))
        torch.backends.cudnn.enabled = False
        train_loss = 0.0
        self.model.train()
        global_it = 0
        num_img_tr = len(self.train_loader)
        if self.lr_staget_ind > 1 and epoch % (self.lr_stage[self.lr_staget_ind]) == 0:
            adjust_learning_rate(self.optimizer, self.lr_decay_gamma)
            self.lr *= self.lr_decay_gamma
            self.lr_staget_ind += 1
        for iteration, batch in enumerate(self.train_loader):

            image, target = batch['image'], batch['label']
            image, target = image.cuda(), target.cuda()

            if(image.shape != torch.Size([5, 3, 512, 512])):
                print(
                    "something is wrong with this image, skip to the next batch", image.shape)
                continue
            self.optimizer.zero_grad()
            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels.long())
            print(loss)
            self.writer.add_scalar('Loss/train', loss, global_it)
            loss_val = loss.item()

            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            train_loss += loss.item()
            global_it += 1
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
        self.writer = SummaryWriter(
            'segmentation_runs/validation_epoch_{}'.format(epoch))
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        global_it = 0
        for iter, batch in enumerate(self.val_loader):
            image, target = batch['image'], batch['label']
            image, target = image.cuda(), target.cuda()

            if(image.shape != torch.Size([5, 3, 512, 512])):
                print(
                    "something is wrong with this image, skip to the next batch", image.shape)
                continue
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            self.writer.add_scalar('Test_Loss/validation', loss, global_it)
            print('Test Loss:%.3f' % (test_loss/(iter+1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            global_it += 1
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        self.writer.add_scalar('Accuracy/validation', Acc, self.validation_itr)
        self.writer.add_scalar('Acc_class/validation',
                               Acc_class, self.validation_itr)
        self.writer.add_scalar('mIoU/validation', mIoU, self.validation_itr)
        self.writer.add_scalar('FWIoU/validation', FWIoU, self.validation_itr)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, iter * self.batch_size + image.shape[0]))
        print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(
            Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        self.validation_itr += 1
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


def main(check_point_path, load_check_point):

    trainer = Trainer(0, 10, True, load_check_point=load_check_point,
                      check_point_path=check_point_path, lr=0.0001)

    eval_interval = 2
    for epoch in range(trainer.start_epoch, trainer.end_epoch):
        trainer.training(epoch)
        if epoch % eval_interval == 0:
            print("validating at epoch {}".format(epoch))
            trainer.validation(epoch)


if __name__ == '__main__':
    main("segementic_work_dir\model_best.pth.tar", True)
