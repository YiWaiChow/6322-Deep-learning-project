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
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
import distinctipy

import torch

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


def decode_seg_map_sequence(label_masks):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    print(np.array(rgb_masks).shape)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, plot=True):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 150
    label_colours = distinctipy.get_colors(n_classes, rng=1)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll][0]
        g[label_mask == ll] = label_colours[ll][1]
        b[label_mask == ll] = label_colours[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def test(checkpoint_path):
    batch_size = 1
    _, val_loader, _, num_class = make_data_loader(batch_size, "..\\ADEChallengeData2016\\images",
                                                   "..\\ADEChallengeData2016\\annotations")

    # Define network
    blocks = [2, 4, 23, 3]
    pvt = Segmentic_Pvt(blocks, 150, channels=3,
                        height=512, width=512, batch_size=batch_size)

    # Define Optimizer

    checkpoint = torch.load(checkpoint_path)
    pvt.load_state_dict(checkpoint["state_dict"])

    evaluator = Evaluator(num_class)
    for iter, batch in enumerate(val_loader):
        image, target = batch['image'], batch['label']

        print_target = target.clone()
        print(image.shape)
        plt.imshow(torch.squeeze(image).permute(1, 2, 0).cpu().numpy())
        plt.show()
        plt.imshow(torch.squeeze(print_target).cpu().numpy())
        plt.show()
        with torch.no_grad():
            output = pvt(image)

        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator

        evaluator.add_batch(target, pred)

        # Fast test during the training
        # Acc = evaluator.Pixel_Accuracy()
        # Acc_class = evaluator.Pixel_Accuracy_Class()
        # mIoU = evaluator.Mean_Intersection_over_Union()
        # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print(pred.shape)
        prediction_rgb = decode_segmap(np.squeeze(pred))

        # print('Validation:')
        # print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(
        #     Acc, Acc_class, mIoU, FWIoU))
        break


if __name__ == "__main__":
    test("segementic_work_dir\model_best.pth.tar")
