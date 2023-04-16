
from torch.utils.data import DataLoader
from torch.utils import data
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        # plt.imshow(img)
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        # plt.imshow(img)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(
                0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size * 2)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size * 2))
        mask = mask.crop(
            (x1, y1, x1 + self.crop_size, y1 + self.crop_size * 2))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class Ade20KSegmentation(data.Dataset):
    NUM_CLASSES = 150

    def __init__(self, image_base, annotation_base, split="train"):

        self.split = split
        self.files = {}
        self.base_size = 512
        self.crop_size = 512
        self.images_base = image_base
        self.annotations_base = annotation_base

        self.files[split] = self.recursive_glob(
            rootdir=self.images_base, suffix='.jpg')

        self.void_classes = [150]
        self.valid_classes = list(range(150))
        self.class_names = ['wall', 'building, edifice', 'sky', 'floor, flooring', 'tree', 'ceiling', 'road, route', 'bed ', 'windowpane, window ', 'grass', 'cabinet', 'sidewalk, pavement', 'person, individual, someone, somebody, mortal, soul', 'earth, ground', 'door, double door', 'table', 'mountain, mount', 'plant, flora, plant life', 'curtain, drape, drapery, mantle, pall', 'chair', 'car, auto, automobile, machine, motorcar', 'water', 'painting, picture', 'sofa, couch, lounge', 'shelf', 'house', 'sea', 'mirror', 'rug, carpet, carpeting', 'field', 'armchair', 'seat', 'fence, fencing', 'desk', 'rock, stone', 'wardrobe, closet, press', 'lamp', 'bathtub, bathing tub, bath, tub', 'railing, rail', 'cushion', 'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace, hearth, open fireplace', 'refrigerator, icebox', 'grandstand, covered stand', 'path', 'stairs, steps', 'runway', 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table', 'pillow', 'screen door, screen', 'stairway, staircase', 'river', 'bridge, span', 'bookcase', 'blind, screen', 'coffee table, cocktail table', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove, kitchen stove, range, kitchen range, cooking stove', 'palm, palm tree', 'kitchen island', 'computer, computing machine, computing device, data processor, electronic computer, information processing system', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel, hut, hutch, shack, shanty', 'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', 'towel',
                            'light, light source', 'truck, motortruck', 'tower', 'chandelier, pendant, pendent', 'awning, sunshade, sunblind', 'streetlight, street lamp', 'booth, cubicle, stall, kiosk', 'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', 'airplane, aeroplane, plane', 'dirt track', 'apparel, wearing apparel, dress, clothes', 'pole', 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail', 'escalator, moving staircase, moving stairway', 'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'buffet, counter, sideboard', 'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship', 'fountain', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy', 'washer, automatic washer, washing machine', 'plaything, toy', 'swimming pool, swimming bath, natatorium', 'stool', 'barrel, cask', 'basket, handbasket', 'waterfall, falls', 'tent, collapsible shelter', 'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball', 'food, solid food', 'step, stair', 'tank, storage tank', 'trade name, brand name, brand, marque', 'microwave, microwave oven', 'pot, flowerpot', 'animal, animate being, beast, brute, creature, fauna', 'bicycle, bike, wheel, cycle', 'lake', 'dishwasher, dish washer, dishwashing machine', 'screen, silver screen, projection screen', 'blanket, cover', 'sculpture', 'hood, exhaust hood', 'sconce', 'vase', 'traffic light, traffic signal, stoplight', 'tray', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'fan', 'pier, wharf, wharfage, dock', 'crt screen', 'plate', 'monitor, monitoring device', 'bulletin board, notice board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (
                split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(
            self.annotations_base + "\\" + filename_without_ext + '.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            train_set = self.transform_tr(sample)
            return train_set
        elif self.split == 'val':
            val_set = self.transform_val(sample)
            return val_set
        elif self.split == 'test':
            test_set = self.transform_ts(sample)
            return test_set

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            FixedResize(size=self.crop_size),
            # tr.RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            FixedResize(size=self.crop_size),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            FixedResize(size=self.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)


def make_data_loader(batch_size, images_folder, annotations_folder):

    train_set = Ade20KSegmentation(image_base=images_folder + "\\training",
                                   annotation_base=annotations_folder+"\\training", split='train')
    val_set = Ade20KSegmentation(image_base=images_folder + "\\validation",
                                 annotation_base=annotations_folder+"\\validation", split='val')
    # test_set = Ade20KSegmentation(root=, split='test')
    test_set = val_set
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False)

    test_loader = val_loader
    # test_loader = DataLoader(
    # test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, num_class


if __name__ == '__main__':
    train_loader, val_loader, test_loader, num_class = make_data_loader(5, "..\\ADEChallengeData2016\\images",
                                                                        "..\\ADEChallengeData2016\\annotations")
    x = next(iter(train_loader))
    print(x["image"].shape, x["label"].shape)
