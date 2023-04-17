# EECS6322 Course project: Reproducibility Challenge

This repository is the implementation of [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122).

## Project Structure


```
.
├── README.md
|
├── classification
│   ├── classification_test.ipynb
│   ├── classification_train.ipynb
|
├── detection
│   ├── retinanet
|   |   |── __init__.py
|   |   |── anchors.py
|   |   |── coco_eval.py
|   |   |── dataloader.py
|   |   |── losses.py
|   |   |── model.py
|   |   |── oid_dataset.py
|   |   |── pvt.py
|   |   |── utils.py
│   ├── coco_validation.py
│   ├── train.py
|
├── segmentation
│   ├── dataset.py
│   ├── model.py
│   ├── segmentation_model.py
│   ├── training.py
│   ├── test.py
│   ├── util.py
|
├── ckpt_cifar100
│   ├── cifar100_new_params.pth
│   ├── cifar100_og_params.pth
|
├── ckpt_coco17
│   ├── coco_retinanet.pt
|
├── ckpt_ade20k
|
└── model.py
```

## Datasets

### CIFAR100
The CIFAR100 dataset can be loaded from the PyTorch dataloader. You may refer to [classification_test.ipynb](classification/classification_train.ipynb) or [PyTorch CIFAR100 Documentation](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html).

### COCO2017
To download the [COCO 2017 dataset](https://cocodataset.org/#download), run the following commands:

```download
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Here is how the dataset should be structured:
```
.
├── COCO2017
│   ├── annotations
|   |   |── instances_train2017.json
|   |   |── instances_val2017.json
│   ├── images
|   |   |── train2017
|   |   |   |── 0.jpg
|   |   |   |── 1.jpg
|   |   |   |── ...
|   |   |── val2017
|   |   |   |── 2.jpg
|   |   |   |── 3.jpg
|   |   |   |── ...
└──
```

### ADE20k
The structure of the dataset has been changed to fit into the dataloader, please download from my google drive through this link:

https://drive.google.com/file/d/1kQZGjiKCMdv2SpYVDftdez0gSRPDEX4H/view?usp=sharing

Here is how the dataset should be structured:
```
.
├── ADE20k
|   ├── metadata
│   ├── images
|   |   |── train
|   |   |   |── 0.jpg
|   |   |   |── 1.jpg
|   |   |   |── ...
|   |   |── val
|   |   |   |── 2.jpg
|   |   |   |── 3.jpg
|   |   |   |── ...
│   ├── annotation 
|   |   |── train
|   |   |   |── 0.jpg
|   |   |   |── 1.jpg
|   |   |   |── ...
|   |   |── val
|   |   |   |── 2.jpg
|   |   |   |── 3.jpg
|   |   |   |── ...
└──
```

## Training

### Image Classification on CIFAR100

To train the classification model, run [classification_train.ipynb](classification/classification_train.ipynb).

### Object Detection on COCO2017

To train the detection model, run this command:

```train
python .\detection\train.py --dataset coco --coco_path <path_to_coco>
```

### Semantic Segmentation on ADE20K

To train the segmentation model, run the segmentation/training.py

The dataset is assumed to be located a folder 1 layer outside the project root directory
"..\\ADEChallengeData2016"

## Evaluation

### Image Classification on CIFAR100

To evaluate the classification model on CIFAR100, run [classification_test.ipynb](classification/classification_test.ipynb).

### Object Detection on COCO2017

To evaluate the detection model on COCO2017, run this command:

```eval
python .\detection\coco_validation.py --coco_path <path_to_coco> --model_path .\ckpt_coco17\coco_retinanet.pt
```

### Semantic Segmentation on ADE20K

To evaluate the segmentation model on ADE20K, call the test() function in segmentation/test.py


## Pre-trained Models

You can download the pretrained models/weights here:

- classification model: [cifar100_new_params.pth](ckpt_cifar100/cifar100_new_params.pth) <br>
  trained on CIFAR100 using AdamW with parameters lr=5e-5, betas=[0.9, 0.999], weight_decay=1e-8.

- detection model: [coco_retinanet.pt](ckpt_coco17/coco_retinanet.pt) <br>
  trained on COCO2017 using AdamW with parameters lr=1e-4.
  
- segmentation model: https://drive.google.com/file/d/1gB889CepIlE-Q_bOvZCRqAVFV6V4SbXQ/view?usp=sharing <br>
  trained on ADE20K using AdamW with parameters lr=0.0001, weight_decay=0.001. 

## Results

Our model achieves the following performance on :

### Image Classification on CIFAR100

| Model name                | Top 1 Error     |
| ------------------        |---------------- |
| PVT-Tiny                  |     24.9%       |
| PVT-Small                 |     20.2%       |
| PVT-Medium                |     18.8%       |
| PVT-Large                 |     18.3%       |
| PVT_classification (Ours) |      68.38%     |

### Object Detection with RetinaNet on COCO2017

| Backbone              | AP                | AP<sub>50</sub>  | AP<sub>75</sub>  | AP<sub>S</sub>   | AP<sub>M</sub>   | AP<sub>L</sub>   |
| ------------------    |----------------   | --------------   |----------------  | --------------   | --------------   | --------------   |
| PVT-Tiny              |     36.7%         |      56.9%       |      38.9%       |      22.6%       |      38.8%       |      50%         |
| PVT-Small             |     40.4%         |      61.3%       |      43%         |      25.0%       |      42.9%       |      55.7%       |
| PVT-Medium            |     41.9%         |      63.1%       |      44.3%       |      25.0%       |      44.9%       |      57.6%       |
| PVT-Large             |     42.6%         |      63.7%       |      45.4%       |      25.8%       |      46%         |      58.4%       |
| PVT_detection (Ours)  |     2.5%          |      5.3%        |      2%          |      0.8%        |      2.5%        |      4.3%        |

### Semantic Segmentation on ADE20K

Tested on 2000 image:

| Backbone                  | Accuracy  | class Accurary | mIoU | fwIoU |
| ------------------        |-------    | -------------- |----- |-------| 
| PVT_segmentation (Ours)   |  49.25%   |      7.1%      | 4.8% | 32.2% |

Comparsion with performance in the paper:

| Backbone                  | mIoU  |
| ------------------        |-------| 
| PVT_segmentation (Ours)   | 4.8%  |
| PVT_Tiny                  | 35.7% |
| PVT_Small                 | 39.8% |
| PVT_Medium                | 41.6% |
| PVT_Large                 | 42.1% |
| PVT_Large*                | 44.8% |


