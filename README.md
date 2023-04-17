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


### ADE20k
The structure of the dataset has been changed to fit into the dataloader, please download from my google drive through this link:

https://drive.google.com/file/d/1kQZGjiKCMdv2SpYVDftdez0gSRPDEX4H/view?usp=sharing

## Training

To train the classification model, run [classification_train.ipynb](classification/classification_train.ipynb).

To train the detection model, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

To train the segmentation model, run the [segmentation/training.py](segmentation/training.py)

The dataset is assumed to be located a folder 1 layer outside the project root directory
"..\\ADEChallengeData2016"

## Evaluation

To evaluate the classification model on CIFAR100, run [classification_test.ipynb](classification/classification_test.ipynb).

To evaluate the detection model on COCO2017, run this command:

```eval
python eval.py --input-data <path_to_data> --alpha 10 --beta 20
```

To evaluate the segmentation model on ADE20K, call the test function in segmentation/test.py


## Pre-trained Models

You can download the pretrained models/weights here:

- classification model: [cifar100_new_params.pth](ckpt_cifar100/cifar100_new_params.pth) <br>
  trained on CIFAR100 using AdamW with parameters lr=5e-5, betas=[0.9, 0.999], weight_decay=1e-8.

- detection model: [coco_retinanet.pt](ckpt_coco17/coco_retinanet.pt) <br>
  trained on COCO2017 using AdamW with parameters lr=1e-4.
  
- segmentation model: https://drive.google.com/file/d/1gB889CepIlE-Q_bOvZCRqAVFV6V4SbXQ/view?usp=sharing <br>
  trained on ADE20K using parameters AdamW with parameters lr=0.0001, weight_decay=0.001. 

## Results

Our model achieves the following performance on :

### Image Classification on CIFAR100

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| PVT_classification |     85%         |      95%       |

### Object Detection on COCO2017

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| PVT_detection      |     85%         |      95%       |

### Semantic Segmentation on ADE20K

Tested on 2000 image
| Model name         | Accuracy  | class Accurary | mIoU | fwIoU |
| ------------------ | --------  | -------------- | ---- | ----- |
| PVT_segmentation   |  49.25%   |      7.1%      | 4.8% | 32.2% |
