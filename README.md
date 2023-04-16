# EECS6322 Course project: Reproducibility Challenge

This repository is the implementation of [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122).. 

## Project Structure


```
.
├── README.md
├── classification_train.ipynb
├── detection_train.ipynb
├── segmentation_train.ipynb
├── ckpt_cifar100
│   ├── cifar100_new_params.pth
│   ├── cifar100_og_params.pth
├── ckpt_coco2017
├── ckpt_ade20k
└── model.py
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |
