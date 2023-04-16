# EECS6322 Course project: Reproducibility Challenge

This repository is the implementation of [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122).

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

To train the classification model, run classification_train.ipynb.

To train the detection model, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

To train the segmentation model, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate the classification model on CIFAR100, run classification_eval.ipynb.

To evaluate the detection model on COCO2017, run this command:

```eval
python eval.py --input-data <path_to_data> --alpha 10 --beta 20
```

To evaluate the segmentation model on ADE20K, run this command:

```eval
python eval.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Pre-trained Models

You can download the pretrained models/weights here:

- classification model: /path/to/file <br>
  trained on CIFAR100 using AdamW with parameters lr=5e-5, betas=[0.9, 0.999], weight_decay=1e-8.

- detection model: /path/to/file <br>
  trained on COCO2017 using parameters x,y,z.
  
- segmentation model: /path/to/file <br>
  trained on ADE20K using parameters x,y,z. 

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

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| PVT_segmentation   |     85%         |      95%       |
