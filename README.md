This repository provides the necessary resources to replicate the project “Why Resizing Matters: Insights from Deep Learning on Breast Ultrasound".

### **Package Management** ###

All of the projects in this repository are managed by UV: https://github.com/astral-sh/uv. This is an extremely fast package manager that enables faster and easier dependency replication than the default `pip` package installer. It is strongly recommended you use this when navigating this repository. 

### **Datasets** ###

The following links are the locations of the datasets used within the study:

BrEaST: https://www.cancerimagingarchive.net/collection/breast-lesions-usg/

BUSI: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

OASBUD : https://zenodo.org/record/545928#.Y_TIs4DP20n

RODTOOK : http://www.onlinemedicalimages.com/index.php/en/81-site-info/73-introduction

UDIAT : http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php

BrEaST, BUSI, and OASBUD are simple downloads. RODTOOK requires navigating the website and downloading the images one by one. For UDIAT, a licensing agreement must be signed. 

To replicate the curated dataset used in this research project, it is advised to follow the ‘dataset_splits.xlsx’ file provided in this repository. It details all sample names, mask names, origin datasets, and whether they belong in the train/val/test split.

Preprocessing implementations are also included. Note that these are separate Python projects.

### **Models** ###

10 models were used, 5 CNN and 5 ViT. Some were pre-trained on ImageNet-1K, while others were pre-trained on ImageNet-21K. As of writing this, `torchvision` in PyTorch only provides pre-trained weights for models trained on ImageNet-1K. Therefore, the ImageNet-21K models require extra accommodation to function properly. Please follow the training scripts provided in this repository. 

Please note that some training scripts require imports from Big Transfer (https://github.com/google-research/big_transfer). These need to be sourced manually.

| **Architecture** | **Model** | **Link** |
| --- | --- | --- |
| CNN | VGG16-1K | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html |
|  | InceptionV3-1K | https://docs.pytorch.org/vision/main/models/inception.html |
|  | ResNet50x1-1K | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html |
|  | ResNet50x1-21K | https://github.com/google-research/big_transfer |
|  | MobileNetV3-Large-21K | https://github.com/Alibaba-MIIL/ImageNet21K?tab=readme-ov-file |
| ViT | ViT-Tiny/16-21K | https://huggingface.co/WinKawaks/vit-tiny-patch16-224 |
|  | ViT-Base/16-1K | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html |
|  | ViT-Base/16-21K | https://huggingface.co/google/vit-base-patch16-224-in21k |
|  | ViT-Large/16-1K | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_l_16.html |
|  | ViT-Large/16-21K | https://huggingface.co/google/vit-large-patch16-224-in21k |

### Citation ###
If you use the curated dataset or code from this repository, please consider citing the paper:

```bibtex
@InProceedings{bristow2026miua,
    author    = {Joe Bristow and Moi Hoon Yap},
    title     = {Why Resizing Matters: Insights from Deep Learning on Breast Ultrasound},
    booktitle = {Proceedings of the Medical Image Understanding and Analysis (MIUA)},
    month     = {July},
    year      = {2026},
    pages     = {to appear}
}
```
