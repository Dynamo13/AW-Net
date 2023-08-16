# AW-Net: A Novel Fully Connected Attention-based Medical Image Segmentation Model
This repository contains the codebase of AW-Net which focuses on segmenting multimodal medical images such as MRI, PET, and CT scan images for multiple organs such as brain, breast, and spine. 


## Overview
A novel fully connected segmentation model which provides a solution to problem of segmenting multi-modal 3D/4D medical images by incorporating a novel regularized transient block. AW-Net uses L1 regularizers followed by dropout layers to improve model performance. The implementation is inspired from ***Attention UW-Net: A fully connected model for automatic segmentation and annotation of chest X-ray***  [Code](https://github.com/Dynamo13/Attention_UWNet) | [Paper]( https://www.sciencedirect.com/science/article/abs/pii/S0010482522007910).

## AW-Net Architecture

## Datasets
The datasets used in the paper can be downloaded from the links below:
- [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [RSNA Cervical Spine 2022](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)
- [Duke Breast Cancer MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903)
- [QIN Breast](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast)
