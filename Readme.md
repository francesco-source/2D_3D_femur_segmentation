# Methods for 3D Femur Segmentation

This repository contains the source code for the project "Methods for 3D Femur Segmentation," which is part of the Pattern Recognition course. The project aims to enhance 3D femur segmentation from CT images using various deep learning methods.

## Project Overview

Segmentation in the medical field, particularly femur segmentation, is crucial as it provides doctors with valuable information for patient diagnosis and treatment. This project focuses on improving 3D femur segmentation from CT images using deep learning techniques.

## Contributors

- Francesco Pivi
- Matteo Fusconi

Both contributors are pursuing a Master’s degree in Artificial Intelligence at the University of Bologna.

## Contact

- Francesco Pivi: [francesco.pivi@studio.unibo.it](mailto:francesco.pivi@studio.unibo.it)
- Matteo Fusconi: [matteo.fusconi4@studio.unibo.it](mailto:matteo.fusconi4@studio.unibo.it)

## Abstract

The project explores various deep learning methods for 2D and 3D binary segmentation in the medical field. We fine-tuned TotalSegmentator for 3D segmentation and trained UNet, UNet++, and DeepLabv3 models with ResNet50 and ResNet34 backbones for 2D segmentation. The results indicate that fine-tuning a large 3D UNet on a small dataset is suboptimal, whereas 2D finetuned models achieved better performance.

## Methods and Models

### 3D Segmentation

- **TotalSegmentator**: A 3D full-resolution nnUNet used as the baseline.
- **nnUNet v2**: Used for fine-tuning on the specific dataset.

### 2D Segmentation

- **UNet**: Trained with ResNet50 and ResNet34 backbones.
- **UNet++**: Enhanced version of UNet, trained with ResNet50 and ResNet34 backbones.
- **DeepLabv3**: Trained with ResNet50 and ResNet34 backbones.

## Evaluation Metrics

- Dice score
- Mean Intersection over Union (mIOU)
- Volume Similarity

## Results

- **DeepLabv3**: Best performance in segmenting entire femurs with a Dice score of 0.952.
- **ResNet34**: More effective for segmenting femur heads with a Dice score of 0.970.
- **UNet++**: Consistently outperformed UNet across various metrics.

## Data

The dataset consists of 40 CT images of femurs provided by Ospedale Rizzoli. Each image is stored in NRRD format.

## Acknowledgements
We would like to thank Ospedale Rizzoli for providing the dataset and the University of Bologna for their support. The phd candidate Riccardo Biondi and the professor Gastone Castellani.

## References
1. UNet: Convolutional Networks for Biomedical Image Segmentation
2. UNet++: A Nested U-Net Architecture for Medical Image Segmentation
3. DeepLabv3: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
4. nnUNet: Self-adapting Framework for U-Net-Based Medical Image Segmentation
5. ResNet: Deep Residual Learning for Image Recognition

For more detailed information, please refer to the project report.
