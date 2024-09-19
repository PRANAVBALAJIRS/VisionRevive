# Indoor Training Set (ITS) [RESIDE-Standard] & Synthetic Objective Testing Set (SOTS) [RESIDE]

## Overview
This repository uses data from the Indoor Training Set (ITS) and the Synthetic Objective Testing Set (SOTS) from the **REalistic Single Image DEhazing (RESIDE)** dataset, widely recognized for its use in evaluating single image dehazing algorithms.

### Dataset Summary:
**ITS (Indoor Training Set):**
- **Images**: 1399 clear images
- **Hazy Images**: 13,990 synthetic hazy images
- **Trans Images**: 13,990 transmission images
- **Purpose**: Used for training single image dehazing models.

**SOTS (Synthetic Objective Testing Set):**
- **Subsets**: Indoor and outdoor clear and hazy images
- **Purpose**: Used for testing and evaluating the performance of dehazing models.

## Dataset Context
**RESIDE** is a comprehensive benchmark consisting of synthetic and real-world hazy images. The dataset allows for detailed study and evaluation of single image dehazing algorithms. It highlights diverse data sources and content, with a clear division into multiple subsets for different training and evaluation purposes.

- **ITS**: This set is designed for training dehazing models using a large number of paired hazy and clear images, with each clear image corresponding to multiple hazy images with varying levels of haze.
- **SOTS**: This set is designed for testing the performance of trained dehazing models on both indoor and outdoor scenes, providing a diverse and realistic benchmark for algorithm evaluation.

## Usage
This dataset is crucial for:
- **Training** single image dehazing models on synthetic data (ITS).
- **Testing and evaluating** dehazing models using synthetic objective data (SOTS).

The training and testing datasets ensure the generalization of models to real-world scenarios and help benchmark their performance against established standards.

## Dataset Source
This dataset is part of the **RESIDE-Standard** benchmark and was obtained from the official RESIDE Dataset homepage. For more details on the dataset, please refer to the related publication mentioned below.

## Citation
Please cite the following paper if you use the dataset in your work:
```bibtex
@article{li2019benchmarking,
  title={Benchmarking Single-Image Dehazing and Beyond},
  author={Li, Boyi and Ren, Wenqi and Fu, Dengpan and Tao, Dacheng and Feng, Dan and Zeng, Wenjun and Wang, Zhangyang},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={1},
  pages={492--505},
  year={2019},
  publisher={IEEE}
}
