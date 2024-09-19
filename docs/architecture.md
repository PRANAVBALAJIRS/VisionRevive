# FFA-Net Architecture with Efficient Attention and Progressive Attention Refinement

## Overview
This document outlines the architecture of the **Feature Fusion Attention Network (FFA-Net)**, designed for **single image dehazing** tasks. The network incorporates multiple attention mechanisms to enhance dehazing performance, including:

- **Channel Attention (CA)**
- **Pixel Attention (PA)**
- **Efficient Attention (EA)**
- **Progressive Attention Refinement (PAR)**

These mechanisms enable the model to selectively focus on significant features in the image, enhancing the dehazing results.

---

## Key Components:

1. **Channel Attention (CA)**
2. **Pixel Attention (PA)**
3. **Efficient Attention**
4. **Progressive Attention Refinement**

## 1. Feature Fusion Attention Network (FFA)
The core structure of FFA-Net consists of three Groups of blocks, where each block contains a combination of convolutional layers and attention mechanisms. The network is designed to refine the features progressively and combines them at different stages to achieve robust dehazing performance.

### Network Structure:
- **Input**: Hazy image of size (H, W, 3).
- **Output**: Dehazed image of the same size as the input.

#### Main Components:
- **Pre-processing**: The input image is passed through a convolutional layer to extract initial features.
- **Groups (G1, G2, G3)**: The network contains three groups (G1, G2, G3), each consisting of multiple blocks. Each block applies convolutions and attention mechanisms, including channel and pixel attention as well as the novel efficient and progressive attention refinement.
- **Attention Fusion**: After processing through the groups, the features from G1, G2, and G3 are concatenated and refined using channel attention and a weighted summation.
- **Post-processing**: The final fused features are passed through convolutional layers to produce the dehazed output image.

## 2. Attention Mechanisms

### 2.1 Channel Attention (CA)
Channel attention focuses on the inter-channel relationships of features. It enhances or suppresses specific channels by learning which channels are most important for dehazing.

- **Adaptive Average Pooling**: Averages the spatial information to generate channel statistics.
- **1x1 Convolutions and ReLU**: Used to reduce and then restore channel dimensions, producing attention weights.
- **Sigmoid Activation**: Scales the input features by channel importance.

### 2.2 Pixel Attention (PA)
Pixel attention focuses on spatially specific regions of the image, emphasizing critical areas such as edges or regions heavily affected by haze.

- **Convolutions and ReLU**: Learn spatial attention weights.
- **Sigmoid Activation**: Produces a mask to weigh spatial regions in the image.

### 2.3 Efficient Attention
Efficient attention reduces the computational complexity by applying lightweight attention modules. This mechanism learns to emphasize important feature maps with minimal overhead.

- **1x1 Convolutions**: Used to compute attention maps efficiently.
- **Sigmoid Activation**: Produces the attention map for feature refinement.

### 2.4 Progressive Attention Refinement
Progressive attention refinement further enhances the feature extraction by applying both channel and pixel attention in sequence. This allows the network to refine features progressively as they pass through different layers.

- **Channel and Pixel Attention**: Both types of attention are applied progressively to refine the features at each step.
- **Multiplicative Combination**: Combines both attention outputs to generate refined feature maps.

## 3. Block Structure
Each block in the network applies the following sequence of operations:

1. **Convolutional Layer**: Extracts feature maps from the input.
2. **ReLU Activation**: Applies non-linearity.
3. **Convolutional Layer**: Further refines the features.
4. **Channel Attention**: Enhances important channels.
5. **Pixel Attention**: Focuses on spatially important regions.
6. **Efficient Attention**: Applies lightweight feature refinement.
7. **Progressive Attention Refinement**: Refines features with both channel and pixel attention mechanisms.

The residual connection ensures that the original features are retained and progressively refined.

## 4. Group Structure
Each Group consists of multiple blocks stacked together, followed by a convolutional layer. The output of each group is combined using attention mechanisms in the final layers.

## 5. Feature Fusion
The outputs from the three groups (G1, G2, G3) are concatenated and then refined using channel attention. The fused features are then processed further to generate the final dehazed output image.

## Conclusion
The FFA-Net architecture integrates multiple attention mechanisms to progressively refine features for single image dehazing. By leveraging Efficient Attention and Progressive Attention Refinement, the network achieves high performance with minimal computational overhead. This architecture is designed to handle diverse hazy conditions effectively and produce clear, dehazed images.
