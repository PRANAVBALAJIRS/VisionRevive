FFA-Net Project with Enhanced Attention and Progressive Refinement
Overview
Welcome to the FFA-Net Project repository. This project contains the implementation of the FFA-Net model with enhancements including Efficient Attention (EA) and Progressive Attention Refinement (PAR). This repository is designed to provide a comprehensive understanding of the FFA-Net architecture and the novel modifications made. Please note that due to computational constraints, the model was trained from scratch using Kaggle's GPU resources. As a result, this repository cannot be cloned and used directly for training.

Important Notice
This repository is structured to offer a clear understanding of the project components. However, if you wish to run the model, you should download and execute the FFA_NET_with_EA_and_PR.ipynb file available in this repository. For the trained model's weights and parameters, you can download the FFA_model.pth file from the repository. For further details and to run the notebook, please refer to my Kaggle page.

Project Structure


Architecture
The FFA-Net model consists of the following key components:

FFA-Net Backbone: The core architecture that includes attention mechanisms and progressive refinement modules.
Novelty Introduced
Efficient Attention (EA):

EA aims to improve computational efficiency while maintaining performance. It modifies the standard attention mechanism to reduce computational overhead.
Progressive Attention Refinement (PAR):

PAR refines features progressively through multiple stages, enhancing the model's ability to focus on important features and details.

Loss Function:
The loss function used in this project combines pixel-wise loss (L1/L2 loss) and perceptual loss. This hybrid approach ensures that both pixel-level accuracy and high-level feature representations are preserved during the training process. Here’s a detailed explanation of each component and how they are combined:

1. Pixel-Wise Loss
Pixel-wise loss measures the difference between the predicted and ground truth images at the pixel level. Two common types of pixel-wise loss are L1 loss and L2 loss:

L1 Loss (Mean Absolute Error): L1 loss calculates the absolute difference between the predicted pixel values and the actual pixel values. It is defined as:

L1 Loss
=
1
𝑁
∑
𝑖
=
1
𝑁
∣
𝑥
^
𝑖
−
𝑥
𝑖
∣
L1 Loss= 
N
1
​
  
i=1
∑
N
​
 ∣ 
x
^
  
i
​
 −x 
i
​
 ∣
where:

𝑥
^
𝑖
x
^
  
i
​
  is the predicted pixel value at position 
𝑖
i,
𝑥
𝑖
x 
i
​
  is the ground truth pixel value at position 
𝑖
i,
𝑁
N is the total number of pixels.
L2 Loss (Mean Squared Error): L2 loss computes the squared difference between the predicted and ground truth pixel values. It is defined as:

L2 Loss
=
1
𝑁
∑
𝑖
=
1
𝑁
(
𝑥
^
𝑖
−
𝑥
𝑖
)
2
L2 Loss= 
N
1
​
  
i=1
∑
N
​
 ( 
x
^
  
i
​
 −x 
i
​
 ) 
2
 
where:

𝑥
^
𝑖
x
^
  
i
​
  is the predicted pixel value at position 
𝑖
i,
𝑥
𝑖
x 
i
​
  is the ground truth pixel value at position 
𝑖
i,
𝑁
N is the total number of pixels.
L2 loss is often used when smoothness and stability are desired, while L1 loss is preferred for preserving edges and details.

2. Perceptual Loss
Perceptual loss evaluates the quality of the generated image based on high-level feature representations rather than raw pixel differences. This loss is based on the idea that the perceptual quality of an image can be better captured by comparing features extracted from pre-trained deep neural networks (such as VGG) rather than comparing pixel values directly.

The perceptual loss is computed as follows:

Feature Extraction: Pass both the generated image 
𝐼
^
I
^
  and the ground truth image 
𝐼
I through a pre-trained feature extractor (e.g., VGG-19) to obtain feature maps. Let 
𝐹
(
𝐼
^
)
F( 
I
^
 ) and 
𝐹
(
𝐼
)
F(I) denote the feature maps of 
𝐼
^
I
^
  and 
𝐼
I, respectively, at different layers of the network.

Feature Loss Calculation: Compute the loss based on the difference between feature maps. The perceptual loss can be defined as:

Perceptual Loss
=
∑
𝑙
1
𝑁
𝑙
∑
𝑖
=
1
𝑁
𝑙
∥
𝐹
𝑙
(
𝐼
^
)
𝑖
−
𝐹
𝑙
(
𝐼
)
𝑖
∥
2
2
Perceptual Loss= 
l
∑
​
  
N 
l
​
 
1
​
  
i=1
∑
N 
l
​
 
​
  
​
 F 
l
​
 ( 
I
^
 ) 
i
​
 −F 
l
​
 (I) 
i
​
  
​
  
2
2
​
 
where:

𝐹
𝑙
(
𝐼
^
)
F 
l
​
 ( 
I
^
 ) and 
𝐹
𝑙
(
𝐼
)
F 
l
​
 (I) are the feature maps at layer 
𝑙
l for the predicted and ground truth images,
𝑁
𝑙
N 
l
​
  is the number of elements (e.g., pixels or activations) in the feature map at layer 
𝑙
l,
∥
⋅
∥
2
2
∥⋅∥ 
2
2
​
  denotes the squared L2 norm.
The perceptual loss helps ensure that high-level features and textures are preserved in the generated image.

3. Combined Loss Function
To balance the pixel-wise accuracy and perceptual quality, the combined loss function is formulated as a weighted sum of the pixel-wise loss and perceptual loss:

Total Loss
=
𝛼
⋅
Pixel-Wise Loss
+
𝛽
⋅
Perceptual Loss
Total Loss=α⋅Pixel-Wise Loss+β⋅Perceptual Loss
where:

𝛼
α and 
𝛽
β are weights that control the importance of pixel-wise loss and perceptual loss, respectively.
Summary of Mathematical Formulas
L1 Loss:
L1 Loss
=
1
𝑁
∑
𝑖
=
1
𝑁
∣
𝑥
^
𝑖
−
𝑥
𝑖
∣
L1 Loss= 
N
1
​
  
i=1
∑
N
​
 ∣ 
x
^
  
i
​
 −x 
i
​
 ∣
L2 Loss:
L2 Loss
=
1
𝑁
∑
𝑖
=
1
𝑁
(
𝑥
^
𝑖
−
𝑥
𝑖
)
2
L2 Loss= 
N
1
​
  
i=1
∑
N
​
 ( 
x
^
  
i
​
 −x 
i
​
 ) 
2
 
Perceptual Loss:
Perceptual Loss
=
∑
𝑙
1
𝑁
𝑙
∑
𝑖
=
1
𝑁
𝑙
∥
𝐹
𝑙
(
𝐼
^
)
𝑖
−
𝐹
𝑙
(
𝐼
)
𝑖
∥
2
2
Perceptual Loss= 
l
∑
​
  
N 
l
​
 
1
​
  
i=1
∑
N 
l
​
 
​
  
​
 F 
l
​
 ( 
I
^
 ) 
i
​
 −F 
l
​
 (I) 
i
​
  
​
  
2
2
​
 
Combined Loss Function:
Total Loss
=
𝛼
⋅
Pixel-Wise Loss
+
𝛽
⋅
Perceptual Loss
Total Loss=α⋅Pixel-Wise Loss+β⋅Perceptual Loss
Comparison with Traditional Loss Functions
Traditional loss functions like L1 and L2 focus solely on pixel-wise differences. While effective for training, they may not capture perceptual quality or high-level features effectively. Combining these with perceptual loss enables the model to achieve better image quality by preserving textures and details, which might be missed by pixel-wise loss alone.

In your project, this combination allows for both fine-grained pixel accuracy and high-level perceptual quality, leading to improved results in tasks like image dehazing.

Feel free to adjust the weight values 
𝛼
α and 
𝛽
β based on the specific needs of your model and dataset.

Mathematics Behind Attention Mechanisms
1. Channel Attention (CA)
Channel Attention focuses on enhancing or suppressing the significance of different channels in feature maps. This mechanism is designed to emphasize the most informative channels while reducing the less relevant ones.

Formula and Computation
Channel attention is typically computed using a squeeze-and-excitation (SE) block, which can be represented as follows:

Global Average Pooling: Compute the global average pooling across spatial dimensions for each channel to obtain a channel-wise descriptor.

𝑧
𝑐
=
1
𝐻
×
𝑊
∑
ℎ
=
1
𝐻
∑
𝑤
=
1
𝑊
𝑥
𝑐
,
ℎ
,
𝑤
z 
c
​
 = 
H×W
1
​
  
h=1
∑
H
​
  
w=1
∑
W
​
 x 
c,h,w
​
 
where 
𝐻
H and 
𝑊
W are the height and width of the feature map, 
𝑥
𝑐
,
ℎ
,
𝑤
x 
c,h,w
​
  is the feature value at channel 
𝑐
c and spatial position 
(
ℎ
,
𝑤
)
(h,w), and 
𝑧
𝑐
z 
c
​
  is the channel descriptor for channel 
𝑐
c.

Fully Connected Layers: Pass the channel-wise descriptors through a small neural network (often consisting of fully connected layers) to compute channel-wise attention weights.

𝑠
𝑐
=
𝜎
(
𝑊
2
⋅
ReLU
(
𝑊
1
⋅
𝑧
𝑐
+
𝑏
1
)
+
𝑏
2
)
s 
c
​
 =σ(W 
2
​
 ⋅ReLU(W 
1
​
 ⋅z 
c
​
 +b 
1
​
 )+b 
2
​
 )
where 
𝑊
1
W 
1
​
  and 
𝑊
2
W 
2
​
  are weights, 
𝑏
1
b 
1
​
  and 
𝑏
2
b 
2
​
  are biases, and 
𝜎
σ is the sigmoid activation function.

Reweighting: Multiply the original feature map by the computed attention weights.

𝑥
~
𝑐
,
ℎ
,
𝑤
=
𝑠
𝑐
⋅
𝑥
𝑐
,
ℎ
,
𝑤
x
~
  
c,h,w
​
 =s 
c
​
 ⋅x 
c,h,w
​
 
where 
𝑥
~
𝑐
,
ℎ
,
𝑤
x
~
  
c,h,w
​
  is the reweighted feature map.

Comparison with Traditional Methods
In traditional convolutional neural networks (CNNs) without channel attention, all channels are treated equally. Channel Attention introduces a mechanism to dynamically adjust the importance of each channel, thereby improving the network's ability to focus on relevant features. Traditional methods do not have this adaptive reweighting mechanism, which can limit their ability to capture complex feature dependencies.

2. Progressive Attention (PA)
Progressive Attention involves adjusting attention weights based on the importance of features at various stages in the network. This mechanism is designed to adaptively focus on different regions or features over multiple stages of processing.

Formula and Computation
Attention Calculation: Compute the attention weights for each stage based on the feature maps at that stage. Typically, this involves a mechanism similar to the channel attention but applied at different stages.

𝛼
𝑖
,
𝑗
=
exp
⁡
(
𝑒
𝑖
,
𝑗
)
∑
𝑘
exp
⁡
(
𝑒
𝑖
,
𝑘
)
α 
i,j
​
 = 
∑ 
k
​
 exp(e 
i,k
​
 )
exp(e 
i,j
​
 )
​
 
where 
𝑒
𝑖
,
𝑗
e 
i,j
​
  is the attention score for the feature at position 
(
𝑖
,
𝑗
)
(i,j), and 
𝛼
𝑖
,
𝑗
α 
i,j
​
  is the attention weight.

Feature Aggregation: Aggregate features based on attention weights.

𝑓
~
𝑖
=
∑
𝑗
𝛼
𝑖
,
𝑗
⋅
𝑓
𝑗
f
~
​
  
i
​
 = 
j
∑
​
 α 
i,j
​
 ⋅f 
j
​
 
where 
𝑓
~
𝑖
f
~
​
  
i
​
  is the aggregated feature for position 
𝑖
i, and 
𝑓
𝑗
f 
j
​
  is the feature at position 
𝑗
j.

Comparison with Traditional Methods
Traditional attention mechanisms typically compute attention weights once and apply them throughout the network. Progressive Attention refines these weights iteratively, allowing for dynamic adjustment and better feature focusing. This progressive refinement can lead to improved performance in tasks requiring fine-grained attention.

3. Efficient Attention (EA)
Efficient Attention aims to reduce the computational complexity of the attention mechanism while maintaining performance. It introduces optimizations to make attention calculations more efficient.

Formula and Computation
Approximate Attention Calculation: Instead of computing exact attention weights, EA uses approximations to reduce computational costs. One common approach is the use of low-rank approximations or kernelized attention mechanisms.

Attention
(
𝑄
,
𝐾
,
𝑉
)
=
Softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=Softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 )V
can be approximated by:

Attention
(
𝑄
,
𝐾
,
𝑉
)
≈
Softmax
(
𝑄
𝐾
~
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)≈Softmax( 
d 
k
​
 
​
 
Q 
K
~
  
T
 
​
 )V
where 
𝐾
~
K
~
  is an approximation of 
𝐾
K.

Sparse Attention: Use sparse matrices to approximate the full attention matrix, reducing the number of computations required.

SparseAttention
(
𝑄
,
𝐾
,
𝑉
)
=
Softmax
(
𝑄
𝐾
𝑠
𝑇
𝑑
𝑘
)
𝑉
SparseAttention(Q,K,V)=Softmax( 
d 
k
​
 
​
 
QK 
s
T
​
 
​
 )V
where 
𝐾
𝑠
K 
s
​
  is a sparse approximation of 
𝐾
K.

Comparison with Traditional Methods
Traditional attention mechanisms require computing the full attention matrix, leading to high computational complexity. Efficient Attention reduces this complexity by using approximations or sparse representations, making it more scalable to large datasets and models.

4. Progressive Attention Refinement (PAR)
Progressive Attention Refinement improves feature extraction by refining attention maps at multiple levels or stages.

Formula and Computation
Initial Attention: Compute initial attention maps similar to traditional attention mechanisms.

Refinement Stages: Iteratively refine attention maps using additional attention mechanisms or layers.

𝐴
(
𝑡
+
1
)
=
Refine
(
𝐴
(
𝑡
)
)
A 
(t+1)
 =Refine(A 
(t)
 )
where 
𝐴
(
𝑡
)
A 
(t)
  is the attention map at stage 
𝑡
t, and 
Refine
Refine represents the refinement process.

Feature Integration: Combine features using refined attention maps.

𝑥
~
𝑖
=
∑
𝑗
𝐴
𝑖
,
𝑗
(
𝑇
)
⋅
𝑥
𝑗
x
~
  
i
​
 = 
j
∑
​
 A 
i,j
(T)
​
 ⋅x 
j
​
 
where 
𝑥
~
𝑖
x
~
  
i
​
  is the refined feature for position 
𝑖
i, and 
𝐴
(
𝑇
)
A 
(T)
  is the final refined attention map.

Comparison with Traditional Methods
Traditional attention mechanisms typically do not involve iterative refinement of attention maps. PAR introduces a multi-stage refinement process, which can lead to better feature extraction and improved model performance.

Differences from Original FFA-Net
The original FFA-Net model includes 12 blocks per group. In this modified version, the architecture uses 5 blocks per group to reduce computational requirements. Despite this reduction, the modified model achieves comparable SSIM values and close PSNR values to the original model.

Results
Training Loss:
Step 5000: Loss = 0.14956 | SSIM = 0.8106 | PSNR = 17.9430
Step 10000: Loss = 0.02472 | SSIM = 0.8500 | PSNR = 20.0667
Step 15000: Loss = 0.07915 | SSIM = 0.8584 | PSNR = 20.3824
Step 20000: Loss = 0.09964 | SSIM = 0.8501 | PSNR = 20.7447
The results may appear lower than those reported in the original paper due to the reduced number of blocks and computational constraints. However, the modified model still achieves competitive performance, particularly in SSIM, and maintains close PSNR values.

Visual Comparisons
Please find the side-by-side comparisons of haze and dehaze images in the figures directory. These screenshots compare the results of our model against the original paper's dehazed images.

Getting Started
Download the Notebook: Obtain the FFA_NET_with_EA_and_PR.ipynb from this repository.
Download Trained Weights: Download FFA_model.pth for pretrained model parameters.
Run the Notebook: Follow the instructions within the notebook to run the model and test its performance.
For additional details or to run the notebook, please visit my Kaggle page.

Acknowledgements
Kaggle for providing the computational resources.
The original authors of FFA-Net for their foundational work.
For any questions or issues, please contact [Your Name] at [Your Email].