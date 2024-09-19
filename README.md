<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>FFA-Net Project with Enhanced Attention and Progressive Refinement</h1>

<h2>Overview</h2>
<p>Welcome to the FFA-Net Project repository. This project contains the implementation of the FFA-Net model with enhancements including Efficient Attention (EA) and Progressive Attention Refinement (PAR). This repository is designed to provide a comprehensive understanding of the FFA-Net architecture and the novel modifications made. Please note that due to computational constraints, the model was trained from scratch using Kaggle's GPU resources. As a result, this repository cannot be cloned and used directly for training.</p>

<h2>Important Notice</h2>
<p>This repository is structured to offer a clear understanding of the project components. However, if you wish to run the model, you should download and execute the <strong>FFA_NET_with_EA_and_PR.ipynb</strong> file available in this repository. For the trained model's weights and parameters, you can download the <strong>FFA_model.pth</strong> file from the repository. For further details and to run the notebook, please refer to my Kaggle page.</p>

<h2>Project Structure</h2>

<h3>Architecture</h3>
<p>The FFA-Net model consists of the following key components:</p>
<ul>
    <li><strong>FFA-Net Backbone:</strong> The core architecture that includes attention mechanisms and progressive refinement modules.</li>
</ul>

<h3>Novelty Introduced</h3>
<ul>
    <li><strong>Efficient Attention (EA):</strong> EA aims to improve computational efficiency while maintaining performance. It modifies the standard attention mechanism to reduce computational overhead.</li>
    <li><strong>Progressive Attention Refinement (PAR):</strong> PAR refines features progressively through multiple stages, enhancing the model's ability to focus on important features and details.</li>
</ul>

<h3>Loss Function</h3>
<p>The loss function used in this project combines pixel-wise loss (L1/L2 loss) and perceptual loss. This hybrid approach ensures that both pixel-level accuracy and high-level feature representations are preserved during the training process. Here’s a detailed explanation of each component and how they are combined:</p>

</body>
</html>


### 1. Pixel-Wise Loss
Pixel-wise loss measures the difference between the predicted and ground truth images at the pixel level. Two common types of pixel-wise loss are L1 loss and L2 loss:

- **L1 Loss (Mean Absolute Error)**: L1 loss calculates the absolute difference between the predicted pixel values and the actual pixel values. It is defined as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BL1%20Loss%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%7C%20%5Chat%7Bx%7D_i%20-%20x_i%20%7C" />
</p>

Where:

<p align="left"> <ul> <li><b>&#x03C6;<sub>i</sub></b> is the predicted pixel value at position <b>i</b>,</li> <li><b>x<sub>i</sub></b> is the ground truth pixel value at position <b>i</b>,</li> <li><b>N</b> is the total number of pixels.</li> </ul> </p>

- **L2 Loss (Mean Squared Error)**: L2 loss computes the squared difference between the predicted and ground truth pixel values. It is defined as:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BL2%20Loss%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28%20%5Chat%7Bx%7D_i%20-%20x_i%20%29%5E2" /> </p>
Where:

<p align="left"> <ul> <li><b>&#x03C6;<sub>i</sub></b> is the predicted pixel value at position <b>i</b>,</li> <li><b>x<sub>i</sub></b> is the ground truth pixel value at position <b>i</b>,</li> <li><b>N</b> is the total number of pixels.</li> </ul> </p>

### 2. Perceptual Loss
Perceptual loss evaluates the quality of the generated image based on high-level feature representations rather than raw pixel differences. This loss is based on the idea that the perceptual quality of an image can be better captured by comparing features extracted from pre-trained deep neural networks (such as VGG) rather than comparing pixel values directly.

The perceptual loss is computed as follows:

Feature Extraction: Pass both the generated image <b>ψ<sub>I</sub></b> and the ground truth image <b>I</b> through a pre-trained feature extractor (e.g., VGG-19) to obtain feature maps. Let <b>F(<b>ψ<sub>I</sub></b>)</b> and <b>F(<b>I</b>)</b> denote the feature maps of <b>ψ<sub>I</sub></b> and <b>I</b>, respectively, at different layers of the network.

Feature Loss Calculation: Compute the loss based on the difference between feature maps. The perceptual loss can be defined as:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BPerceptual%20Loss%7D%20%3D%20%5Csum_%7Bl%7D%20%5Cfrac%7B1%7D%7BN_l%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN_l%7D%20%7C%7C%20F_l%28%5Chat%7BI%7D%29_i%20-%20F_l%28I%29_i%20%7C%7C_%7B2%7D%5E2" /> </p>
Where:

<p align="left"> <ul> <li><b>F<sub>l</sub>(&#x03C8;<sub>I</sub>)</b> and <b>F<sub>l</sub>(I)</b> are the feature maps at layer <b>l</b> for the predicted and ground truth images,</li> <li><b>N<sub>l</sub></b> is the number of elements (e.g., pixels or activations) in the feature map at layer <b>l</b>,</li> <li><b>&#x2225;&middot;&#x2225;<sub>2</sub></b> denotes the squared L2 norm.</li> </ul> </p>
The perceptual loss helps ensure that high-level features and textures are preserved in the generated image.

### 3. Combined Loss Function
To balance the pixel-wise accuracy and perceptual quality, the combined loss function is formulated as a weighted sum of the pixel-wise loss and perceptual loss:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BTotal%20Loss%7D%20%3D%20%5Calpha%20%5Ccdot%20%5Ctext%7BPixel-Wise%20Loss%7D%20%2B%20%5Cbeta%20%5Ccdot%20%5Ctext%7BPerceptual%20Loss%7D" /> </p>
Where:

<ul> <li><b>&#x03B1;</b> and <b>&#x03B2;</b> are weights that control the importance of pixel-wise loss and perceptual loss, respectively.</li> </ul>

### Comparison with Traditional Loss Functions
Traditional loss functions like L1 and L2 focus solely on pixel-wise differences. While effective for training, they may not capture perceptual quality or high-level features effectively. Combining these with perceptual loss enables the model to achieve better image quality by preserving textures and details, which might be missed by pixel-wise loss alone.

<p> In your project, this combination allows for both fine-grained pixel accuracy and high-level perceptual quality, leading to improved results in tasks like image dehazing. </p> <p> Feel free to adjust the weight values <b>&#x03B1;</b> and <b>&#x03B2;</b> based on the specific needs of your model and dataset. </p>

### 1. Channel Attention (CA)
Channel Attention focuses on enhancing or suppressing the significance of different channels in feature maps. This mechanism is designed to emphasize the most informative channels while reducing the less relevant ones.

**Formula and Computation:**

**Global Average Pooling:** Compute the global average pooling across spatial dimensions for each channel to obtain a channel-wise descriptor.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?z_c%20%3D%20%5Cfrac%7B1%7D%7BH%20%5Ctimes%20W%7D%20%5Csum_%7Bh%3D1%7D%5E%7BH%7D%20%5Csum_%7Bw%3D1%7D%5E%7BW%7D%20x_%7Bc%2Ch%2Cw%7D" />
</p>

Where:
<ul>
  <li><b>H</b> and <b>W</b> are the height and width of the feature map,</li>
  <li><b>x<sub>c,h,w</sub></b> is the feature value at channel <b>c</b> and spatial position <b>(h,w)</b>,</li>
  <li><b>z<sub>c</sub></b> is the channel descriptor for channel <b>c</b>.</li>
</ul>

**Fully Connected Layers:** Pass the channel-wise descriptors through a small neural network (often consisting of fully connected layers) to compute channel-wise attention weights.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?s_c%20%3D%20%5Csigma%28W_2%20%5Ccdot%20%5Ctext%7BReLU%7D%28W_1%20%5Ccdot%20z_c%20%2B%20b_1%29%20%2B%20b_2%29" />
</p>

Where:
<ul>
  <li><b>W<sub>1</sub></b> and <b>W<sub>2</sub></b> are weights,</li>
  <li><b>b<sub>1</sub></b> and <b>b<sub>2</sub></b> are biases, and</li>
  <li><b>&#x03C3;</b> is the sigmoid activation function.</li>
</ul>

**Reweighting:** Multiply the original feature map by the computed attention weights.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?x_%7Bc%2Ch%2Cw%7D%20%5Ctilde%20%3D%20s_c%20%5Ccdot%20x_%7Bc%2Ch%2Cw%7D" />
</p>

Where:
<ul>
  <li><b>x<sub>c,h,w</sub><sup>&#x223C;</sup></b> is the reweighted feature map.</li>
</ul>

**Comparison with Traditional Methods:**
In traditional convolutional neural networks (CNNs) without channel attention, all channels are treated equally. Channel Attention introduces a mechanism to dynamically adjust the importance of each channel, thereby improving the network's ability to focus on relevant features. Traditional methods do not have this adaptive reweighting mechanism, which can limit their ability to capture complex feature dependencies.

### 2. Progressive Attention (PA)
Progressive Attention involves adjusting attention weights based on the importance of features at various stages in the network. This mechanism is designed to adaptively focus on different regions or features over multiple stages of processing.

**Formula and Computation:**

**Attention Calculation:** Compute the attention weights for each stage based on the feature maps at that stage.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Calpha_%7Bi%2Cj%7D%20%3D%20%5Cfrac%7B%5Cexp%28e_%7Bi%2Cj%7D%29%7D%7B%5Csum_%7Bk%7D%20%5Cexp%28e_%7Bi%2Ck%7D%29%7D" />
</p>

Where:
<ul>
  <li><b>e<sub>i,j</sub></b> is the attention score for the feature at position <b>(i,j)</b>,</li>
  <li><b>&#x03B1;<sub>i,j</sub></b> is the attention weight.</li>
</ul>

**Feature Aggregation:** Aggregate features based on attention weights.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?f_%7Bi%7D%20%5Ctilde%20%3D%20%5Csum_%7Bj%7D%20%5Calpha_%7Bi%2Cj%7D%20%5Ccdot%20f_j" />
</p>

Where:
<ul>
  <li><b>f<sub>i</sub><sup>&#x223C;</sup></b> is the aggregated feature for position <b>i</b>,</li>
  <li><b>f<sub>j</sub></b> is the feature at position <b>j</b>.</li>
</ul>

**Comparison with Traditional Methods:**
Traditional attention mechanisms typically compute attention weights once and apply them throughout the network. Progressive Attention refines these weights iteratively, allowing for dynamic adjustment and better feature focusing. This progressive refinement can lead to improved performance in tasks requiring fine-grained attention.

### 3. Efficient Attention (EA)
Efficient Attention aims to reduce the computational complexity of the attention mechanism while maintaining performance. It introduces optimizations to make attention calculations more efficient.

**Formula and Computation:**

**Approximate Attention Calculation:** Instead of computing exact attention weights, EA uses approximations to reduce computational costs.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BAttention%7D%28Q%2CK%2CV%29%20%3D%20%5Ctext%7BSoftmax%7D%28%5Cfrac%7BQK%5ET%7D%7Bd_k%7D%29V" />
</p>

can be approximated by:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BAttention%7D%28Q%2CK%2CV%29%20%5Capprox%20%5Ctext%7BSoftmax%7D%28%5Cfrac%7BQ%7D%7B%5E%7BK%7D%7D%7B%7Bd_k%7D%29V" />
</p>

Where:
<ul>
  <li><b>K~</b> is an approximation of <b>K</b>.</li>
</ul>

**Sparse Attention:** Use sparse matrices to approximate the full attention matrix, reducing the number of computations required.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BSparseAttention%7D%28Q%2CK%2CV%29%20%3D%20%5Ctext%7BSoftmax%7D%28%5Cfrac%7BQK_s%5ET%7D%7Bd_k%7D%29V" />
</p>

Where:
<ul>
  <li><b>K_s</b> is a sparse approximation of <b>K</b>.</li>
</ul>

**Comparison with Traditional Methods:**
Traditional attention mechanisms require computing the full attention matrix, leading to high computational complexity. Efficient Attention reduces this complexity by using approximations or sparse representations, making it more scalable to large datasets and models.

### 4. Progressive Attention Refinement (PAR)
Progressive Attention Refinement improves feature extraction by refining attention maps at multiple levels or stages.

**Formula and Computation:**

**Initial Attention:** Compute initial attention maps similar to traditional attention mechanisms.

**Refinement Stages:** Iteratively refine attention maps using additional attention mechanisms or layers.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?A%28t%2B1%29%20%3D%20%5Ctext%7BRefine%7D%28A%28t%29%29" />
</p>

Where:
<ul>
  <li><b>A(t)</b> is the attention map at stage <b>t</b>,</li>
  <li><b>Refine</b> represents the refinement process.</li>
</ul>

**Feature Integration:** Combine features using refined attention maps.

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?x_%7Bi%7D%20%5Ctilde%20%3D%20%5Csum_%7Bj%7D%20A_%7Bi%2Cj%7D%28T%29%20%5Ccdot%20x_j" />
</p>

Where:
<ul>
  <li><b>A<sub>i,j</sub>(T)</b> is the refined attention weight,</li>
  <li><b>x<sub>j</sub></b> is the feature at position <b>j</b>.</li>
</ul>

**Comparison with Traditional Methods:**
Traditional attention mechanisms do not adaptively refine attention weights over multiple stages. Progressive Attention Refinement iteratively adjusts these weights, potentially leading to better feature extraction and model performance.

### Summary Table

<table border="1">
  <thead>
    <tr>
      <th>Attention Mechanism</th>
      <th>Advantages</th>
      <th>Limitations</th>
      <th>Comparison with Traditional Methods</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Channel Attention</td>
      <td>Enhances relevant channels, improves feature discrimination.</td>
      <td>May require additional computations for channel-wise descriptors.</td>
      <td>Traditional methods treat all channels equally, lacking adaptive focus.</td>
    </tr>
    <tr>
      <td>Progressive Attention</td>
      <td>Refines attention weights over multiple stages, improves feature focus.</td>
      <td>More complex to implement, potentially higher computational cost.</td>
      <td>Traditional methods use fixed attention weights throughout the network.</td>
    </tr>
    <tr>
      <td>Efficient Attention</td>
      <td>Reduces computational complexity, scalable to larger models.</td>
      <td>Approximation might affect performance in some cases.</td>
      <td>Traditional methods require full attention matrix computation, which is more computationally intensive.</td>
    </tr>
    <tr>
      <td>Progressive Attention Refinement</td>
      <td>Iteratively refines attention maps, potentially better feature extraction.</td>
      <td>Complex implementation, may involve higher computation in refinement stages.</td>
      <td>Traditional attention mechanisms do not refine weights iteratively, potentially limiting performance.</td>
    </tr>
  </tbody>
</table>

### Differences from Original FFA-Net

The original FFA-Net model includes 12 blocks per group. In this modified version, the architecture uses 5 blocks per group to reduce computational requirements. Despite this reduction, the modified model achieves comparable SSIM values and close PSNR values to the original model.

### Results

**Training Loss:**
<ul>
  <li>Step 5000: Loss = 0.14956 | SSIM = 0.8106 | PSNR = 17.9430</li>
  <li>Step 10000: Loss = 0.02472 | SSIM = 0.8500 | PSNR = 20.0667</li>
  <li>Step 15000: Loss = 0.07915 | SSIM = 0.8584 | PSNR = 20.3824</li>
  <li>Step 20000: Loss = 0.09964 | SSIM = 0.8501 | PSNR = 20.7447</li>
</ul>

The results may appear lower than those reported in the original paper due to the reduced number of blocks and computational constraints. However, the modified model still achieves competitive performance, particularly in SSIM, and maintains close PSNR values.



### Getting Started

1. **Download the Notebook:** Obtain the [FFA_NET_with_EA_and_PR.ipynb](path/to/FFA_NET_with_EA_and_PR.ipynb) from this repository.
2. **Download Trained Weights:** Download [FFA_model.pth](path/to/FFA_model.pth) for pretrained model parameters.
3. **Run the Notebook:** Follow the instructions within the notebook to run the model and test its performance.

For additional details or to run the notebook, please visit my Kaggle page.

### Acknowledgements

- Kaggle for providing the computational resources.
- The original authors of FFA-Net for their foundational work.

For any questions or issues, please contact Pranav Balaji R S at pranavbalajirs@gmail.com
