---
title: 'Top models for image classification'
description: "Explore the top neural network models in computer vision 👾, including EfficientNet, YOLOv8, ViT, and OmniVec. Learn how each excels in tasks like image classification, real-time object detection, and multi-modal learning."
discover:
    description: "Explore the top neural network models in computer vision, including EfficientNet, YOLOv8, ViT, and OmniVec. Learn how each excels in tasks like image classification, real-time object detection, and multi-modal learning."
date: 2024-10-11
---

Deep learning models have received a lot of attention in recent years. Tasks such as fraud detection, natural language processing and self-driving vehicles are hard to imagine without the help of neural networks. Today, I want to take a look at one of these task, specifically image classification. Here is my overview of the top image classification models.

Image classification is one of the simplest tasks in computer vision. The model has to classify an entire image into one of a set of predefined classes. Despite the simplicity of the task, as the number of classes grows, the complexity of the task increases drastically. 

#### Evaluation Metrics for Classification

Classification models are evaluated using top-1 and top-5 accuracy. In top-1 accuracy, the model predicts correctly if the output class is the same as the true label. In top-5 accuracy, the model predicts correctly if one of the five predicted labels is the true one.

## EfficientNet

<table>
  <tr>
    <th>Name</th>
    <td>EfficientNet</td>
  </tr>
  <tr>
    <th>Developer</th>
    <td>Google</td>
  </tr>
  <tr>
    <th>Parameters</th>
    <td>from 5.3 million (B0) to 66 million (B7)</td>
  </tr>
  <tr>
    <th>Tasks</th>
    <td>image classification</td>
  </tr>
  <tr>
	  <th>Top-1 ImageNet</th>
	  <td>84.3% with 66M parameters</td>
  </tr>
</table>

[EfficientNet](https://arxiv.org/abs/1905.11946) is a family of convolutional neural networks (CNNs) that efficiently scale up in terms of layer depth, layer width, input resolution, or a combination of all these factors. 

### EfficientNet Architecture

- **Conv** (convolutional layer) the first layer of the model. It applies a convolution operation to the input, passing information through filters (or kernels) to detect features like edges, textures, and shapes in an image. Each convolutional layer generates a feature map that is passed to subsequent layers for further abstraction.
- **MBConv** (mobile inverted bottleneck convolution)  the main building block of EfficientNet, to which authors also added squeeze-and-excitation (SE) optimization. MBConv was originally used in MobileNet model. It is based on an inverted residual structure, where the shortcut connections are between the thin bottleneck layers. The inverted residual with linear bottleneck module takes an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear  convolution. The squeeze-and-excitation block adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
- **Conv, Pooling, FC**  the last block consists of a convolutional layer, pooling layer for downsampling the spatial dimensions of feature maps (reducing the number of parameters and computations), and fully connected (FC) layer that flattens the learned feature maps and makes prediction based on them.

##  YOLO (You Only Look Once)

<table>
  <tr>
    <th>Name</th>
    <td>YOLOv8</td>
  </tr>
  <tr>
    <th>Developer</th>
    <td>Ultralytics</td>
  </tr>
  <tr>
    <th>Parameters</th>
    <td>from 3.2 million (Nano) to 68.2 million (Extra Large)</td>
  </tr>
  <tr>
    <th>Tasks</th>
    <td>object detection, image segmentation, image classification</td>
  </tr>
  <tr>
	  <th>Top-1 ImageNet</th>
	  <td>79.0% with 57.4M parameters</td>
  </tr>
</table>

The [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-cls.yaml) image classification model is designed to detect 1,000 pre-defined classes in images in real-time. YOLOv8 is trained on ImageNet dataset with an image resolution of 224x224. The model is higly optimized for real-time object detection. For instance, the YOLOv8n model achieves a mean Average Precision (mAP) of 37.3 on the COCO dataset and a speed of 0.99 ms on A100 TensorRT. 

### YOLOv8 Architecture

the authors of YOLOv8 divide the model's architecture into three main components

 - **Backbone** the convolutional neural network responsible for extracting features from the input image. YOLOv8 uses a custom CSPDarknet53 CNN, originally employed as the backbone for YOLOv4. It uses a CSPNet strategy to partition the feature map of the base layer into two parts, and then merges them through a cross-stage hierarchy. The use of a split and merge strategy allows for more gradient flow through the network. Darknet-53 is a CNN that acts as a backbone for the YOLOv3. Improvements upon predecessor Darknet-19 include the use of residual connections, as well as more layers. Darknet-19 is a CNN that used as the backbone of YOLOv2. Similar to the VGG models, it mostly uses 3x3 filters and doubles the number of channels after every pooling step.
 - **Neck** also known as the feature extractor, the neck merges feature maps from different stages of the backbone to capture information at various scales. YOLOv8 uses a novel C2f module instead of the traditional Feature Pyramid Network (FPN). The C2f module is a faster implementation of the C2 mode. The C2 module in YOLOv8 stands from CSP (Cross Stage Partial) Bottleneck with 2 convolutions.
 - **Head** is responsible for making predictions, which are then aggregated to obtain the final detections.
 

## ViT (Vision transformer)


<table>
  <tr>
    <th>Name</th>
    <td>ViT</td>
  </tr>
  <tr>
    <th>Developer</th>
    <td>Google</td>
  </tr>
  <tr>
    <th>Parameters</th>
    <td>from 86 million (Base) to 632 million (Huge)</td>
  </tr>
  <tr>
    <th>Tasks</th>
    <td>image classification</td>
  </tr>
  <tr>
	  <th>Top-1 ImageNet</th>
	  <td>88.55% with 632M parameters</td>
  </tr>
</table>

The [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) is a pure transformer model applied to sequences of image patches, performing very well on image classification tasks. In this model, an image split into patches and provided these sequence of linear embeddings of these patches as an input to a transformer. Image patches are treated the same way as tokens (words) in an NLP application. In designing the ViT model, the authors closely followed the original Transformer model [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

Compared to CNNs, ViT is less data-efficient, but has higher capacity. Some of the largest modern computer vision models are ViTs, with models containing up to 22 billion parameters.

### ViT architecture

 - **Linear projection of flattened patches** to handle 2D images, images reshaped into a sequence of flattened 2D patches, which are then linearly projected into embeddings.
 - **Transformer encoder** consists of alternating layers of multi-headed self-attention. Layer normalization (LN) is applied before each block, and residual connection is used after each block to enhance gradient flow.
 -  **MLP head** includes one hidden layer during pre-training phase and a single linear layer during fine-tuning to make the final classification.


## OmniVec

<table>
  <tr>
    <th>Name</th>
    <td>OmniVec</td>
  </tr>
  <tr>
    <th>Developer</th>
    <td>TensorTour</td>
  </tr>
  <tr>
    <th>Parameters</th>
    <td>sizes of encoders + transformer backbone size + prediction head size</td>
  </tr>
  <tr>
    <th>Tasks</th>
    <td>text summarization, audio event classification, video action recognition, image classification</td>
  </tr>
  <tr>
	  <th>Top-1 ImageNet</th>
	  <td>92.4%</td>
  </tr>
</table>

The [OmniVec](https://arxiv.org/abs/2311.05709) network is designed to handle multiple modalities,  such as visual, audio, text and 3D data. First network pre-trained by [self-supervised](https://markogolovko.com/blog/supervised-vs-unsupervised-machine-learning/#self-supervised-learning) masked training, followed by sequential training for the different tasks. For evaluating OmniVec framework, transformer-based modality encoders were used. Image data, for example, was encoded with a ViT. 


###  OmniVec architecture

 - **Modality specific encoders** learn embeddings from different modalities, allowing for cross modality knowledge sharing. Each encoder takes as input one modality at a time and extracts feature embedding for that modality. 
 - **Shared transformer backbone** common part of the framework, maps the input embeddings from various encoders into a shared embedding space. While different modalities pass through different encoders, they are processed by this shared transformer network. OmniVec can utilize any standard transformer architecture for this backbone.
 - **Task specific heads**  independent networks are used for fine-tuning and evaluation, allowing OmniVec to learn and perform specific tasks for each modality. The task heads can support a wide range of tasks across different domains, including computer vision and natural language processing.


## Conclusion

Initially, my goal was to write about the top models for image classification and rank them from best to worst. However, as I explored the leading computer vision models, I realized that such a ranking isn't feasible. Each of the fours neural networks listed here has its own strengths and limitation. 

EfficientNet's main advantage lies in its scalable efficiency, achieving better performance with fewer parameters and lower computational costs compared to traditional CNNS. YOLOv8, on the other hand, excels in real-time object detection and image segmentation. Although ViT requires more data to train and is slower than the previous two models, it surpasses CNNs in image classification task.  Finally, the OmniVec framework demonstrates how ViT can be combined with other modality encoders for multitask learning, showcasing the potential of multimodal neural networks