---
title: >-
  A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network
tags:
  - IFD
categories: IFD
thumbnail: /images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/fig.6.png
journal: Measurement(IF:5.6)
date: 2023-12-02 19:57:54
---

# 引言

1. 滚动轴承是旋转机械的重要部件，因此，精确检测和诊断轴承的状态对于机械系统来说非常必要。
2. 介绍了机器学习方法在故障诊断中的应用。**然而，在大多数情况下，这些方法无法直接从原始信号中捕捉到特征，而且处理海量数据的能力有限。因此，有必要研究能够捕捉原始信号特征并自动识别机器健康状况的诊断方法。**
3. 介绍了深度学习方法在故障诊断中的应用。CNN 在从复杂的二维图像中捕捉信息方面显示出优势，这引起了许多学者的关注。然而，DL 方法也有许多缺点，现总结如下：
   1. 大多数深度学习方法，如 SAE 和 DBN，都是通过快速傅立叶变换（FFT）或离散小波变换（DWT）将原始信号转换为频域信号，并通过 CSC 将原始信号转换为双频域信号。此外，转换理论的选择将取决于专家的经验。因此，深度学习将考虑输出知识。
   2. 随着精度的提高，模型的结构也变得复杂，从而降低了模型的可解释性。
   3. 为了达到高精度，收集大量数据至关重要。然而，实验数据往往并不丰富。
4. 为解决上述问题，文中提出了一种采用新 STIM 算法的卷积神经网络。
   1. 首先，采用数据扩增法来增加原始数据的数量。
   2. 其次，提出新的 STIM，将原始振动数据交换为二维灰色图像。因此，字符是从原始振动信号中获得的，而不是依靠专家的知识。
   3. 第三，为了从灰色图像中获取深度特征，我们提出了一个由卷积层、池化层、Dropout层、全连接层和 Softmax 层组成的 CNN 模型。
   4. 最后，与其他传统方法和 DL 方法相比，所提出的诊断方法提高了文本的分类能力。

# CNN





# 提出的智能诊断方法

本节提出了一种基于 CNN 框架和 STIM 的新模型，它可以直接捕捉原始时域数据中的特征，提高泛化性能。首先，采用预处理理论 "数据增强 "来增加原始数据的数量。然后，建立 CNN 模型。最后，通过输入来自两个不同测试的灰色图像对模型进行训练。

## 数据扩增

随着训练参数的增加和结构的复杂化，CNN 的诊断精度也会提高。此外，在大多数情况下，模型会因训练样本不足而产生过拟合。为解决上述问题，该方法引入了数据增强操作 [30]。增加训练样本的常用方法有移位、水平/垂直翻转、随机裁剪和颜色抖动。本文采用了移位这一常见操作。数据扩增将通过分离这些重叠的训练样本来获得大量的训练数据。下面的示例解释了数据扩增的过程。当数据长度为 110674 时，每个训练样本的长度为 784，移位步长为 110，可以得到 1000 个训练样本，具体说明如下：

$N'=\frac{N-N_S}{step}+1$

其中，$N′$表示增强后的样本数，$N$ 是原始数据的个数，$N_s$ 是每个样本的个数，步长表示移动步长。整个过程如图 4 所示。

![fig.4](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/fig.4.png)



## 信号到图像映射

![fig.5](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/fig.5.png)

在故障诊断模型中，数据预处理是一个重要环节，可以从振动信号中提取特征。然而，直接处理初始数据是不可能的。本研究提出了一种新型的 STIM，它可以直接处理原始信号，**并将原始时域信号有效地转换为二维图像。**

如图5所示，利用转换理论，初始一维数据依次满足图像的像素。图像的大小为$N×N$，得到了一系列长度为N2的数据。设$L(i) i=1，…，N_2$表示一系列不同的数据。$P(j，k)，j=1，…，N k=1，…，N$，表示二维图像的像素强度；具体描述如下：

$P(j, k)= L((j 1)× N + k)× 255$

像素强度反映了原始数据之间的差异值。根据公式 (5)，每个点的归一化值从 0 到 255，图像的大小取决于信号数据的体积。

## 网络结构介绍

本节将建立一种新型 CNN，用于从输入的二维信号中捕捉字符。所提出的 CNN 由 Conv 1、BN1、Conv 2、BN2、Pool 1、Pool2、Fc1、Fc2 和 softmax 层组成。此外，在两个不同的层中采用了 dropout 操作，以减少过拟合。一个是全连接层，另一个是软最大层。拟议 CNN 模型的基本情况如图 6 所示。

核大小的决定对于从灰色图像中提取特征至关重要。因此，本文中 Conv1 和 Conv2 的内核大小为 5×5，卷积步长为 1。Conv 1 和 Conv 2 分别有 32 和 64 个卷积核。卷积核的增加提高了提取特征的能力。卷积层中采用了零填充法，以保持维度不变，从而保留了上一层的大量信息。池化层的核大小为 2 × 2，卷积步长为 2。因此，地图的特征大小减少了一半。CNN 模型采用全连接层，该层的神经元数量为 1024 个。最后，通过 softmax 层获得分类结果。表 1 列出了各层所涉及的参数和细节。

![fig.6](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/fig.6.png)

![table.1](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/table.1.png)

# 实验

## 数据集

1. Case West Reserve University

对于空载，测试分别进行了一马力、两马力和三马力（1 匹马力=746 瓦）负载的三种实验。每个负载被视为一个数据集，包含十种故障类型，并得到四种数据集（A、B、C、D）。

## 对比方法

1. DFCNN-Chinese Journal of Aeronautics-2020
2. CNN based LeNet-5-TIE-2018

## 实验结果

![table.5](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/table.5.png)

![fig.15](/images/A-new-bearing-fault-diagnosis-method-based-on-signal-to-image-mapping-and-convolutional-neural-network/fig.15.png)

# 总结

信号-图片

两层卷积



