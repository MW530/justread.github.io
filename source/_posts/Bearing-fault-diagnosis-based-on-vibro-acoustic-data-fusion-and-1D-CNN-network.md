---
title: Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network
tags:
  - IFD
  - multi-modal
categories: IFD
thumbnail: /images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.2.png
journal: Measurement(IF:5.6)
date: 2023-11-23 19:43:50
---

# 引言

1. 轴承是旋转机械的重要组成部分，也是最容易损坏的部件之一。因此，轴承故障诊断是旋转机械维修的一个组成部分，在过去几十年中受到了广泛的关注。

2. 分别介绍了基于振动信号和基于声学信号的故障诊断方法。

3. 轴承故障诊断算法的核心在于信号特征提取和模式识别。然后分别列举了一些信号特征提取和模式识别的方法。

   1. 信号特征提取：
      1. FFT
      2. wavelet transform
      3. EMD
      4. VMD
   2. 模式识别
      1. SVM
      2. BP neural networks
      3. Bayesian classifiers
      4. nearest neighbor classifiers

4. 介绍了基于深度学习的故障诊断方法。深度学习不需要手动提取，从根本上摆脱了对人类干预和专家知识的依赖。

5. 介绍了CNN网络以及在故障诊断中的应用。

6. 然而，目前大多数轴承故障诊断方法，包括传统方法和深度学习方法，本质上都是基于单模态测量，而使用单模态传感器（加速度计或麦克风）进行故障诊断往往无法考虑到故障的复杂性。

7. 介绍了传统多模态传感器融合方法，以及目前的缺点。

   > 在传统的多模态传感器融合方法中，多模态传感器特征是手动提取的，并简单地连接成一个长向量来实现数据融合。然而，这些研究仍处于初步阶段，在检查诊断方法的有效性时没有考虑不同的噪声环境。

8. 基于这些先前的研究和故障诊断的最新进展，本文提出了一种用于轴承故障诊断的深度学习多模态传感器融合方法，称为基于1D CNN的振声传感器数据融合（1D-CNN-based vibro-acoustic sensor data fusion, VAF）算法。在该算法中，对加速度计和麦克风这两个不同模态传感器同时采集的信号进行处理，并在基于1D CNN的特征提取阶段提取信号的特征。然后在融合阶段对提取的特征进行融合，并最终由softmax分类器进行分类，以确定轴承故障的类型。



# 提出的方法

## 提出的1D-CNN

基于CNN的基本原理，本文提出了一种新的1D-CNN结构，用于轴承故障信号的特征提取和分类。

在1D-CNN算法中，归一化后的原始数据直接导入1D-CNN。1D-CNN具有强大的特征提取能力：1D-CNN中的交替卷积层和池化层可以自动提取隐藏在原始数据中的非线性特征，自适应特征学习在全连接层完成。通过这种方式，1D-CNN算法消除了传统算法中的手动特征提取过程，实现了端到端的信息处理。

图1显示了本文提出的1D-CNN的具体结构。它由五个卷积层、五个池化层、一个全连接层和一个softmax层组成。在经过第一个卷积层之后，信号被转换成一组特征图，然后通过最大池对其进行下采样。这些先前的操作重复四次，以将最后一个池化层的特征连接到完全连接层，然后该完全连接层将由ReLU功能激活并转移到softmax层。

该模型有五个卷积层和池化层。卷积核的大小在第1层为64×1，在第2层和第3层为32×1，而在第4层和第5层为16×1。池化内核的大小在第1层为16×1，在第2、3、4和5层为2×1。节点号在fullyconnected层为100，在softmax层有10个输出，对应于实验中轴承故障的10种状态。

![fig.1](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.1.png)

## 提出的基于1D-CNN VAF算法 

本文提出了一种基于1DCNN的声振传感器数据融合算法（1DCNN-based VAF）用于轴承故障诊断。该算法以声学振动信号的融合数据集为输入，对这些信号进行智能分类，识别轴承故障。

具体来说，基于1D CNN的VAF算法主要分为三个阶段：

1. 多模态传感器特征提取阶段：使用5层卷积池化结构提取振动和声学特征，通过该结构对振动和声学信号分别进行5次卷积和池化。
   1. 振动信号特征提取
   2. 声学信号特征提取
2. 融合阶段：通过将所有提取的振动和声学特征馈送到完全连接的层中来实现的，在该层中实现特征融合的过程。
3. 分类阶段：通过将所有提取的振动和声学特征馈送到完全连接的层中来实现的，在该层中实现特征融合的过程。

图2展示了基于1D细胞神经网络的VAF算法的整个结构。

![fig.2](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.2.png)



# 实验

## 数据集

![fig.5](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.5.png)

![table.6](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/table.6.png)

## 实验结果

![table.7](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/table.7.png)

![fig.9](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.9.png)

![fig.8](/images/Bearing-fault-diagnosis-based-on-vibro-acoustic-data-fusion-and-1D-CNN-network/fig.8.png)



# 总结

1. 本文的创新点在于融合了多模态信号。模型上的创新仅仅是应用。
2. 但是本文的多模态数据是声音和震动，是不是换成热成像效果会更好。因为本质上声音和震动信号都是震动信号。







