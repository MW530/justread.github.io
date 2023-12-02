---
title: >-
  Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis
tags:
  - IFD
  - denoising
categories: IFD
thumbnail: /images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/fig.4.png
journal: IEEE Transactions on Industrial Informatics(IF:12.3)
date: 2023-12-02 15:18:22
---

# 引言

1. 旋转机械对于工业很重要，但是由于工作环境恶劣，容易损坏。因此对其进行准确的诊断非常重要。

2. 故障诊断的方法主要分为两类：

   1. **基于信号分析的方法**
   2. **以机器学习为驱动的方法**

   一般来说，基于信号分析的故障诊断方法通过检测与故障相关的振动成分或特征频率来识别故障。然而，对于大型旋转机器，振动信号通常由许多不同的振动分量组成，包括齿轮的啮合以及轴和轴承的旋转。因此，传统的基于信号分析的故障诊断方法往往难以确定与故障相关的振动成分和特征频率。

3. **机器学习驱动的故障诊断方法**能够在不识别故障相关组件和特征频率的情况下诊断故障。然而，提取的统计参数（峰度、均方根、能量和熵）往往没有足够的判别能力来区分故障，从而导致诊断准确率较低。因此，在机器学习驱动的故障诊断中，寻找具有区分性的特征集成为一项长期挑战。

4. 深度学习方法被用来进行故障诊断。为了取代传统的统计参数，深度学习方法可自动学习原始振动信号的特征，从而提高诊断准确性。

5. 介绍了残差网络及其在故障诊断中的应用。

6. 从大型旋转机器收集到的振动信号通常包含大量噪声。作为局部特征提取器，ResNets 中使用的卷积核可能会因噪声干扰而无法检测到与故障相关的特征。因此，有必要开发新的深度学习方法，用于在强背景噪声下对旋转机械进行基于振动的故障诊断。

7. 本文开发了两种深度残差收缩网络（DRSN），即具有信道共享阈值的 DRSN（DRSN-CS）和具有信道明智阈值的 DRSN（DRSNCW），以提高 ResNets 对高噪声振动信号的特征学习能力，最终实现高诊断准确性的目标。主要贡献概述如下。

   1. 软阈值处理（即一种流行的收缩函数）作为非线性转换层被插入到深度结构中，以有效消除与噪声相关的特征。
   2. 阈值是通过专门设计的子网络自适应确定的，因此每个振动信号都有自己的一套阈值。
   3. 软阈值处理中考虑了两种阈值，即信道共享阈值和通道阈值，这也是 DRSN-CS 和 DRSN-CW 这两个术语的由来。

# 相关工作

## ResNet

## 软阈值

在软阈值处理方法中，原始信号一般会被转换到一个近零数字并不重要的域，然后应用软阈值技术将近乎零的特征转换为零。

例如，作为一种经典的信号去噪方法，小波阈值通常由三个步骤组成：小波分解、软阈值和小波重构。

为了确保信号去噪的良好性能，小波阈值处理的一项关键任务是设计一个滤波器，**该滤波器可以将有用信息转换为非常正或负的特征，将噪声信息转换为接近零的特征**。然而，设计这样的滤波器需要很多信号处理方面的专业知识，一直是一个具有挑战性的问题。

软阈值的功能可表示如下：

$\left.y=\left\{\begin{array}{ll}x-\tau&x>\tau\\0&-\tau\leq x\leq\tau\\x+\tau&x<-\tau\end{array}\right.\right.$

其中，x 是输入特征，y 是输出特性，τ 是阈值，即一个正参数。在 ReLU 激活函数中，软阈值不是将负特征设为零，而是将接近零的特征设为零，从而保留有用的负特征。

软阈值处理过程如图 3(a) 所示。从图 3(b) 可以看出，输出对输入的导数要么为一，要么为零，这可以有效防止梯度消失和爆炸问题。导数可表示如下：

$\left.\frac{\partial y}{\partial x}=\left\{\begin{array}{ll}1&x>\tau\\0&-\tau\leq x\leq\tau\\1&x<-\tau\end{array}\right.\right..$

![fig.3](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/fig.3.png)

# 提出的方法

![fig.4](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/fig.4.png)



## DRSN-CS

所提出的 DRSN-CS 是 ResNet 的一种变体，它使用软阈值去除与噪声相关的特征。

软阈值作为非线性变换层插入到构建单元中。

此外，阈值可以在构建单元中学习，下面将对此进行介绍。

如图 4(a)所示，名为 "具有通道共享阈值的残差收缩构建单元（RSBU-CS）"的构建单元与图 2(a)中的 RBU 不同，RSBU-CS 有一个特殊模块，**用于估计软阈值处理中使用的阈值**。在该特殊模块中，GAP 应用于特征图 x 的绝对值，得到一个 1-D 向量。然后，将 1-D 向量传播到双层 FC 网络中，得到一个缩放参数，该参数与 [25] 中提出的参数类似。然后，在两层 FC 网络的末端应用一个 sigmoid 函数，从而将缩放参数缩放至（0，1）范围内，可表示如下：

$\alpha=\frac{1}{1+e^{-z}}$

其中，z 是 RSBUCS 中双层 FC 网络的输出，α 是相应的缩放参数。**然后，将缩放参数 α 乘以 |x| 的平均值，得到阈值。**之所以这样安排，是因为软阈值的阈值不仅要为正，而且不能太大。如果阈值大于特征图的最大绝对值，那么软阈值的输出将是零。总之，RSBU-CS 使用的阈值表示如下：

$\tau=\alpha\cdot\underset{\begin{array}{c}i,j,c\\\end{array}}{\operatorname*{average}}|x_{i,j,c}|$

其中，τ 是阈值，i、j 和 c 分别是特征图 x 的宽度、高度和通道的索引。阈值可以保持在一个合理的范围内，这样软阈值处理的输出就不会全为零。与图 2(b) 和 (c) 中的 RBU 类似，可以构建跨距为 2、通道数加倍的 RSBU-CS。

所提出的 DRSN-CS 的简要架构如图 4（b）所示，与经典 ResNet 相似。唯一不同的是，RSBU-CS 被用作构建单元，而不是 RBU。在 DRSN-CS 中堆叠了多个 RSBU-CS，因此可以逐渐减少与噪声相关的特征。所开发的 DRSN-CS 的另一个优点是，阈值是在深度架构中自动学习的，而不是由专家手动设置的，因此在实施所开发的 DRSN-CS 时不需要信号处理方面的专业知识。

## DRSN-CW

所提出的 DRSN-CW 是 ResNet 的另一种变体，与 DRSN-CS 不同的是，**它对特征图的每个通道都应用了单独的阈值**，下面将对此进行介绍。带通道阈值的 RSBU（RSBU-CW）如图 4（c）所示。使用绝对运算和 GAP 层将特征图 x 简化为一维向量，然后传播到双层 FC 网络中。FC 网络的第二层有一个以上的神经元，神经元的数量等于输入特征图的通道数。FC 网络的输出用以下方法缩放到 (0, 1) 的范围内

$\alpha_c=\frac{1}{1+e^{-z}}$

其中，zc 是第 c 个神经元的特征，αc 是第 c 个缩放参数。之后，阈值的计算方法如下：

$\tau_c=\alpha\cdot\underset{\begin{array}{c}i,j,c\\\end{array}}{\operatorname*{average}}|x_{i,j,c}|$

其中τc是特征图的第cth通道的阈值，i、j和c分别是特征图x的宽度、高度和通道的指数。与DRSN-CS类似，阈值可以是正的并保持在合理的范围内，从而防止输出特征全为零。

所开发的 DRSN-CW 的整体架构如图 4（d）所示。多个 RSBU-CW 堆叠在一起，这样就可以通过各种非线性变换和软阈值作为收缩函数来学习辨别特征，从而消除与噪声相关的信息。

# 实验

## 数据集

自建

![table.1](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/table.1.png)



## 对比方法

1. CNN
2. ResNet
3. DRSN-CS（自己的）
4. DRSN-CW（自己的）



## 结果

![table.3](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/table.3.png)

![table.4](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/table.4.png)

![table.5](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/table.5.png)

![fig.9](/images/Deep-Residual-Shrinkage-Networks-for-Fault-Diagnosis/fig.9.png)

# 总结

本文提出了一种使用自适应阈值来替换Relu层，以消除噪音的方法。并且给出了两种变体，分别是所有通道用一个阈值和不同通道用不同阈值。
