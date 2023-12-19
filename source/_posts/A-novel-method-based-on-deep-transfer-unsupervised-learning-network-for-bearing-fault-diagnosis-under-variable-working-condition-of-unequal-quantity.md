---
title: >-
  A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity
tags:
  - IFD
  - DBN
categories: IFD
thumbnail: /images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.2.png
journal: Knowledge-Based Systems(IF:8.8)
date: 2023-12-19 22:55:17
---

# 创新点

1. 利用 DCDBN 强大的特征学习能力和 DMLP 的自适应特征分类能力，提出了一种新的 DCDBN-DMLP 模型，用于识别变化工况下的轴承故障。稀释卷积使得 CDBN 的视野更加开阔，同时可以控制参数的数量。借助扩张卷积的优势，DCDBN 可以详细提取重要的可转移特征，而无需分析杂乱无章的特征。
2. 与其他现有方法相比，所提出的模型具有较强的域适应能力，在不同故障严重程度、载荷和故障类型下都能达到较高的精度。此外，该模型还能深度利用源域和目标域之间的分布，不仅在单工况向多工况的转移任务中表现出优势，在多工况向单工况的转移任务中也表现出优势。



# 方法

## CNDBN

CDBN 是一种无监督分层生成网络模型，可以自适应地从原始数据中提取高级特征。CDBN 结合了 CNN 和 DBN 的优点，具有翻译不变和权重共享的特点。与 DBN 相似，CDBN 由多个卷积受限波尔兹曼机（multiple convolutional restricted Boltzmann machine, CRBM）构建。



高斯 CDBN 是在 CDBN 的基础上，使用高斯单元设计可见层。CDBN 由三层组成：可见层 V、隐藏层 H 和池化层 P。本文使用空洞卷积来修改卷积算子，它可以在不损失分辨率的情况下系统地聚合多尺度信息。如图 1 所示，可见层由 NV ×NV 的高斯单元阵列组成，隐藏层由 K。每组是一个 NH × NH 的二进制单元阵列，共有 N2 个 HK 隐藏单元。组与组之间有一个 NW × NW（NW = NV - NH + 1）滤波器，该滤波器与组内的所有隐藏单元共享。池化层中每个组的大小为 NP × NP，每个隐藏组 Hk 被划分成小块，由一个 C × C 阵列组成。隐藏层最后与池化层相连。



高斯 CDBN 的能量函数可以写成：

![Eq.1](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/Eq.1.png)

![fig.1](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.1.png)

![table.1](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/table.1.png)



在网络中加入空洞卷积，公式 (1) 可以修正如下：

![eq.2](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/eq.2.png)



## Maximum Mean Discrepancy (MMD)



## 提出的方法

![fig.2](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.2.png)



### 多层可迁移特征差异

MMD 被用于减少从跨领域数据中获得的转移特征的分布差异。为了提高域适应能力，可转移特征的分布差异不仅存在于全连接层（F1 至 Fi-1），还存在于隐藏层 H1 和 H2。因此，在**无监督 CDBN 的训练过程中，调整所有层的分布非常重要**。因此，多层可转移特征差异的计算方法如下。

$\begin{aligned}
MMD_{\mathbf{k}}(\mathcal{Z}^{\mathcal{L},s},\mathcal{Z}^{\mathcal{L},t})& =\sum_{l\in\mathcal{L}}\kappa_l(\frac1{n^2}\sum_{i=1}^n\sum_{j=1}^nk(x_i^s,x_j^s)  \\
&-\frac2{mn}\sum_{i=1}^n\sum_{j=1}^mk(x_i^s,y_j^t) \\
&+\frac1{m^2}\sum_{i=1}^m\sum_{j=1}^mk(y_i^t,y_j^t))
\end{aligned}$

其中，ZL,s 和 ZL,t 表示源域和目标域的可转移特征，L = {H1, H2, F1, ., Fi-1} 为特征提取层。由于多层 MMD 值的变化，采用了一个权衡参数 κl 来调整多层 MMD 的大小。

### 伪标签技术

引入伪标签技术是为了解决目标域中未标记样本无法用于训练共享分类器参数的问题[32]。本文中，MLP 的最后一层分类器是全连接层 Fi。利用 softmax 函数预测目标域对应样本中标签的概率分布。



输入样本为 X = (x1, x2, ... , xi)，相应的标签为 Y = (y1, y2, ... , yi)，因此将呈现 softmax 函数的输出结果：

$\left.O=\left[\begin{array}{c}p(y_i=1|x_i;\theta)\\p(y_i=2|x_i;\theta)\\\vdots\\p(y_i=k|x_i;\theta)\end{array}\right.\right]=\left[\begin{array}{c}e^{\theta_2x_i}\\e^{\theta_2x_i}\\\vdots\\e^{\theta_Nx_i}\end{array}\right]/\sum_{j=1}^Ne^{\theta_jx_i}$

其中，p(yi = k|xi; θ ) 表示输入样本的概率，N 表示类别数，θ 指最后一个分类器层的参数。分类器层包含权重和偏置。鉴于 softmax 分类器的特性，输出为正且总和为 1。



因此，目标域的伪标签生成如下。

![eq.12](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/eq.12.png)



其中，yj 是第 j 个伪标签。



# 实验

![fig.3](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.3.png)

## 数据集

1. Case Western Reserve University
2. 自建



## 对比方法

1. DBN
2. TCA
3. DDC
4. DCBN-DMLP



## 结果

![fig.7](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.7.png)

![fig.8](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.8.png)

![fig.9](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/fig.9.png)

![table.3](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/table.3.png)

![table.4](/images/A-novel-method-based-on-deep-transfer-unsupervised-learning-network-for-bearing-fault-diagnosis-under-variable-working-condition-of-unequal-quantity/table.4.png)

# 总结

1. 多层MMD可以借鉴
