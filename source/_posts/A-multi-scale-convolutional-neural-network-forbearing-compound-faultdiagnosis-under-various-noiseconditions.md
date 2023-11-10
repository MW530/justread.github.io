---
title: >-
  Amulti-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions
tags:
  - IFD
  - CNN
categories: IFD
thumbnail: /images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/fig.2.png
journal: Science China-Technological Sciences (IF:4.6)
date: 2023-11-10 09:57:08
---

# 引言

1. 滚动轴承很重要，复合故障更具有挑战性。
2. 信号处理和人工智能是处理复合故障的两种常用处理方法。
3. 人工智能方法包括机器学习和深度学习方法。并且同时介绍了机器学习和深度学习方法。
   1. 然而，由于实际工作场景复杂多变，轴承振动信号很容易被噪声污染，这使得特征提取成为一个巨大的挑战。同时，与单一断层不同，复合断层特征难以准确提取和定位，这给断层分类的任务带来了极大的困难。
   2. 提出了一种全新的抗噪声多尺度卷积神经网络以克服在不同强度噪声水平下进行复杂故障诊断的挑战。
4. 本文创新点如下：
   1. 本文将CNN和残差学习理论相结合，提出了一种在噪声工作条件下有效提取去噪信息的剩余预处理块。然后，我们设计了一个误差损失函数来更新反向传播过程中的块参数，目的是从不同灵敏度的噪声中获得干净的输入。
   2. 随着模型性能的提高，在多尺度学习的基础上，应用多尺度卷积块从振动信号中提取多尺度特征。
   3. 提出的AM-CNN是一种获得领域不变特征的端到端智能诊断方法，不仅适用于复杂的故障诊断，而且在不同的环境下具有良好的跨领域能力。



# 方法

AM-CNN的结构如下：

![fig.2](/images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/fig.2.png)



## 数据预处理

与前一篇文章[26]类似，我们将振动信号分段，采样点为5120个。然后，受该技术的启发，我们将STFT用于振动段，以产生输入信息。

## 残差结构

![fig.4](/images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/fig.4.png)

## 多尺度卷积块

![fig.5](/images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/fig.5.png)

## 整体结构

![fig.6](/images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/fig.6.png)

![table.2](/images/A-multi-scale-convolutional-neural-network-forbearing-compound-faultdiagnosis-under-various-noiseconditions/table.2.png)

# 实验

## 数据集

自建数据集包含多种复合故障



## 实验结果

| Algorithm | Accuracy | F1-macro |
| --------- | -------- | -------- |
| CNN       | 99.51    | 99.88    |
| LSTM      | 99.27    | 99.83    |
| ANCNN_1d  | 12.93    | 33.52    |
| ANCNN_tf  | 0.0      | 39.11    |
| ANNN_1d   | 13.17    | 49.35    |
| CNN_1d    | 100      | 100      |
| MSCNNLSTM | 95.61    | 97.44    |
| AM-CNN    | 100      | 100      |

