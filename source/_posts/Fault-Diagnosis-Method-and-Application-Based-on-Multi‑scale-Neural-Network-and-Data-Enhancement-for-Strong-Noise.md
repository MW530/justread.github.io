---
title: "Fault-Diagnosis-Method-and-Application-Based-on-Multi‑scale-Neural-Network-and-Data Enhancement-for-Strong-Noise"
tags:
  - IFD
categories: IFD
thumbnail: /images/Fault-Diagnosis-Method-and-Application-Based-on-Multi‑scale-Neural-Network-and-Data-Enhancement-for-Strong-Noise/fig.3.png
journal: Journal of Vibration Engineering & Technologies (IF:2.7)
date: 2023-11-09 23:02:19
---

# 引言

1. 故障诊断很重要；传统故障诊断方法（VMD，SVM）；深度学习方法（CNN）。
2. CNN方法用的很多，但是其大多都只考虑单尺度的卷积核（好像并不是）。
3. 现有的方法对于噪声处理的不是很好。具体有：
   1. 首先，单尺度卷积核的频率分辨率范围有限，只能从原始故障信号的部分故障带中提取故障特征。
   2. 第二，在强噪声干扰的恶劣工作条件下，原始故障信号被噪声淹没。
4. 为了解决上述问题，做出了以下改进：
   1. 本文提出了一种多尺度深度卷积神经网络（MSD-CNN）模型。通过提取原始故障信号的多尺度信息，解决了单尺度卷积核故障信号特征提取不完善的问题。
   2. 使用了数据增强来解决数据噪声的问题。



# 方法

## 网络

![fig.1](/images/Fault-Diagnosis-Method-and-Application-Based-on-Multi‑scale-Neural-Network-and-Data-Enhancement-for-Strong-Noise/fig.1.png)

## 数据增强

由于机械设备受到环境、机械设备和部件耦合的内外干扰，采集到的故障信号具有很大的噪声。该模型难以提取有效的故障信号特征，导致故障诊断精度快速下降。本文提出了一种针对强噪声干扰的数据增强方法，该方法可以通过故障信号数据集的重构和故障信号数据的增强来增强对故障特征信息的识别。

1. 故障信号数据集构造（重采样）：使用大小为W的窗口对故障信号进行步长为S的单向重叠采样。采样的子信号（S1，S2，…，Sn）构成故障信号数据集（F1），并分为训练数据集（T1）和测试数据集（T2）。设L（t）为原始一维时域信号，Sn（t）是重叠采样后的子信号。n是子信号样本的数量。
2. 故障信号数据集的增强。如图6所示，如图2（b）所示，随机选择训练数据集中的训练样本（T1）的1%，并将所选择的训练样本作为一个整体（T3）。数据集（T3）与不同强度的噪声信号（D1，D2，…，Dn）叠加，**以形成不同强度的噪音数据集**（N1，N2，…，Nn）。噪声数据集（N1，N2，…，Nn）和训练数据集（T1）被组合，并且组合的数据集替换原始训练数据集。由于添加了额外的噪声数据集，提高了训练数据集中样本的数量和多样性。
2. ![fig.2](/images/Fault-Diagnosis-Method-and-Application-Based-on-Multi‑scale-Neural-Network-and-Data-Enhancement-for-Strong-Noise/fig.2.png)

# 实验

实验流程图：

![fig.3](/images/Fault-Diagnosis-Method-and-Application-Based-on-Multi‑scale-Neural-Network-and-Data-Enhancement-for-Strong-Noise/fig.3.png)

## 数据集

1. CWRU

dataset A：原始CWRU数据集

dataset B：加了高斯噪声的CWRU数据集

## 结果

![fig.7](/images/Fault-Diagnosis-Method-and-Application-Based-on-Multi-scale-Neural-Network-and-Data-Enhancement-for-Strong-Noise/fig.7.png)

# 总结

比较简单的方法，多尺度的思想挺多的。数据增强也只是加了噪声。
