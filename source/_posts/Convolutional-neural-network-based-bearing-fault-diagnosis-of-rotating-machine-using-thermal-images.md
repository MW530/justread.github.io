---
title: >-
  Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images
tags:
  - IFD
  - thermal images
categories: IFD
thumbnail: /images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/fig.4.png
journal: Measurement(IF:5.6)
date: 2023-12-03 23:48:58
---

# 引言

1. 旋转机械部件的状态监测至关重要，任何故障都可能导致严重损坏和机器停机，增加维护成本。红外热像仪是一种非侵入性和高准确性的无损检测工具，用于自适应性旋转机械故障诊断，引起了广泛关注。**传统的振动技术会产生大量信号噪声，而红外热像仪则克服了润滑相关问题并提供轴承表面温度图或热图像。**这种新兴技术提供了完整的故障信息利用机会，并吸引了技术人员和研究人员的关注。
2. 介绍了机器学习算法以及深度学习算法在故障诊断中的应用；IRT 是一种广为接受的技术，由于其非侵入性、非破坏性和快速性等优点，在无损检测和评估 (NDTE) 中有着广泛的应用。然后介绍IRT在故障诊断中的一些应用。
3. 这项工作的主要目标是分析所介绍的基于 CNN 的方法的可用性和准确性，并利用热图像将其结果与基于 ANN 的方法进行比较。在恒定载荷的不同转速条件下，采集了五个故障轴承和一个健康轴承的热图像。然后，应用所提出的 CNN 方法自动从热图像中提取特征。本文的重要贡献如下。
   1. 将 CNN 与热图像相结合，创建了一种新的智能轴承诊断技术，可自动提取轴承状况识别的深层特征。它具有无创、无损、快速等诸多优点，还避免了像振动信号那样的传感器物理安装。
   2. 实验结果表明，使用 IRT 可以准确识别轴承故障源的每个已知元素，而且识别精度更高。此外，还对拟议的 ANN 和 CNN 的有效性和鲁棒性进行了比较研究。



# 实验设置和数据采集

![fig.2](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/fig.2.png)

![fig.3](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/fig.3.png)

红外热像仪 FLIR-P640 的帧频为 7 帧/秒，用于采集 IRT 数据。在涉及距离、相对湿度和刻度温度的重要参数中，灵敏度是最重要的参数，取决于外壳材料的表面特性。发射率的典型范围是 0.1-0.95。在本研究中，房屋表面的发射率取值为 0.64。目标物体与红外热像仪之间的距离对于捕捉高质量的分辨率图像也很重要。系统运行一小时达到稳定状态，然后按照 [28] 的方法采集五分钟的数据。图像从轴向拍摄，以便清楚地了解各种故障条件下产生的模式[29]。图 3 显示了以 19 Hz 频率采集的不同轴承状态下的原始热图像。然而，仅凭原始热图像无法区分不同的故障情况，因此需要一种智能算法来开发自诊断系统，表 1 列出了每种轴承情况下 IRT 图像数据集的数量。

![table.1](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/table.1.png)

# 方法

本节描述了所提出的基于神经网络和CNN的轴承故障诊断方法。首先，所提出的神经网络方法需要特征提取和特征选择过程。其次，所提出的基于CNN的方法需要对热图像进行预处理并将其传递给CNN。图4说明了所提出的基于神经网络和CNN的轴承故障诊断方法的示意图。

![fig.4](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/fig.4.png)

## 提出的基于 ANN 的轴承故障诊断

在这一提出的工作中，流程如下：

1. 从 IRT 数据中提取连续帧，然后采用基于 ANN 的轴承故障条件特征提取过程。在解释时域信号时会用到几个参数。在这项工作中，有 下面这些统计时域参数或特征：
      1. 平均值 (y)
      1. 均方根值 (yrms)
      1. 标准偏差 (σ)
      1. 形状因子 (ysf)
      1. 峰度 (ymk)
      1. 偏斜度 (ysk)
      1. 峰值振幅 (ymax)
      1. 波峰因数 (ycf )
      1. 脉冲因数 (yif)
      1. 信噪比
      1. 方差 (σ2)
      1. 能量 (ye)
      1. 熵 (yen)
      1. 边际误差 (σ)
      1. 熵（yen）
      1. 边际系数（ymf）
      1. 信噪比
      1. 失真率（ySINAD）
2. 提取的特征随后在 0 到 1 的范围内进行归一化处理
3. 然后再通过邻域成分分析（NCA）这种非参数方法进行降维，这是提高分类准确性的重要步骤。
4. 通过使用 NCA 估算的每个特征的特征权重，计算出轴承数据集六个类别的最佳特征排序。
5. 为了提高准确性，特征相关性得分达到一个合适的等级后，采用六个特征进行进一步分类。在三个转速和恒定负载条件下，从每个轴承状态的原始热图像数据（包括一些统计测量值）中，ANN 共提取了 15 个特征。
6. 研究中使用了 4 个计算节点和 20 个隐藏层，反向传播过程中使用的参数有：lick 1000 epoch、最大梯度 1010、均方误差 1010，并在训练过程中使用了 sigmoid 激活函数。网络的权重和偏置是随机初始化的。在原始热图像数据中，70% 为训练数据，10% 为验证数据，20% 为测试数据，用于评估 ANN 分类器的性能。

## 提出的基于 CNN 的轴承故障诊断

![fig.5](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/fig.5.png)

CNN 能够从轴承缺陷的不同热图像中学习多阶段不变性。此外，随着严重程度的增加，图 4 中的模式也会根据特定缺陷发生变化。基于 CNN 和热图像的轴承故障诊断流程图。 轴承可能有利于 CNN 区分各种故障情况 [26]。本研究中使用的 CNN 模型源自 **LeNet-5** 模型架构[30,31]，由于其像素特征提取特性[20]，该模型在旋转机械故障诊断中也表现出色。卷积层和池化层是拟议模型的主要部分；拟议 CNN 模型的结构如图 5 所示。



# 实验

## 对比方法

1. CNN+IRT-2020- Chin. J. Aeronaut.
2. TCNN--2019IEEE Access
3. IRT-2020-IEEE Sens J
4. IRT-2015-Infrared Phys. Technol
5. IRT+Vibration-2018-TII

## 结果

![table.3](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/table.3.png)

![table.4](/images/Convolutional-neural-network-based-bearing-fault-diagnosis-of-rotating-machine-using-thermal-images/table.4.png)



# 总结

1. 使用IRT图片进行诊断是一个可行方向，但是实际上在故障诊断领域2018就开始有人做了。
2. 使用人工神经网络ANN和CNN融合也是一个方向。













