---
title: Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model
tags:
  - IFD
  - CNN
  - Multi-scale
categories: IFD
thumbnail: /images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/fig.4.png)
journal: Journal of Intelligent Manufacturing (IF:8.3)
date: 2023-11-24 22:39:32
---

# 引言

1. 转动部件很常见，其损坏会导致很多问题。因此有必要对转动轴承进行诊断。

2. 轴承的故障诊断一般基于振动信号，且可被分为两部分：

   1. 特征提取
   2. 分类

   由于振动信号同时包含轴承状态和噪音，因此基于单一特征是不够的。并且常常需要将时域信息转变为时频域信息。

3. 并不是所有的信号都重要，因此需要从高维特征中挑选合适的特征。这样可以降低计算负担的同时增加分类准确率。

4. 人工智能（AI）已广泛应用于模式识别，在图像处理方面取得了显著成就。介绍了一系列的人工智能方法。

5. 介绍深度学习在故障诊断中的一些应用。上述各种算法和方法都取得了令人满意的结果，其中一些已在工业领域得到实际应用。然而，它们仍然存在一些局限性：

   1. 振动信号与故障类型之间的映射非常复杂。因此，故障诊断效果在很大程度上依赖于人工提取的特征质量，智能诊断方法的优势并未得到发挥。
   2. 人工特征提取是一项非常费力和耗时的工作，通常需要大量与信号处理和数学相关的核心知识。
   3. 一般来说，ANN 和 SVM 网络结构较浅，这限制了它们适应复杂非线性信号的能力。虽然增加隐藏层数可以提取更多有用的特征，但也会大大增加计算负担。
   4. 之前的大多数研究只考虑了一小部分故障类型，通常为 3-6 种。因此，当故障数量增加或出现不同故障类型时（如实际故障案例中常见的情况），这些方法就会失效。

6. 本文提出了多尺度卷积神经网络和长短期记忆（MCNN-LSTM）故障诊断模型，以解决上述问题。MCNN-LSTM 包括特征提取器和分类器，可将原始数据直接输入模型，无需预处理。建议的方法具有以下优点：

   1. 紧凑的网络结构和原始数据输入，可实时检测轴承状态。
   2. 无需预处理（如 EMD、HHT 等）即可从原始信号中自动学习特征。
   3. 利用小型数据集提供有效的训练和分类方法。



# 相关工作

## 滚动轴承故障特征

## 卷积神经网络

## LSTM

# 提出的多尺度 CNN 和 LSTM 模型

先对信号进行下采样，然后再将其提交给所提出的模型，以提高计算速度和性能。原始信号直接输入模型，进行自动特征提取和故障分类。第一个模块（特征提取器）由两个具有不同内核大小和深度的一维 CNN 组成。原始信号同时输入 CNN，以提取不同频域的特征。我们为 CNN_1 采用了较大的感受野（20 × 20 和 10 × 10）来自动提取低频特征，而 CNN_2 则从高频信号中提取特征，因此采用了较小的感受野（6 × 6）。CNN_1 和 CNN_2 的特征向量通过元素相乘的方式进行融合。一维 CNN 作为特征提取器有几个优势：它们能自动学习不同振动信号的基本含义；

使用共享权重策略可以大大降低输入向量的维度，同时也减少了参数的数量；上一层的输出和下一层的输入受到内核大小的限制。

第二个模块是分类器，由分层 LSTM 和全连接层组成，用于在输入和输出之间构建复杂的非线性模型。图 4 显示了 LSTM_1 的隐藏状态如何为 LSTM_2 提供输入，而 LTSM_2 的输出如何输入到全连接层进行分类。LSTM 网络的输入分为几个步骤。下一步的输出受上一步输出的影响。因此，LSTM 网络能很好地利用时序信号的特性，充分提取振动信号的内部特征。最后，softmax 函数将神经元输出转换为 10 种滚动轴承健康状态的概率分布。其中，z 表示第 j 个神经元的输出。

![fig.4](/images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/fig.4.png)



# 实验

## 数据集

1. CWRU

![fig.6](/images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/fig.6.png)

## 实验结果

![fig.7](/images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/fig.7.png)

![table.3](/images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/table.3.png)

![table.4](/images/Bearing-fault-diagnosis-base-on-multi-scale-CNN-and-LSTM-model/table.4.png)

# 总结

结合了CNN和LSTM。



