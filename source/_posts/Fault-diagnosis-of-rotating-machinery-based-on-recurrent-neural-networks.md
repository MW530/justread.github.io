---
title: Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks
tags:
  - IFD
  - GNN
categories: IFD
thumbnail: /images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.3.png
journal: Measurement(IF:5.6)
date: 2023-11-24 16:03:38
---

# 引言

1. 现代工业系统中的设备复杂性高，旋转机械的故障诊断一直是学术界和工业界关注的焦点。旋转机械中的滚动轴承容易受到各种损伤，在极端工作条件下容易发生故障，影响系统的性能并造成巨大损失。因此，有必要开发基于旋转机械的有效故障诊断方法。

2. 故障诊断可以被视为一个模式识别任务，其主要包含三个步骤：

   1. 震动信号收集
   2. 特征提取：特征提取的目的是构造旋转机械振动信号的特征特征，并将原始数据转换为一组具有明显物理意义或统计意义的特征。
   3. 故障识别：然后，将提取的特征发送到机器学习算法中。

   然而，这些方法严重依赖于特征提取过程，需要专家知识，这直接影响了故障分类的准确性。

3. 介绍了深度学习方法解决了上述问题，然后列举了一些在故障诊断领域的应用。然而，DNN中大量的参数阻碍了故障诊断的能力，这可能导致昂贵的计算成本和过拟合。相反，卷积神经网络（CNNs）和递归神经网络（RNN）在故障诊断中表现出了优越的性能。

4. 与CNN相比，由于其递归的隐藏层，RNN在利用时间序列振动信号的时间信息方面具有优势。然后列举了RNN在故障诊断方向的一些应用。然而，时间信息并没有得到充分利用。换言之，RNN在提高故障诊断性能方面仍有很大潜力。此外，这些文献中忽略了抗噪声鲁棒性的重要性，这在极端工作条件下是必不可少的。

5. 本文提出了一种基于门控递归单元递归神经网络的旋转机械故障诊断新方法。本文的目的主要有两点：

   1. 提高旋转机械故障分类的准确性。
   2. 对噪音保持稳健，旨在减少极端工作条件的影响。

   本文的主要贡献总结如下：

   1. 提出了一种基于GRU的旋转机械故障诊断新方法，利用时间序列振动信号中的时间信息进行故障诊断。
   2. 通过从一维振动信号中构造图像，然后利用线性层来提升每个信号段的维数，开发了一种改进的特征提取方法，这有利于GRU学习和捕捉代表性特征。最后应用分类模块实现了故障识别。
   3. 在FDGRU上应用了残差连接和学习率衰减两种策略，使训练过程更加稳定，有助于提高故障识别的准确性。

​	

# 递归神经网络和门控递归单元网络
![RNN_Structure](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/RNN_Structure.jpg)

![RNN_Structure2](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/RNN_Structure2.jpg)

递归神经网络是人工神经网络的一类，经常用于处理序列数据。特别是有两种RNN变体被提出来解决长期依赖性问题：

1. 门控递归单元（GRU）
2. 长短期记忆网络（LSTM）：长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。



## LSTM

![LSTM_Inner](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/LSTM_Inner.png)

![LSTM_Structure](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/LSTM_Structure.png)

## GRU

![GRU_1](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/GRU_1.png)

![GRU_2](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/GRU_2.png)

图2-4中的 ⊙ 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 ⊕ 则代表进行矩阵加法操作。

与LSTM相比，GRU具有更少的参数，并且在有限的数据下更容易收敛。因此，本文采用GRU作为我们的主要框架。在本节中，将简要介绍RNN和GRU。

![fig.1](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.1.png)

$$\begin{aligned}
&z_{t}=\sigma(W_{z}x_{t}+U_{z}h_{t-1}), \\
&r_{t}=\sigma(W_{t}x_{t}+U_{t}h_{t-1}), \\
&\tilde{h}_{t}=\mathrm{tanh}(Wx_{t}+U(r_{t}\odot h_{t-1})), \\
&h_{t}=(1-z_{t})\odot h_{t-1}+z_{t}\odot\tilde{h}_{t}.
\end{aligned}$$

对于RNN，神经元不仅可以接受来自其他神经元的信息，还可以接受自己的信息，形成具有环路的网络结构。因此，RNN在利用时间信息方面比CNN更有优势。RNN的基本架构如图1（a）所示。给定输入序列$𝑋 = (𝑥_1,𝑥_2,𝑥_3,...,𝑥_𝑇)$, 当前隐藏输出$ℎ_𝑡$ 隐藏层的，

$$h_t=\sigma(U_{x_t}+W_{h_{t-1}}+b)$$

其中，$𝑥_𝑡$ 表示时间步长 𝑡 时的当前输入信息，𝑇 表示输入的总序列长度，转换函数 $𝜎(⋅)$ 是激活函数，𝑈、𝑏 和 𝑏 是参数矩阵和向量。具体地说，𝑈 是输入层到隐层的权重矩阵，而𝑈 是隐层在前一个时间点 $𝑡 - 1$ 的值作为当前时间点 𝑡 的输入的权重。参数可以通过反向传播算法随时间学习。需要注意的是，当前的隐输出𝑡 是由当前的输入信息$𝑥_𝑡$和之前的隐输出𝑡-1 决定的，这使得 RNN 能够保持对之前信息的记忆。然而，由于在训练过程中存在梯度消失或梯度爆炸的问题，RNN 在处理长期依赖关系时会遇到很大困难。为了解决这个问题，人们提出了门控递归单元，即引入门控机制。

GRU引入两个门（重置门𝑟 和更新门𝑧) 控制信息流，可以学会保留重要信息和丢弃无关信息。具体而言，重置门𝑟 用于调整以前的内存和新输入的组合，并更新门𝑧 决定要保留的先前内存量。因此，GRU能够利用来自长链序列输入的足够有用的信息来进行预测，并避免长期依赖性问题。GRU的体系结构如图1(b)所示。

其中，$𝜎(⋅)$ 表示激活函数，$\hat{h}_t$表示单元值，旨在剔除无关的历史信息。需要注意的是，在每个时间步中，包括W和 𝑈 在内的模型参数都是共享的，这就减少了训练参数的总数。

RNN相关可以看这里，比较简洁的介绍：[GRU](https://zhuanlan.zhihu.com/p/32481747)，[LSTM](https://zhuanlan.zhihu.com/p/32085405)

# 提出的方法

图像构造介绍了一种将一维振动数据转换为二维图像的数据预处理方法，FDGRU结构介绍了所提出的神经网络设计的细节，并介绍了FDGRU方法在基于FDGRU的故障诊断中的一般过程。

![fig.2](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.2.png)

![fig.3](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.3.png)

![fig.4](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.4.png)



## 基于FDGRU的故障诊断

1. 收集旋转机械系统的振动信号。
2. 在不进行人工特征提取的情况下，将初始的一维振动信号依次转换为二维图像。
3. 使用线性层来增加处理后的训练数据的维度，以便后续模块学习潜在特征。
4. 应用GRU处理先前线性层的输出，并从时间序列数据中学习高级特征。然后，将学习到的代表性特征发送到分类模块中，以实现故障识别。
5. 使用测试数据验证所提出的FDGRU方法的性能。



# 实验

## 数据集

1. Case Western Reserve University (CWRU) Bearing Data Center
2. Self-priming Centrifugal Pump (SPCP) Dataset: This dataset is provided by Lu et al.

## 训练详细

CWRU和SPCP数据集中的每个信号样本包含4096个数据点。（𝑁 = 64）。在训练过程中，RMSProp用于大小为24的批次的优化，丢弃率为0.5，以缓解过拟合问题。初始学习率被设置为0.0002，并且当训练时期达到25时以学习率衰减0.1的因子减小。在我们的实验中，训练历元的总数被设置为35。除非另有说明，所有实验都进行了十次，以避免测试过程中的偶然性，平均值被视为最终的分类结果进行分析。

## 结果

![table.3](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/table.3.png)

![table.4](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/table.4.png)

![fig.8](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/fig.8.png)

![table.5](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/table.5.png)

![table.6](/images/Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks/table.6.png)



# 总结

本文的主要创新点在于使用GRU网络，然后提出了一种框架，即Linear Layer-GRU-MLP的结构。然后加了一些tricks: residual connection & learning rate decay。















