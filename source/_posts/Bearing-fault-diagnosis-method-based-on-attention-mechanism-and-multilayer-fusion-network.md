---
title: >-
  Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network
tags:
  - IFD
categories: IFD
thumbnail: /images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.6.png
journal: ISA Transactions（IF:7.3）
date: 2023-12-18 19:10:06
---

# 创新点

1. 为了从不同类型的传感器信号中提取和融合信息，深度结构设计由多个独立的分支网络和一个中心网络组成。分支网络分别用于提取多传感器信号的代表性特征。同时，中心网络用于从不同类型的传感器信号中融合这些代表性特征。
2. 与现有的融合框架不同，深层结构中的中心网络在每一层分支网络之间设置一系列融合节点，然后在多个融合层挖掘提取特征的相关信息。
3. 每个融合节点都引入了注意机制，利用自适应确定的来自不同类型传感器信号的特征权重来融合这些特征，而不是简单地用平均权重来融合。



# 方法

## motivation

一般来说，故障诊断需要获取多种监测信号，如**振动、声音和电流信号**。然而，由于产生机制、采集设备甚至安装位置的不同，不同类型的信号揭示故障的能力也大相径庭。因此，人们希望充分利用监测信号并融合其信息，以准确识别故障。

虽然可以利用文献[25-27]中的处理方法将多路信号直接重组为二维矩阵，然后输入到深度结构中提取代表性特征并识别故障，但重构后的输入矩阵容易破坏原有的时域或频域序列，使数据失去原有的物理意义。同时，相对于同类信号的融合，多类信号之间的相互干扰更为明显，使得模型难以提取有效特征。此外，单一信号的特征也可能被掩盖，从而导致可能的信息丢失。因此，为每个传感器信号设计独立的特征提取通道，并使用多分支网络处理多传感器信号是非常必要的。



考虑到深度神经网络固有的多层特性，信息融合在每个网络层都有不同的表现。因此，我们希望在网络层中设置多个融合点，自适应地融合多层次的特征。



此外，多传感器特征具有冗余性和相关性，直接将特征串联成特征向量并不可取。因此，基于注意力机制（AMF）的融合策略可实现多种特征的有效选择和组合，从而提高信息融合的质量，改善故障诊断的性能。



## Inception layer

![fig.2](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.2.png)

作为对 CNN 的改进，在分支网络中利用 Inception 网络提取不同类型信号的特征，Inception 网络的结构如图 2 所示。与传统的卷积层不同，它是由多个具有不同大小卷积核的并行卷积层和一个池化层组成。与卷积层相比，Inception可以有效减少网络结构中的参数数量。此外，特征被独立地输入到多个卷积层和池化层，然后在通道维度上串联输出特征作为输出。由于在同一层中使用了 1 × 1、1 × 5、1 × 7 卷积核，因此可以提供不同大小的感受野，增加特征提取的丰富性。因此，"感知层 "用于提取单传感器数据的深层故障特征。

## 多层融合框架

![fig.3](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.3.png)

一般来说，多传感器特征具有冗余性和相关性的特点，直接将特征串联成特征向量并不能总是聚焦于与故障最相关的特征。因此，本文设计了 AMF，在每个融合点对多传感器特征进行融合。

### AMF（注意力融合机制）

1. 首先，利用全局平均池化操作，在空间维度上压缩来自不同传感器信号的特征。
2. 其次，将两个传感器信号的压缩特征进行组合，生成全局表示信息 Fg∈R2M 。此外，为了使激励信号完全校准每个传感器的特征，特征学习过程应该是非线性的，因此在全局信息特征之后增加了全连接操作，以改善非线性。
3. 在压缩特征 Fz 的基础上，用软关注生成激励信号 P1 和 P2∈ RM，从而自适应地选择各信道的特征。此外，还添加了 SoftMax 函数，以获得各信道特征的激励概率。
4. 第四，通过门控机制，对不同传感器数据的每个特征进行重新校准，并通过激励信号进行融合。

经过上述四个步骤后，两个传感器信号的特征就可以在每个融合点进行融合。由于卷积层和池化层的存在，不同融合点之间的特征维度并不一致。为了融合不同层次的特征，当前层次的融合特征将首先经过维度变换模块（DTM），如图 5 所示。

![fig.4](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.4.png)

![fig.5](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.5.png)

## 整体结构

![fig.6](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.6.png)

![fig.7](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.7.png)



# 实验

## 数据集

1. public bearing data of Paderborn University
2. experimental data obtained in our laboratory



![table.1](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/table.1.png)

![table.2](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/table.2.png)

![table.3](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/table.3.png)

![fig.10](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.10.png)

![fig.11](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.11.png)



## 对比方法

1. SVM
2. DCNN
3. ECNN

![fig.13](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/fig.13.png)



## 结果

![table.6](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/table.6.png)

![table.9](/images/Bearing-fault-diagnosis-method-based-on-attention-mechanism-and-multilayer-fusion-network/table.9.png)



# 总结

1. 新思路，**融合电流信号**进行诊断。**注意力融合机制**也是一个可行的融合方法。
2. 网络结构上倒是不新鲜。
