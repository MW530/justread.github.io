---
title: >-
  Global contextual residual convolutional neural networks for motor fault
  diagnosis under variable-speed conditions
tags:
  - IFD
categories: IFD
thumbnail: https://xxxxxxxxxx.png
journal: Reliability Engineering & System Safety (IF:8.1)
date: 2023-11-19 15:01:59
---

# 引言

1. 由于现代工业系统的结构复杂性和功能耦合性，任何微小的故障都可能引发连锁反应，使工业生产的安全问题日益突出。在此背景下，设备监测与故障诊断技术受到了业界的广泛关注。

2. 故障诊断方法主要分为两类：

   1. 模型驱动：主要是基于对现行系统的理论理解，这需要大量的专家知识。这种方法具有复杂度高、泛化能力低的特点，不适用于大型复杂系统。
   2. 数据驱动：通常采用基于深度学习的技术来建立反映输入和系统状态之间关系的非线性映射，更有利于结构复杂的现代工业系统的监测。

3. 介绍了深度学习的一些代表作以及在工业故障诊断中的一些应用。

4. 介绍了多尺度CNN的一些文章。

5. 在变速条件下，振动信号的振幅和频率随速度波动大而变化很大。特别是对于工业系统，设备经常在变速工况下运行，使振动信号时变且不稳定。因此，从变速条件下采集的信号中提取判别特征对现有的CNN模型来说是一个挑战。

6. 先有方法的缺点：

   1. 大多数现有的基于CNN的方法未能充分探索测量信号的层次表示。**它们只使用来自最后一个卷积层的特征进行故障识别，这使得中间层提取的一些有用的分层特征没有得到充分利用**。
   2. 此外，它们缺乏故障信号学习机制，即不分青红皂白地处理具有不同脉冲和模式的信号。**这使得他们无法有利地处理不平衡的数据，因为他们可能倾向于对来自多数类别的数据进行特征映射，而忽略了对来自少数类别的数据的特征学习。**由于不同的特征对于故障检测任务具有不同的重要性。因此，网络应该更加重视判别特征的提取。

   为了克服上述困难，实现非平稳条件下的电机故障识别，提出了一种全局上下文残差卷积神经网络（GC-ResCNN）。综上所述，本文的主要贡献如下：

   1. 为了充分利用CNN模型提取的特征信息，设计了一种新的层次结构，将所有中间层的特征封装到最终表示中。来自中间层的特征通常包含大量有价值的详细信息。充分利用这些中间层的特征信息，特别是在变速条件下采集样本时，可以使网络在框架中获得更全面的特征信息。从而促进网络实现更高的诊断精度。
   2. 为了将故障信号学习能力融入网络，引入**全局上下文模块**来引导模型学习更多的判别特征。此外，在模型中应用了残差跳跃学习策略，促进了CNN模型的特征转移。此外，网络学习到的特征可以相互促进，也可以相互矛盾，因此引入了多特征融合层来自适应地整合这些提取特征的多级信息。
   3. 为了更好地处理在非平稳条件下收集的运动数据，基于上述改进，**开发了一种新的基于CNN的模型**，称为GC ResCNN。利用从感应电机试验台和工业电机泵系统中收集的电机数据集进行了案例研究，对所提出的方法进行了综合性能评估。实验结果表明，与九种最先进的方法相比，该模型在非平稳条件下具有更高的诊断精度，验证了GC ResCNN的优越性。



# 相关工作

## CNN（略）

## Residual learning block（略）

## Global context block-全局上文模块

![fig.2](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/fig.2.png)

近年来，非局部激活模块已成为深度学习领域的研究热点之一。全局上下文模块是基于非局部网络和SENet的理论开发的。全局上下文块的架构如图2所示。首先，通过聚集来自所有位置的输入特征来提取全局上下文特征信息，这是由上下文建模组件实现的。然后，进行注意力池以获得全局上下文特征。给定输入特征向量x，公式如下：

$\beta=Softmax(p_1(x)) \\ k=\beta \odot x$

𝛽 是注意力权重；𝑝1表示1×1的卷积；⊙表示逐元素乘法运算符。然后，特征转换模块被用于捕获信道方式的相互依赖性：

$𝑦 = 𝑝_3(𝑅𝑒𝐿𝑈 (𝐿𝑁(𝑝_2(𝑘))))$,

$𝑝_2$和$𝑝3$表示1×1的卷积运算；𝐿𝑁（‧）是层规范化操作。最后，实现融合操作以聚合全局上下文特征：

$𝑦= \hat{y} + 𝑥$.



------

GCB模块出自GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond(2019)一文.

GCB研究思路类似于DPN，DPN深入探讨了ResNet和DenseNet的优缺点，然后结合ResNet和DenseNet的优点提出了DPN，同样GCNet深入探讨了Non-local和SENet的优缺点，然后结合Non-local和SENet的优点提出了GCNet。

其主要原理是：

NLNet就是采用自注意力机制来建模像素对关系。然而NLNet对于每一个位置学习不受位置依赖的attention map，造成了大量的计算浪费。

SENet用全局上下文对不同通道进行权值重标定，来调整通道依赖。然而，采用权值重标定的特征融合，不能充分利用全局上下文。

通过严格的实验分析，作者发现non-local network的全局上下文在不同位置几乎是相同的，这表明学习到了无位置依赖的全局上下文。比如对COCO数据集的可视化：

![COCO](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/COCO.png)

**基于上述观察，本文提出了GCNet，即能够像NLNet一样有效的对全局上下文建模，又能够像SENet一样轻量。**

具体见[这里](https://zhuanlan.zhihu.com/p/64988633)。以及[原文](https://arxiv.org/abs/1904.11492?context=cs.LG)。

## 多特征融合层

CNN模型学习到的特征要么相互促进，要么相互矛盾，因此自适应特征集成机制对于所提出的模型是必要的。在本研究中，引入了多特征融合层来自适应地选择和组合学习到的特征，以提高这些模型在模式识别任务中的性能。多特征融合的体系结构如图3所示。

![fig.3](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/fig.3.png)

在多特征融合层中，我们首先馈送输入特征$𝑌^{𝑟,𝑡}(𝑌^{𝑟,𝑡}∈𝑅^{𝐶×𝐻×𝑊 }) $1×1卷积层，以减少输入特征图的混叠效应。然后应用ReLU激活函数来引入非线性。然后，使用另一个1×1卷积运算来映射跨通道特征信息。此外，采用Sigmoid形层来获得空间注意力图$𝑆^{𝑟,𝑡} ∈ 𝑅^{𝐶×𝐻×𝑊} $:

$$S^{r,t}=sigmoid(g_2(ReLU(g_1(Y^{r,t})))),$$

其中$𝑔_𝑖(i=1,2)$表示𝑖×1卷积运算。然后，可以通过以下方式激活特征图：

$$A^{r,t}=Y^{r,t}\odot(1+S^{r,t}),$$

其中⊙表示元素乘法。最后，使用3×1卷积运算来进一步合并来自不同分支的信息：

$\hat{Y}=g_{3 \times 1}(A^{r,t})$

其中$𝑔_{3×1}$表示3×1卷积运算；和𝑌 表示输出特征图。

# 提出的模型

![fig.4](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/fig.4.png)

所提出的GC ResCNN的架构如图所示。4。所提出的GC ResCNN架构由四个分支组成，即主分支、分支1、分支2和分支3。主分支由3个残差块组成。

主分支可以利用残差学习策略提取深层非线性特征。

分支1、分支2和分支3具有相同的结构；它们都由几个卷积层和一个GC块组成。来自主分支的侧出特征被分别输入到具有21个通道的两个卷积层。每个卷积层都伴随着ReLU激活函数和批量归一化层。此外，应用逐元素求和运算来融合输出特征。此外，GC模块将全局学习能力集成到模型中。

# 实验

## 数据集

1. Multiscale kernel based residual convolutional neural network for motor fault diagnosis under nonstationary conditions

## 对比方法

1. WD-CNN-2017Sensor
2. DRCNN-2019ISA Trans
3. RNN-WDCNN-2020Sensors
4. MSCNN-2019J Intell Manuf
5. MK-ResNet-2020J Intell Manuf
6. MCNN-LSTM-2020J Intell Manuf
7. MA1DCNN-2020TII
8. DFPCN-2020TIM
9. DCACN-2020IEEE Access



## 实验结果

![table.1](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/table.1.png)

![table.2](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/table.2.png)

![table.4](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/table.4.png)

![table.6](/images/Global contextual residual convolutional neural networks for motor fault diagnosis under variable-speed conditions/table.6.png)

# 总结

本文提出的网络主要改进就是残差和全局上下文模块（GCNet里提出来的，不知道为什么没有引用）。

对于每一层都进行GC操作，然后融合。整体创新不算特别大。
