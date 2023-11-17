---
title: >-
  Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework
tags:
  - IFD
  - GAN
categories: IFD
thumbnail: /images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.3.png
journal: Advanced Engineering Informatics (IF:8.8)
date: 2023-11-17 16:49:13
---

# 引言

1. 转动部件，比如转子和齿轮容易损坏，因此，有必要对轴承和齿轮的健康状态进行诊断和评估。

2. 深度学习方法特征提取能力很强，但是需要大量的标记数据。然而很难确定能获取到大量有标签数据，这样会导致过拟合的问题。因此，如何使用有限的标记故障样本来训练准确可靠的基于深度学习的故障诊断模型是值得研究的。

3. 介绍无监督GAN模型及其一些变体。

4. 上述无监督GANs在旋转机械故障诊断中得到了很好的应用。然而，他们没有直接考虑故障诊断场景中的多模式数据增强。当多模式故障诊断样本同时用于训练无监督的GANs时，将导致以下问题。

   1. 不同健康状况的不同特征分布可能会影响训练中鉴别器的二进制分类。
   2. 生成器的输入是随机噪声。因此，生成器无法准确拟合不同健康状况的分布，从而难以生成与真实样本相似的样本。
   3. 要控制生成样本的健康状况并不容易，因为生成器只接受没有标签信息的随机噪声。

   因此，需要根据健康状况的数量反复训练多个 GAN，才能有效、稳定地生成多模式故障样本，这在很大程度上降低了诊断效率。

5. 介绍了Auxiliary classifier GAN (ACGAN)及其各种变体。

6. 然而，上述研究大多在某些领域使用了ACGAN，并没有注意到ACGAN的缺点。因此，仍有一些局限性需要克服。

   1. 这些ACGAN中的判别器同时发挥分类和判别作用。当判别器输出错误的结果时，最终生成的样本的质量可能会降低。
   2. 大多数ACGAN的损失函数都是用JensenShannon（JS）散度设计的，由于离散特性，在训练过程中容易出现不稳定性和梯度消失。
   3. 当鉴别器太强大而无法区分真与假时，生成器会生成完全相同的样本，这被称为模型崩溃。

   综上所述，如何解决ACGAN存在的问题，实现高效的多模式样本生成和旋转机械故障诊断已成为一项紧迫的任务。

7. 本文提出了一种改进的 ACGAN（MACGAN），它采用了新的结构框架和损失函数，用于旋转机械的多模式数据增强和故障诊断。其创新如下：
   1. 通过引入独立分类器，开发了一种新的 ACGAN 框架，提高了判别和分类精度之间的兼容性，从而增强了高质量多模式数据生成的能力。
   2. 在新框架上重新设计了新的损失函数，并引入了 Wasserstein 距离，从而有效地解决了模型崩溃和梯度消失的问题。
   3. 采用频谱归一化（SN）策略来限制判别器的权重，而不是权重剪切法，从而提高了训练的稳定性。



# 相关工作



## ACGAN

![fig.1](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.1.png)

ACGAN 是 GAN 的一种有监督改进模型，由生成器 G 和判别器 D 组成，用于相互训练。与无监督 GAN 不同的是，ACGAN 在生成器的随机噪声输入中嵌入了标签信息，其判别器不仅能区分样本来源，还能区分样本类别。此外，ACGAN 还可以通过在对抗训练中引入真实样本和生成样本的分类损失来学习相应的类别信息。无监督 GAN 和基本 ACGAN 的结构如图 1 所示，ACGAN 的监督目标函数包括两部分，分别定义为

![eq.1](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/eq.1.png)

其中，$L_Source$ 表示用于衡量从真实样本中区分样本有效性的目标函数，$L_Class$ 表示用于衡量样本类别有效性的目标函数，Pr(x) 是真实样本分布，Pz(z) 是噪声向量的先验分布，D(x) 表示 x 在真实样本中被选中的概率、 ExP ̃ r(x) 表示真实样本分布 Pr(x) 对 x 的期望，EzP ̃ z(z) 是噪声采样 z 的期望，G(z, cg) 是生成器 G 生成的样本，cr 和 cg 分别是 x 和 G(z, cg) 的标签，P(c = cr|x) 和 P(c = cg|G(z, cg)) 分别是真实样本和生成样本类别标签的条件概率分布。对于 G 来说，目标是最小化$ L_{Class} - L_{Source}$，而 D 则是在对抗训练过程中最大化 $L_{Class} +L_{Source}$。



## Waterstone距离

ACGAN使用JS散度来测量生成的样本和真实样本之间的分布距离。由于JS发散的离散特性，在训练过程中会出现梯度消失和模型崩溃。因此，Arjovsky等人[45]提出了Wasserstein GAN（WGAN），它使用Wasserstein距离来代替损失函数中的JS散度。Waterstone距离比 JS 发散更平滑，当两个分布在高维空间中不重叠时，它也能提供有意义的梯度。Waterstone距离方程如下所示。

![eq.3](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/eq.3.png)

其中，$P_{g(y)}$ 表示生成样本 y 的分布，$inf_{γ∼∏(pr(x),pg(y))} $是真实样本和生成样本的联合概率分布的下确值，$E(x,y)γ∼[‖x y‖]$是真实样本 x 和生成样本 y 之间的期望距离。

在实际计算过程中，不能直接求解$inf_{γ∼∏(pr(x),pg(y))} $，因此通过将网络权重参数限制在一定范围内来近似计算Wasserstein距离，这被称为Lipschitz连续性条件。最后，WGAN的目标函数如下所示。

![eq.4](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/eq.4.png)

# 提出的方法

## ACGAN的新结构框架开发

在本文中，为了提高判别和分类之间的兼容性，以执行高效的多模式样本生成，通过引入独立的神经网络作为分类器，开发了具有新结构框架的MACGAN，如图2所示。

![fig.2](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.2.png)

具体来说，MACGAN将鉴别器的分类功能分离，并赋予其独立的分类器，以提高生成效果。MACGAN具体细节如下：

1. 生成器由全连接层和二维（2D）转置卷积层组成。输入噪声经过上采样操作后被映射到生成的样本上。鉴别器和分类器由全连接层和二维卷积层设计。
2. 为了加快训练收敛速度，避免过拟合，在生成器和分类器中，只在每个卷积层和转置卷积层上添加了批量归一化（BN）。对于鉴别器，BN将通过“除方差”和“乘比例因子”两种运算来破坏其Lipschitz连续性条件。
3. 选择整流线性单元（ReLU）和Leaky ReLU作为每个卷积层和转置卷积层的激活函数[46]。同时，为了满足多分类的需要，分类器的输出层使用了Softmax函数。此外，三个网络的优化器都是使用自适应矩估计（Adam）来选择的。

## 基于Wasserstein距离和SN的新损失函数设计

所提出的MACGAN通过使用Wasserstein距离而不是JS散度来改进基本ACGAN的损失函数，新的目标函数如下：

![eq.5-9](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/eq.5-9.png)

其中$L_D$、$L_G$、$L_C$分别是鉴别器D、生成器G和分类器C的损失函数；$L_C^R$、$L_C^G$分别是具有实样本和生成样本的分类器的损失函数。添加生成样本的分类损失 $L_C^G$ 可以有效缓解 MACGAN 中分类器的过拟合问题，λ1、λ2 分别为分类器损失的比率系数。

为了满足Wasserstein距离的Lipschitz连续性条件，鉴别器的权重参数应该限制在一个恒定的范围内，这被称为权重裁剪。然而，在权重裁剪方面存在两个主要问题[47]。

1. 经过多次权重裁剪操作后，网络参数将收敛到裁剪的边界值，无法获得所需的函数映射。
2. 权重裁剪不稳定，不适当的权重裁剪边值可能导致多层前向传播后梯度消失（边值过低）或梯度爆炸（边值过高）。

为了解决上述问题并提高训练的稳定性，本文在判别器中采用 SN 操作来代替权值剪切。SN 通过控制鉴别器中各层权重矩阵的谱规范，满足公式 (7) 和 (8) 中 ExP ̃ r(x) [D(x)] 和 EzP ̃ z(z) [D(G(z, cg))] 的 Lipschitz 条件。带有权重矩阵 W 的线性层 f(h)= Wh 的 Lipschitz norm 定义如下。

![eq.10-11](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/eq.10-11.png)

其中，suphσ(∇f (h))表示计算 Lipschitz 准则，可转换为计算权重矩阵 W 的 σ(W)，而 σ(W)（也称为 L2 矩阵准则）用于计算权重 W 的最大奇异值，需要确保 σ(W)接近 1，然后 SN 的定义如下式所示。

$W_{SN}(W)=\frac{W}{\sigma(W)}$

在设计了新的损失函数后，可以基于等式对MACGAN进行迭代训练。

## 

![fig.3](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.3.png)

所提出方法的总体框架如图上所示。主要步骤如下。

1. 采集具有各种故障模式的旋转机械的振动信号并转换成二维灰度图像，然后将其分为少量训练样本和足够的测试样本。
2. 基于新的结构框架和新的损失函数开发MACGAN。
3. 将少量故障样本输入MACGAN进行训练，以生成辅助样本。然后，使用两个定量指标和可视化技术对生成的样本的质量进行了评估，并显示了其优越性。
4. 生成的样本用于辅助训练具有高精度和良好稳定性的基于深度学习的故障诊断模型。



# 实验

## 数据集

1. Case Western Reserve University
2. gears from the University of Connecticut

## 结果

### CWRU

![fig.6](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.6.png)

![fig.11](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.11.png)

![table.6](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/table.6.png)

### University of Connecticut

![fig.15](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.15.png)

![fig.18](/images/Multi-mode-data-augmentation-and-fault-diagnosis-of-rotating-machinery-using-modified-ACGAN-designed-with-new-framework/fig.18.png)

# 结论

本文的主要改进点有3点：

1. 通过引入独立分类器，开发了一种新的 ACGAN 框架-实际上不算。
2. 通过Wasserstein距离设计了一种新的损失函数，从而有效地解决了模型崩溃和梯度消失的问题。（在文献中实际上提到了Zou等人[36]提出了一种基于Wasserstein距离的多尺度ACGAN，可以有效地生成高分辨率的船舶切片。）
3. 采用频谱归一化（SN）策略来限制判别器的权重，而不是权重剪切法，从而提高了训练的稳定性。(也是别人用在GAN中的-Spectral normalization for generative adversarial networks)

所以，emmm。just like：GAN+A; GAN+B; GAN+C = GAN+(A+B+C)。
