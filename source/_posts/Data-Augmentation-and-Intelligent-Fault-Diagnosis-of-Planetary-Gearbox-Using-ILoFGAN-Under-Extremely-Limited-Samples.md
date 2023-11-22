---
title: >-
  Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples
tags:
  - IFD
  - GAN
categories: IFD
thumbnail: /images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.4.png
journal: IEEE Transactions on Reliability(IF:5.9)
date: 2023-11-22 10:12:01
---

# 引言

1. 齿轮工具箱应用广泛，因此对其故障进行诊断非常重要。数据驱动的方法得到了很多关注，特别是基于深度学习的方法。然而，深度学习的成功很大程度上取决于训练样本的充足性。不幸的是，随着可靠性和质量的不断提高，在真实的工业场景中，机器大部分时间都在健康状态下工作，这意味着很难获得足够的故障样本。
2. 为了应对这一挑战，研究人员相继开发了数据增强方法，如数据采样和数据生成。数据采样技术主要包括随机过采样、合成少数过采样技术和自适应合成。**尽管这些方法可以合成新的故障样本，但它们专注于复制或插值技术，而没有考虑数据分布的影响，这将导致缺乏多样性。**
3. 数据生成技术旨在从原始数据中捕获特征，并生成具有相似分布的新样本。然后介绍GAN网络的各种变种以及GAN网络在故障诊断中的应用案例。
4. 从文献中可以看出，上述各种GANs在机械故障诊断中已被相继研究。然而，在存在极其有限的故障样本的情况下，仍需要解决以下问题，以进一步提高诊断性能。
   1. 如上所述，应用于GAN中的生成器的输入大多是随机噪声或带有一些标签的随机噪声，这些噪声不能完全捕获每个训练样本的信息，并使其难以提取多样性和代表性的特征表示。
   2. 上述GAN大多生成原始的时域振动信号或频谱信号，而没有涉及提取振动信号的时间-频谱的相关性特征。
   3. 在上述研究工作中，用于训练GAN的现有故障样本数量通常每类超过20个。
5. 描述本文的主要研究点是在极少样本的样本生成情况。
6. 为了解决这些问题，提出了一种改进的局部融合GAN（ILoFGAN），用于在极其有限的样本下对行星齿轮箱进行数据扩充。本研究的主要贡献如下：
   1. 为了生成足够高质量和多样性的行星齿轮箱时频图，ILoFGAN用于从有限的信息中充分提取值，并融合极少数样本的局部特征。
   2. 为了帮助局部融合模块提高局部特征匹配的精度和灵活性，构造了一种嵌入多头注意力（MHA）模块的新型生成器，用于挖掘时频图中的各种关键局部特征。

# 相关工作-LOFGAN(local fusion GAN) 

LOFGAN的思路很直观和简洁，主要分为以下几步：

假设有k张图片$X = {x_1,...,x_k}$。确定其中的一张作为base，其他的作为reference。

1. 局部选择：在base中选择一个局部位置，区域大小由参数$\eta$确定。

2. 局部匹配：在确定选择的是哪一部分局部特征后，计算Reference图片特征对应位置与Base图片的相似度并构建相似度矩阵：

   $\begin{equation} M^{(i, j)}=g\left(\phi_{\text {base }}^{(i)}, f_{\text {ref }}^{(j)}\right) \end{equation}$

​	然后就可以找到与Base图片最相似的局部特征进行融合。

3. 局部替换：对选定的局部特征，我们此时有k-1个候选可用于替换的局部特征，那就可以将所有局部表示进行融合，并且将base特征中原始的部分替换掉：

   $\begin{equation} \phi_{\text {fuse }}^{(t)}=\alpha_{\text {base }} \cdot \phi_{\text {base }}^{(t)}+\sum_{i=1, \ldots, k, i \neq \text { base }} \alpha_{i} \cdot \phi_{\text {ref }}^{(i)}(t) \end{equation}$

4. 构建局部重构损失函数：里的局部重构就是将上述特征层面的局部融合在图像层面实现，具体来说，记录下了替换的base图片的特征位置，然后将各个位置还原映射到图片大小，这样就得到了对应位置的经过局部融合后的图片，对这一位置进行局部重构：

   $\begin{equation} \mathcal{L}_{\text {local }}=\|\hat{x}-\operatorname{LFM}(X, \boldsymbol{\alpha})\|_{1} \end{equation}$

其思路如图：

![fig.2](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.2.png)

![fig_LoFGAN](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig_LoFGAN.png)

# 提出的方法

## 多头注意力机制约束的全新生成器

与卷积运算中局部感受域的缺陷相比，注意力机制[31]可以全局提取输入特征中任意两个位置之间的互信息。一般情况下，采用查询-键值模式来获取特征的注意力权重，以有效捕捉全局信息，用下面的函数表示：

$$\text{Attention}(Q,K,V)=\text{SoftMax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$

MHA机制不是在单注意力函数中使用d维查询、键和值，而是分别对查询、键、值进行e次的线性可学习映射，以获得相应的$d_k$、$d_k$和$d_v$维的向量。然后，并行计算每个线性映射结果的注意力函数，以获得$d_k$维度的输出。最后，将这些值连接在一起，并再次进行线性映射，以获得如下最终结果：

$MultiHead(Q,K,V)= Concat (head_1,...,head_e)W^O$

其中：

$headi = Attention(QW^Q_i ,KW^K_i ,VW^V_i )$

其中 $W^Q_i ∈ R^{d_{model}×d_q}$ $W^K_i ∈ R^{d_{model}×d_k}$$W^V_i ∈ R^{d_{model}×d_v}$分别表示查询、键、值和全局向量的线性投影的参数矩阵。因此，嵌入生成器的MHA模块可以同时关注来自不同位置的不同子空间信息。图3显示了MHA模块的结构。

![fig.3](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.3.png)



该模型的生成器主要包括局部融合模块、编码器和解码器。编码器由一个MHA模块和六个卷积块组成。这些卷积块中的每一个都包含卷积层、Leaky校正线性单元（Leaky-ReLU）激活函数和批处理归一化（BN）层。编码器可以快速获得故障样本的全局信息，并更加关注时频图中能量分布的关键局部特征，这有助于LFM更准确地找到基本图像和参考图像之间的局部特征对应位置。此外，多头模式有助于生成器挖掘各种局部特征，使特征向量F包含更多的互信息，这有助于LFM以多种方式融合局部特征，提高生成样本的多样性。解码器的结构与编码器对称，编码器包括一个MHA模块、两个卷积块和四个上采样卷积块。因此，MHA模块使解码器能够更多地关注生成图像的关键细节，以提高相似性的质量。

## ILoGFAN的训练

构建的ILoFGAN的结构如图所示。4，主要由生成器和鉴别器组成。鉴别器使用四个残差块作为特征提取器，其中包含两个卷积层、一个平均池化层和一个残差链路。最后，使用两个全连接层分别评估图像的真实性和分类。

![fig.4](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.4.png)

以下目标函数以及局部重建损失用于指导生成器G和鉴别器D的训练过程：

$$\begin{gathered}
\mathcal{L}_{G} =\mathcal{L}_{\mathrm{adv}}^{G}+\lambda_{\mathrm{cls}}^{G}\mathcal{L}_{\mathrm{cls}}^{G}+\lambda_{\mathrm{local}}\mathcal{L}_{\mathrm{local}}^{G} \text{(7)} \\
\mathcal{L}_{D} =\mathcal{L}_\mathrm{adv}^D+\lambda_\mathrm{cls}^D\mathcal{L}_\mathrm{cls}^D (8) \\
\mathcal{L}_{\mathrm{adv}}^D =\max(0,1-\boldsymbol{D}(X))+\max(0,1+\boldsymbol{D}(z)) \text{(9)} \\
\mathcal{L}_{\mathrm{adv}}^G =-\boldsymbol{D}(z) (10) \\
\mathcal{L}_{\mathrm{cls}}^D =-\log P(c(X)|X) \left.\left(\begin{matrix}{11}\\\end{matrix}\right.\right) \\
\mathcal{L}_{\mathrm{cls}}^G =-\log P(c(X)|z) \text{(12)} 
\end{gathered}$$

其中X表示输入图像，$c(X)$表示图像的类别，$z=G(X,α)$表示生成的图像，$L^D_{adv}$表示鉴别器的对抗性损失，$L^G_{adv}$表示生成器的对抗性损失，$L^D_{cls}$表示鉴别器的分类损失，$L^G_{cls}$表示生成器的分类损失以及$λ^D{cls}$，$λ^G_{cls}$，和$λ_{local}$分别是相应鉴别器的分类损失、生成器的分类损失和生成器的局部重建损失的正则化参数。

## 所提出的故障诊断方法的框架

所提出的方法的框架如图5所示，主要由以下步骤组成：

1. 获取行星齿轮箱在各种工况下的振动信号，故障类型的振动信号极少数。然后，通过连续小波变换将它们变换为相应的时频图样本。
2. 建立ILoFGAN模型，将MHA模块嵌入生成器中，以提高生成质量。然后，用极少数的时频图训练ILoFGAN。
3. 使用ILoFGAN经过训练的生成器为每种故障类型生成大量生成的样本。采用两个评价指标来评估生成样本的相似性和多样性。
4. 生成的样本和原始样本被混合并输入到卷积神经网络（CNN）中，用于特征提取和故障诊断。

![fig.5](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.5.png)





# 实验

## 数据集

1. gearbox measurements from the University of Connecticut
2. The planetary gearbox dataset of this case was collected from the drivetrain dynamic simulator from Southeast University

## 对比方法

1. LoFGAN
2. VAEGAN
3. WGAN_GP
4. ACGAN
5. DCGAN



## 实验结果

![table3](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/table3.png)



![fig.9](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.9.png)

![fig.12](/images/Data-Augmentation-and-Intelligent-Fault-Diagnosis-of-Planetary-Gearbox-Using-ILoFGAN-Under-Extremely-Limited-Samples/fig.12.png)



# 总结

本文的创新主要在于将新的小样本生成GAN用到了机械故障诊断中，其次就是在生成器的编码器和解码器中加入了多头注意力机制。

主要还是新领域的应用。



