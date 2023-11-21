---
title: >-
  Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation
tags:
  - IFD
  - GAN
  - IRT
categories: IFD
thumbnail: /images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.3.png
journal: IEEE Transactions on Industrial Informatics(IF:12.3)
date: 2023-11-20 14:16:52
---

# 引言

1. 转动机械在现代工业中很重要，因此有必要采用策略对其进行维护。
2. 介绍了深度学习方法在故障诊断的应用。然而，在工业实践中获得足够的故障样本是昂贵的。因此提出了一系列的方法来处理小样本问题。然后介绍了一些小样本学习的技术，比如元学习等。在过去的几年里，GAN及其变体在小样本故障诊断领域取得了显著的成果。然后介绍了一系列的基于GAN的方法。
3. 从上述文献中可以发现，现有的使用GAN的小样本故障诊断研究大多是通过分析在稳定速度下收集的加速度信号来进行的。然而，在工业实践中存在以下问题：
   1. 在大型旋转机械内部安装加速度传感器有时会受到限制。此外，当加速度传感器的安装位置远离关键部件时，特性信息可能会受到干扰。
   2. 为了满足实际生产要求，旋转机械的转速通常需要改变，这可能会导致采集到的信号具有更强的非平稳特征。
4. 旋转机械故障引起的过度摩擦会产生异常高温信息，红外热成像可以有效地收集这些信息。与振动分析相比，IRT具有非接触、稳定性强、抗噪声等优点。然后介绍了IRT的相关方法。然而，上述研究的成果也是基于足够的IRT图像。据我们所知，GAN和IRT图像在机械故障诊断领域的联合应用仍处于起步阶段，值得进一步探索。
5. 然而，为了生成高质量和高效的IRT图像，迫切需要解决基本GAN的以下问题。
   1. 由于对抗性训练的特殊性，梯度消失很容易发生。
   2. 基本的GAN不能密切关注IRT图像的全局热相关性特征。
   3. 训练是耗时的，因为生成的样本的质量通常需要手动评估。此外，鉴别器和生成器之间的训练平衡有时可能会失败。
6. 本文提出了一种用于有限样本和速度波动情况下旋转机械故障诊断的数据生成方法——双阈值注意力引导GAN（DTAGAN）。通过对比实验验证了该方法的有效性和优越性。我们文章的主要贡献如下:
   1. 利用Wasserstein距离和梯度惩罚（GP）设计了改进的损失函数，以稳定训练过程。
   2. 构造了注意力引导GAN，重点研究了断层IRT图像的全局热相关特征。
   3. 开发了一种新的双阈值训练机制，以提高生成的IRT图像的质量和训练效率。

# 相关工作

## GAN 网络

基于鉴别器D和生成器G的对偶网络结构，GAN的对抗性训练思想来源于博弈论。G接收随机噪声信号，然后通过上采样将输入映射到类似于真实样本的生成样本。D输出样本属于多层特征提取后的真实样本的概率。



## 注意力模块

由于卷积核的感受野的限制，基于卷积的GAN通常需要多层来学习长距离像素之间的相关性特征。

由于卷积核的感受野的限制，基于卷积的GAN通常需要多层来学习长距离像素之间的相关性特征。卷积层的堆叠大大增加了模型参数和复杂度，多层卷积提取的相关信息很容易被忽略。注意力模块通常用于计算数据中不同位置之间的相关性。作为注意力模块的一员，可以引入自注意力来引导基于卷积的GAN提取图像的全局特征。自我注意的原理如图1所示。

![fig.1](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.1.png)

假设在卷积运算后获得的特征图中存在n个像素。首先，使用权值为 $W_q$ 和 $W_k$ 的两个 1 ∗ 1 卷积核提取输入特征 $x$ 的特征，生成大小为 $n×n$ 的矩阵 $S$。接下来，通过Softmax函数来正则化矩阵S，以此来计算注意力得分矩阵$\beta_{j,i}$，其可以如下进行表示：

$$\begin{aligned}\beta_{j,i}=\frac{\exp(S_{ij})}{\sum_{i=1}^n\exp(S_{ij})},S_{ij}=\boldsymbol{q}(\boldsymbol{x_i})^T\boldsymbol{k}(\boldsymbol{x_j})\end{aligned}$$

其中，$S_{ij}$表示卷积输入特征$x$中第$i$个像素和第j个像素之间的相关性，$β{j,i}$表示需要关注$i$和$j$之间的相关性的程度。可以看出，在$S{i,j}$中考虑了$x$的全局信息。

最后，注意力模块处理后得到的输出特征$y_j$的计算如下：

$$y_j=\gamma o_j+x_j,o_j=W^\mathbf{o}\left(\sum_{j=1}^n\beta_{j,i}\times\boldsymbol{v}(\boldsymbol{x_i})\right)$$

其中$v(x_i)$是值矩阵$v(x)$的元素，$W_o$是大小为1×1的卷积核的权重，$o=(o_1, o_2, …, o_j，…,o_n)$表示自注意模块提取的特征值，γ表示可以使注意模块从近端到远端局部学习全局信息的尺度参数。

# 提出的方法

## 改进的损失函数设计

由于JS发散的离散性，在基于梯度下降的优化过程中存在梯度消失现象。基于Wasserstein距离设计的对抗性损失函数可以有效地解决上述问题[26]。Wasserstein距离的定义是：

$$\begin{aligned}W(p_{r(x)},p_{g(y)})=\inf_{\gamma\sim\prod{(p_{r(x)},p_{g(y)}}}E_{(x,y)\sim\gamma}[||x-y||]\end{aligned}$$

其中，$x$和$y$分别是来自$p_r(x)$的实样本和来自$p_g(y)$的生成样本，$inf_γ～ψ(pr(x),p_g(y))$是联合分布的下确界，$E(x,y)～γ[||x−y|]$是$x$和$y$之间的预期距离。与JS散度相比，Wasserstein距离可以更平滑地反映两个分布之间的差异。

然而，基本Wasserstein GAN（WGAN）中的权重裁剪可能会导致梯度爆炸，并且在多次权重裁剪操作后无法获得所需的函数映射。GP[27]是一种有效的方法，而不是权重裁剪，它基于生成样本和真实样本之间的分布进行计算，表示为:

$$gp=E_{\tilde{x}\sim P_{\tilde{x}}}\left[\left(\left|\left|\nabla_{\tilde{x}}D\left(\tilde{x}\right)\left|\right|\right|\right|_2-1\right)^2\right],\tilde{x}=\varepsilon x+\left(1-\varepsilon\right)y$$

其中ε来自均匀分布，$\nabla_{\tilde{x}} D(\tilde{x})$是D的输出的梯度，而$\parallel\nabla_{\tilde{x}} D(\tilde{x})\parallel_2$是其L2范数。最后，设计的新损失函数是：

$$\begin{aligned}L(G,D)&=-E_{x\sim P_{r(x)}}[D(x)]+E_{y\sim P_{g(y)}}[D(y)]+\lambda\times gp\end{aligned}$$

具体解析可以回顾[这篇文章](https://zhuanlan.zhihu.com/p/25071913)和[原文](https://dl.acm.org/doi/abs/10.5555/3305381.3305404)。



## 基于IRT的注意力导向GAN构建

在IRT图像中，分布在不同区域的热像素代表机器的即时运行状态，并且产生摩擦热的区域之间的距离通常很远。因此，在基于2-D CNN的鉴别器和生成器中引入了自注意模块，以提取IRT图像中的全局热信息。注意力引导GAN的结构如图2所示。详细内容如下两个方面。

![fig.2](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.2.png)

### 生成器结构

首先，通过全连接层将一维噪声扩展，然后通过整形运算将其转换为512×4×4矩阵。然后，通过多个转置卷积层的上采样操作，最终将其转换为3×128×128的IRT图像。在文献中，贾等人[21]声称，中高级特征图上的自注意模块比低级特征图上实现了更好的性能。因此，没有选择8×8和4×4的特征图，消融实验的其他训练效率记录在表I中。可以看出，除了32×32和16×16之外，其他特征图的效率都会显著下降。最后，将自注意模块放置在第三和第四卷积层之后，以协调IRT图像不同位置的细节。GELU（高斯误差线性单位）用作激活函数，以提供更平滑的梯度，定义为

$$\mathrm{GELU}(x)=0.5x\left(1+tanh\left[\sqrt{2/\pi}\left(x+0.044715x^3\right)\right]\right)$$

### 判别器结构

鉴别器和发生器的结构是完全对称的。在进行五层卷积特征提取后，将IRT图像转换为256×4×4的特征矩阵。最后，全连接层的输出表示输入图像属于真实IRT图像的概率。注意力模块被放置在与生成器相同的位置，以学习全局热特征，从而提高图像质量。在鉴别器中，选择leaky rectified linear unit（LReLU）作为每层的激活函数，以避免梯度消失。

### 双阈值训练机制

在基本GAN的训练过程中，基于梯度下降的优化算法具有以下特点。首先，鉴别器需要在训练的初始阶段稍微强大一点，让生成器学习真实样本的特征。然而，强鉴别器的输出接近0，这可能会阻止生成器执行梯度更新。其次，为了确保生成样本的质量，频繁设置冗余迭代次数，导致适应性和效率低下。

因此，理想的训练是鉴别器可以在早期快速更新参数，并在中后期调整优化速度。此外，当生成质量满足要求时，可以自动停止训练。因此，有必要在训练过程中自适应地采用不同的训练策略，以提高生成样本的质量并降低时间成本。

本文开发了一种新的双阈值训练机制，该机制可以自动采用不同的训练策略。

#### 第一阶段

当平均结构相似性（SSIM）值小于第一阈值时，鉴别器的两尺度固定学习率被设置为大于生成器，以实现快速训练。结合IRT图像的特点，SSIM[29]用于实时评估训练过程中生成的图像与真实图像之间的相似性，定义为：

![eq.9](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/eq.9.png)

其中，$μ_x$、$μ_y$、$σ_x$、$σ_y$和$σ_{xy}$分别是真实图像x和生成图像y的统计特征（均值、方差），$C_1=(0.01V)^2$、$C_2=(0.03V)^2$和$C_3=0.5C_2$，其中$V$是像素值。SSIM值越大表示两个图像之间的相似度越高。

详见[此文](https://zhuanlan.zhihu.com/p/93649342)。

#### 第二阶段

当平均SSIM值处于第一阈值和第二阈值之间的稳定状态时，应用循环余弦学习率[30]来调整鉴别器的性能，如下所示：

![eq.10](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/eq.10.png)

其中$η_t$、$η_{max}$和$η_{min}$分别是当前、最大和最小学习率，$T_max$和$T_cur$分别表示周期迭代和当前迭代。

#### 第三阶段

当平均SSIM值保持大于第二阈值时，训练自动停止。

在培训中引入双阈值机制有以下贡献。

1） 自适应地调整鉴别器的学习率以提高生成样本的质量。

2） 培训过程可以自动停止，以节省人工成本。

### 提出方法的总体框架

![fig.3](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.3.png)

该方法的总体框架如图3所示，主要包括以下步骤。

1. 通过IRT相机获得旋转机械在速度波动下不同健康状态的IRT图像，并随机选择有限数量的图像进行训练。

2. 创建DTAGAN，具体操作如下。
   1. 引入Wasserstein距离和GP。
   2. 设计了新的目标函数。构造了注意力引导GAN，重点研究了断层IRT图像的全局热相关特征。
   3. 开发了双阈值训练机制，以提高生成质量和效率。

3. 使用训练后的DTAGAN生成新的IRT图像，以扩展现有的小样本，并从各个角度评估其生成质量。

4. 使用足够的IRT图像来辅助转速波动下旋转机械的故障诊断。实验结果验证了该方法的有效性和优越性。



# 实验

## 数据集

自建

![fig.5](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.5.png)

## 对比方法

Method 1: DTAGAN

Method 2: Self-attention GAN (SAGAN) with the designed loss function; 

Method 3: Wasserstein GAN with GP; 

Method 4: basic SAGAN; 

Method 5: Deep convolution GAN; and 

Method 6: ACGAN are used for comparison to show the superiority of the proposed method.

需要强调的是，所使用的比较GANs的网络层数、卷积核参数、池化层和学习率是完全一致的。

## 实验结果

![fig.7](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.7.png)

![fig.8](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.8.png)

![fig.9](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/fig.9.png)

### 小样本情况

![table.4](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/table.4.png)

![table.5](/images/Dual-Threshold-Attention-Guided-GAN-and-Limited-Infrared-Thermal-Images-for-Rotating-Machinery-Fault-Diagnosis-Under-Speed-Fluctuation/table.5.png)

# 总结

1. 本文的最大的启发是使用IRT图片进行机械故障诊断。后期可以考虑振动信号和IRT图片进行多模态融合故障诊断。
2. 本文提出的方法其实不算创新，如Wasserstein distance和GP都是之前GAN网络中提出来的。







