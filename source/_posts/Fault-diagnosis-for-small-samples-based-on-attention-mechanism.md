---
title: Fault-diagnosis-for-small-samples-based-on-attention-mechanism
tags:
  - IFD
  - small samples
categories: IFD
thumbnail: /images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/fig.4.png
journal: Measurement(IF:5.6)
date: 2023-12-02 22:39:09
---

# 引言

1. 随着工业物联网的发展，旋转机械设备的不确定性显著增强，这成为了一个巨大的挑战。在长期运行过程中，关键部件容易损坏，这将降低工厂效益，甚至造成人员伤亡和生态污染。因此，监测旋转机械设备的状态具有重要意义。
2. 过去几年中，基于信号分析、蜂群智能进化和机器学习的故障诊断方法不断涌现 。然而，这些方法过于依赖专家的先验知识，且特征提取需要人工完成，因此难以处理大数据和学习高级特征。此外，时间复杂度相当高的相关算法也无法保证找出全局最优。最后，面对复杂多变的工业数据，浅层模型很难达到理想的效果。
3. 介绍了深度学习及其在故障诊断中的应用。
4. 介绍小样本故障诊断以及各种应用。
5. 对于小样本，他们或者**利用模型的正则化技术和特征提取优势**，或者**根据真实样本的分布生成大量高质量样本**，或者**应用元学习和迁移学习等新兴机器学习技术**。
6. 大卷积核的设计有利于增强鲁棒性，而深度小卷积核的设计则能有效提取抽象特征。此外，振动信号中的时间步长信息也不容忽视。与 CNN 相比，RNN 刚刚能满足要求。
7. 介绍了RNN、LSTM以及GRU。假设信号只向前传播信息也是不恰当的，因此，性能与 BiLSTM 相似、参数较少且可前后传播信息的 **BiGRU** 是一个不错的选择。
8. 虽然以往的方法取得了相对令人满意的效果，但深度学习模型往往需要大量样本才能实现理想的泛化效果。然而，由于标注数据相对较少，模型往往无法充分学习有限样本中的各种有效特征，容易出现过拟合，增加了学习难度。此外，各种最新的激活函数和梯度下降反向传播算法在小样本下的故障诊断中还没有得到深入的比较探索。最终，由于不同工况的干扰，效率难以保证，这就提出了更高的要求。
9. 因此，针对模型的正则化技术和特征提取优势，提出了一种基于关注机制和 BiGRU 的双路径卷积的新型小样本故障诊断方法。卷积层旨在提取信号的高频特征。本文的主要贡献如下：
   1. 针对小样本故障诊断，从正则化和模型结构出发，提出了一种基于设计注意力机制和 BiGRU 的新方法，并首次探讨了 LSR、激活函数和反向传播算法的影响。同时，所提出的方法具有更高的测试精度。
   2. 讨论了注意力机制和 BiGRU 对训练样本比例的敏感性，其中提出的注意力机制可以捕捉振动信号的信道和空间信息。然后，在 BiGRU 之后设计 GAP 有利于提高诊断性能。此外，还利用了可视化技术来更好地理解 DCA-BiGRU 中的区块。
   3. 针对实际工业数据中包含的噪声，提出了一种基于预训练的小样本转移诊断框架。实验结果证明，与其他复杂工况下的轴承和齿轮箱诊断模型相比，它具有出色的泛化能力、适应性和鲁棒性。



# 相关工作

## CNN

![fig.1](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/fig.1.png)

## 双向门控循环单元

![fig.2](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/fig.2.png)

GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

见Fault-diagnosis-of-rotating-machinery-based-on-recurrent-neural-networks。



# 提出的方法

在故障诊断中，BiGRU 在当前时刻的输出状态是由上一时刻和下一时刻的状态共同决定的。当然，最后一个隐藏神经元的输出通常会作为诊断的最终隐藏特征，因为它具有最丰富的特征。不过，这种策略会忽略其他 GRU 单元学习到的信号特征。

因此，提出了一种名为 DCA-BiGRU 的智能故障诊断方法，它由数据增强、双路径卷积、关注机制、BiGRU、GAP 和诊断层组成，如图 4 所示。

![fig.4](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/fig.4.png)

如图 3 所示，在实际应用中，基于 DCA-BiGRU 的故障诊断具体步骤如下：

1. 获取原始信号，实现数据分割和标准化。
2. 将信号分为训练样本、验证样本和测试样本。
3. 提出模型结构和诊断方法。
4. 离线训练：使用训练集和正则化策略来训练和保存最佳参数。
5. 在线诊断：应用测试集验证模型性能，或加载预训练参数并微调整个模型，利用参数共享迁移学习实现及时训练和故障诊断。

## 双路径卷积和特征融合

双卷积层采用两条路径提取信号的高低频特征。在一条路径上，利用两个较大的卷积核来学习低频特征。正如第 2.1 节所述，较大的卷积核可以增强对噪声的鲁棒性。在另一条路径上，采用较小的卷积核来深化神经网络，该网络集成了四个非线性激活层，以提高判别能力。两者的结合拓宽了模型并提取了多尺度特征，为 BiGRU 进一步学习高级特征奠定了基础。最后，通过元素乘积融合特征，每个通道都包含丰富的特征。

为增强 DCA-BiGRU 在不同领域的适应性，利用 AdaBN 代替 BN，调整源领域到目标领域的统计信息，以提高泛化能力。

## 提出的信号关注机制

注意机制和 LSR 可被视为对成本敏感的学习方法，而 1D-Meta-ACON 可被视为元学习的一种手段。对于小样本，这些正则化方法将有助于模型的泛化和领域适应性。

### 标签平滑正则化

交叉熵损失（CE，l0）往往集中于一个方向，导致调节能力差。因此，增加平滑系数ε来提高诊断的正确率，减少错误诊断，这有助于消除模型的过度自信，提高学习能力。LSR(l) 不仅能提高泛函能力，还能校准模型。它主要应用于图像识别领域，但在故障诊断领域却鲜有研究。

在真实标签下，神经网络会促使自身往正确标签和错误标签差值最大的方向学习，在训练数据较少，不足以表征所有的样本特征的情况下，会导致网络过拟合。

label smoothing可以解决上述问题，这是一种正则化策略，主要是通过soft one-hot来加入噪声，减少了真实样本标签的类别在计算损失函数时的权重，最终起到抑制过拟合的效果。

增加label smoothing后真实的概率分布有如下改变：

![label smooth](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/label smooth.png)

交叉熵损失函数的改变如下：

![lable smooth lose](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/lable smooth lose.png)

最优预测概率分布如下：

![label smooth prob](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/label smooth prob.png)

比如在ImageNet上的表现如下：

![label smooth Imagenet](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/label smooth Imagenet.webp)

### 提出的一维信号关注机制

图 5 中提出了一种一维信号关注机制，它可以告诉我们哪些模型需要关注原始信号。

为了计算通道之间的注意力，压缩输入特征矩阵的维数是必不可少的，通常采用**全局池化**。此外，与关注整体信息的GAP相比，我们认为全局最大值池（GMP）提供了关键的脉冲(𝑥𝐺𝑀𝑃 )对于信号特征矩阵(𝑥),理论上，决定性脉冲被视为故障诊断的主要判别标准，因此GMP比GAP更适合于所提出的注意块，这将通过实验进行验证。

第 c 个通道 GMP 的计算公式为公式 (10)。

$x_{GMP}^{\mathrm{c}}=\max_{0\leq j<d}x_{c}(1,j)$

此外，为了捕捉空间位置信息，最好能建立 $x_{GMP}$ 与 x 之间的关系，因此将它们串联起来，送入共享 1 × 6 的卷积映射函数 F1。如式（11）所示，对这种依赖关系进行编码，就得到了中间特征连接矩阵 $f$。

$f=\delta(F_{1}[cat(x,x_{GMP})])$

其中，δ 是 1D-Meta-ACON 激活函数。

然后，f 被分割成 x′ 和其他。由于变换后的原始特征矩阵 x′ 不仅包含临界脉冲频谱信息，还包含原始信号特征 x 的信息，因此只保留 x′。另一个 1 × 1 卷积映射函数 F2 将 x′ 转换为与 x 相同数目的通道，如式（12）所示。

$𝑔 = 𝜎[𝐹2(𝑓𝑥′ )]$

最后，输出 $y_c$ 如公式 (13) 所示。

$𝑦_𝑐 = 𝑥_𝑐 ⊗ 𝑔_𝑐$

![fig.5](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/fig.5.png)

### 改进后的 1D-Meta-ACON

针对振动信号的非线性特点，在所提出的注意力模块中应用了一种新的激活函数 Meta-ACON[33]。它既不是 ReLU 也不是 Swish，但两者都得到了考虑，并概括为一种一般形式。这是一种可以学习是否激活的形式。

是否激活神经元取决于平滑系数𝛽𝑐 ,以便动态地和自适应地消除不重要的信息。这与所提出的1D信号注意机制的思想类似，它关注信号中的中心部分，有助于提高泛化能力和传输性能。受此启发，它转变为𝛽𝑐适用于1D信号，1D元ACON，其公式如下。

$\beta_c=\sigma[F_4(F_3(\frac{1}{D}\sum_{d=1}^{D}x_{c,d}))]$

1D-Meta-ACON 是一种通用形式，它不仅能解决死神经元问题，而且只需要几个参数就能学会是否激活。

### AMSGradP

AdaBN 与 BN 一样，有助于提高模型的泛化能力和规模不变性。然而，Heo 等人指出带动量的梯度下降（GDM）会导致有效步长在反向传播过程中迅速减小，从而导致收敛速度减慢，甚至出现尖锐的最小值，因此提出了 AdamP [34]，它可以在优化更新过程中放弃径向分量，调节权重规范的增长，延缓有效步长的衰减，从而无障碍地训练模型。

在这项研究中，小样本很容易收敛到局部最优。遗憾的是，作者没有给出更先进的 AMSGrad 的改进方法。受此启发，参考文献 [34] 的思想被引入 AMSGrad。[34] 的思想引入到 AMSGrad 中，称为 AMSGradP。在附录中，算法 1 概述了 AMSGradP 的伪代码。

# 实验结果

## 数据集

1. Case Western Reserve University
2. University of connecticut



## 对比方法

1. PCA-SVM
2. DCNN-BiGRU
3. DCNN
4. DCA

以 G-mean 为指标的对比模型有 PCA-SVM（M1）、DCNN-BiGRU（没有注意力机制，M2）、DCNN（没有注意力机制和 BiGRU，M3）和 DCA（无 BiGRU，M4）。为了避免随机影响，每个实验重复五次，得到如图 7 所示的误差条，A→A 表示训练集→测试集。X 轴表示训练的比例（α）。同时，记录不同模型、不同负载在不同 α 条件下的运行时间，直至提前停止，如表 6 所示。

## 实验结果

![table.4](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/table.4.png)

![table.5](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/table.5.png)

![table.6](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/table.6.png)

![table.10](/images/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/table.10.png)



# 总结

本文的主要应用场景是小样本，但是提出的模型并没有体现具体场景。只是下面这么说了：

> 注意机制和 LSR 可视为对成本敏感的学习方法，而 1D-Meta-ACON 可视为元学习的一种手段。对于小样本，这些正则化方法将对模型的泛化和领域适应性做出贡献。

技术上的创新点在于改进了1D-Meta-ACON，其他的都是使用。
