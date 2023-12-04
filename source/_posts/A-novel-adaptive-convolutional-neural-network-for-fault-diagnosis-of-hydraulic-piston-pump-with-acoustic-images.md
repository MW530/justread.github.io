---
title: >-
  A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images
tags:
  - IFD
  - acoustic images
categories: IFD
thumbnail: /images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.3.png
journal: Advanced Engineering Informatics (IF:8.8)
date: 2023-12-04 14:11:35
---

# 引言

1. 液压柱塞泵在液压传动系统中至关重要，广泛应用于船舶设备和航空航天机械。故障可能导致停机和整个系统受影响，还可能威胁人身安全。因此，诊断和预测液压泵的故障对于确保工业生产的盈利能力和工人的安全非常重要。
2. 介绍了机器学习算法在故障诊断中的应用。尽管基于机器学习的方法可以对故障进行分类，但它们在特征提取方面严重依赖于工程师的经验和知识。因此，这些方法并不适用于具有高度非代表性特征的情况。
3. 介绍了深度学习方法在故障诊断中的应用。
4. 虽然基于 DL 的智能方法在故障诊断方面取得了一些不错的成果，但现有研究仍存在一些局限性：
   1. 基于 CNN 的研究大多集中在齿轮、齿轮箱、电机、转子和轴承的智能故障诊断方面。对泵，尤其是液压活塞泵的研究很少。
   2. 虽然一些基于深度 CNN 模型的进化智能算法已被用于智能故障诊断，但使用贝叶斯优化（BO）的方法却鲜有研究。
   3. 虽然机器学习已被应用于声学领域，但利用声学信号对机械进行智能故障诊断的研究还很少。

5. 因此，本研究的主要贡献如下：
   1. 研究对象是液压柱塞泵。故障诊断旨在识别五种不同的健康状况。
   2. 通过对关键 HPs 采用 BO，构建了用于液压柱塞泵故障诊断的自适应深度 CNN 模型。它可以精确识别各种健康状况。
   3. 所提出的智能方法使用声学信号。通过选择适当的小波基函数，利用连续波变换（CWT）将原始信号转换为时频分布。



# 基础理论

## CNN

## 贝叶斯优化

调整机器学习模型的 HP 通常被认为是一个黑箱优化问题，但训练过程的成本相对较高。因此，探索合适的优化方法具有重要意义和挑战性。

手动调整 HPs 很容易实现，效果也不错，但在很大程度上取决于专家的经验。此外，不同的模型和数据集可能有不同的最优 HPs 集。人工优化的结果很难再现。网格搜索是一种自动优化方法。它通过排列和组合来调整 HPs。随机搜索采用边缘分布。在实际应用中，它优于网格搜索 [40]。虽然这些方法可以自动完成参数调整的整个过程，但它们无法从以前的结果中获取信息，可能会尝试许多无效的参数空间。基于进化算法的各种智能优化方法，如**遗传算法**、**粒子群优化**等，可以利用**全局优化**的能力，但这些方法可能比较耗时。因此，开发一种更精确、更高效的智能算法就显得尤为重要。自提出用于机器学习以来，**BO 已被证明适用于 HP 调整** [41-43]。

![SMBO](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/SMBO.png)

常见的超参数优化算法有：

1. 人工调参（manul tuning）

2. 网格搜索（grid search）：网格搜索随着参数数量的增加呈指数级增长，因此对于超参数较多的情况，该方法面临性能上的问题。著名的支持向量机开源库libsvm使用了网格搜索算法确定SVM的超参数。

   ![grid search](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/grid search.png)

3. 随机搜索（random search）：随机搜索做法是将超参数随机地取某些值，比较各种取值时算法的性能，得到最优超参数值。

4. 贝叶斯优化（Bayesian Optimization）：是一种**informed** search，**会利用前面已经搜索过的参数的表现，来推测下一步怎么走会比较好，从而减少搜索空间，大大提升搜索效率。**

# 提出的方法

## 构建卷积神经网络

与传统的全连接网络相比，在CNN的构建中考虑了图像的拓扑结构。因此，CNN是图像处理中的一个强大工具。LeNet-5和AlexNet是深度CNN模型的两个典型代表，它们利用了自动特征学习在分类和识别中的潜力[50，51]。基于LeNet-5和AlexNet，建立了一种用于液压泵故障诊断的改进CNN。CNN模型的示意结构如图1所示。有两个Conv层，每个层后面都有一个池化层和两个FC层。为了平衡网络模型的优秀特征表示和较低的复杂度，选择相对较小的卷积核作为特征提取器。CNN中的激活函数是ReLU。最大池用于通过下采样来减少特征维度。softmax回归函数完成了预测和分类。

![fig.1](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.1.png)

## 优化超参数

由于人工优化 HPs 的局限性，因此使用 BO 自动优化参数。CNN 中需要优化的 HPs 如下：学习率 (LR)、历时 (epoch)、批量大小、核大小和卷积核数量。

![fig.2](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.2.png)

流程图如图 2 所示。每个 HP 设置为一个范围，选择 GP 回归建立目标函数模型。通过计算前一个数据点的后验概率，可以得到每个 HP 在每个值点的预期均值和方差。均值表示该点的最终预期结果。方差表示该点效果的不确定性。由于噪声对结果的影响，本研究选择了噪声 EI，以确保不会出现过多的递增和递减附加结果。优化目标是测试数据集上的分类准确率。目标函数不是一个恒等式，它可以表示分类准确率和 HP 的关系。

## 提出方法的步骤

所提出的智能故障诊断方法如图 3 所示。

实现过程主要包括以下五个步骤：

1. 声学传感器采集原始信号。
2. 利用 CWT 将获得的一维时间序列数据转换为图像。时频分布经变换策略处理后用作 CNN 的输入。
3. 通过设置 LeNet-5 和 AlexNet 模型的初始 HP，构建初步的深度 CNN 模型。
4. 利用贝叶斯算法优化模型的 HPs。首先，确定需要优化的 HPs。这些参数包括 LR、epoch、批量大小、内核大小和卷积内核数量。其次，为每个 HP 选择一个范围。第三，根据评估函数迭代 BO 循环。所讨论的目标函数实际上是模型性能与 HP 之间的关系。优化的目的是寻找能达到最佳性能的参数组。然后对具有最佳 HPs 的 CNN 模型进行训练和测试。最后，构建出改进的 CNN 模型。
5.  将带有 BO 的深度 CNN 应用于液压泵的故障诊断。

![fig.3](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.3.png)

# 实验

## 参数

![table.3](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/table.3.png)

![table.4](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/table.4.png)

## 数据集

液压轴向柱塞泵试验台如图 4 所示。模拟故障的实验就是在这个试验台上进行的。斜盘轴向柱塞泵有七个柱塞。泵的额定转速为每分钟 1470 转。相应的旋转频率为 24.5 赫兹。实验在燕山大学进行。为了从信号中获取更多的内部信息，防止出现混叠现象，采样频率设置为高于根据奈奎斯特采样定理计算出的理论值。实验期间，采样频率为 10 kHz。

![fig.4](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.4.png)





## 对比方法

1. Traditional LeNet-5
2. LeNet-5AP
3. Improved LeNet-5
4. CNN-BO-N
5. CNN-BO

## 实验结果

![table.5](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/table.5.png)

![fig.10](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/fig.10.png)

![table.6](/images/A-novel-adaptive-convolutional-neural-network-for-fault-diagnosis-of-hydraulic-piston-pump-with-acoustic-images/table.6.png)



# 总结

本文的两个点都是可以参考的：

1. 利用声音信号进行诊断
2. 利用优化算法自动优化网络的参数















