---
title: A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings
tags:
  - IFD
categories: IFD
thumbnail: /images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.1.png
journal: Measurement(IF:5.6)
date: 2023-11-23 20:32:53
---

# 引言

1. 滚动轴承是旋转机械的关键部件。对其进行诊断非常重要。然后介绍了深度学习方法：DAE；DBN；LSTM。此外，由于任何机械部件的振动响应都可以被视为不同振动源的卷积混合，因此由于其卷积运算性质，将CNN应用于轴承故障检测更合理可行。

2. 详细介绍了基于深度学习的故障诊断，再次说明了为什么基于CNN的方法效果更好。介绍了一系列的基于CNN的故障诊断方法。

3. 在现有的大多数基于CNN的检测方法中，故障特征提取往往被忽视（不太理解什么叫做“故障特征提取往往被忽视”）；然后提出softmax分类器性能不行，用深度森林分类器（deep forest (gcForest) classifier）效果更好；然后列举了CNN+gcForest在其他领域的应用；然后提出因为振动测量是一维时间序列信号，需要采用预处理方法将时间序列转换为CNN gcForest模型的适当输入，但现在没有研究对其进行处理。

4. 为了弥补上述研究空白，本研究提出了一种新的混合深度学习模型，该模型使用CWT预处理来结合CNN和GCForestf进行轴承故障诊断。主要包含三个部分：

   1. 使用CWT将轴承振动数据转换为图像。
   2. 使用CNN卷积和池化操作来提取故障特征。
   3. 使用gcForest和通过级联森林策略分类。

   

# 提出的方法

## Signal-to-Image conversion

CWT用于在图像中显示振动信号的时频特性。

![fig.2](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.2.png)

​                                    

## CNN-gcForest模型

gcForest模型是随机森林（RF）的深度学习衍生物。在RF中，原始数据集$x$首先用于使用Bootstrap策略构建$l$个子数据集。然后，每个子数据集都被用来构建一个决策树，因此，所有子数据集将生成一个由$l$个决策树组成的森林。每个决策树将生成一个输出，RF的最终输出由投票或平均策略确定。基于RF，gcForest处理每个决策树，通过计算训练数据集中不同类别的百分比来生成类别的概率分布。因此，gcForest的输出是林中所有决策树的概率分布的平均值。



采用多粒度扫描（MGS）和级联森林来实现gcForest中的深度学习。MGS旨在从输入图像中提取有用的信息。让我们用一个分类问题来描述MGS。首先，通过滑动窗口（窗口大小为$k$）扫描每个灰度图像（N×N矩阵；N是图像的大小），生成$S$个子图像矩阵；每个子图像是一个$k×k$矩阵。如果滑动步长为$j$，则$S＝[(N-k)/j+1]^2$。然后，使用每个子图像同时训练一个完全随机森林和一个随机森林，以产生特征向量。每个森林的输出向量中有$C$个元素，对应于训练数据集的$C$类标签的概率。通过级联两个训练的森林模型的输出向量，为每个子图像获得$2C$元素特征向量。因此，对于每个灰度图像，两个森林模型将生成维度为$S×2C$的特征矩阵。最后，将特征矩阵的每一行连接起来，生成一个$2×S×C$元素概率矢量作为MGS针对每个灰度图像的输出。应该强调的是，可以采用多个滑动窗口来扫描灰度图像，以生成每个灰度图像的输出概率向量。在本研究中，在MGS处理中使用了一个滑动窗口。灰度图像的大小为$N＝28$。滑动窗口的大小为$k＝26$，滑动步长为$j＝1$。因此，子图像的数量为$S＝9$。



级联森林是gcForest执行深度学习策略的核心。它接收MGS的概率向量并输出最终的分类结果。级联森林采用多层结构，每层有两个完全随机森林和两个随机森林。与MGS中的森林类似，级联森林中的每个随机森林模型都输出一个C元素概率向量，因此，每一层的输出维度为4C。在训练过程中自适应地确定层的数量，其中在每一层中使用k倍交叉验证来检查训练性能。如果与前一层相比，当前层的训练性能没有提高，则级联林将在下一层停止生长。



图4描述了级联森林的训练。对于每个灰度图像，第一层的输入是来自MGS的$P(=S×2C)$概率元素。然后，第一层的输出（即4C概率元素）与原始P个概率元素级联，以形成新的矢量（即4C+P个概率元素）作为第二层的输入。重复类似的串联以生成以下层的输入向量，直到最后一层。对最后一层的四个森林模型的输出进行平均，以产生C类的最终概率。将最终概率中的最大值作为gcForest分类结果。

![fig.5](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.5.png)

## 模型概述

开发的CNN-gcForest模型概述如图5所示。首先，使用CNN从灰度图像中提取有用的特征。CNN中的FC5层输出m元素特征向量，其中m是该层中神经元的数量，作为gcForest模型的输入向量。在gcForest中，将特征向量转换为$n×n(n=\sqrt m)$。MGS输出随后用于训练级联森林。最后，级联森林输出最终的分类结果。

![fig.1](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.1.png)



# 实验

## 数据集

1. CWRU
2. XJTU-SY

![fig.8](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.8.png)

![table.1](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/table.1.png)



## 实验结果

![fig.9](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/fig.9.png)

![table.3](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/table.3.png)

![table.5](/images/A-hybrid-deep-learning-model-for-fault-diagnosis-of-rolling-bearings/table.5.png)

# 总结

将CNN中的softmax分类器换成了深度森林分类器。即混合了CNN和gcForest。但是这句话“在现有的大多数基于CNN的检测方法中，故障特征提取往往被忽视”，不太能理解，CNN不就是特征提取吗？







