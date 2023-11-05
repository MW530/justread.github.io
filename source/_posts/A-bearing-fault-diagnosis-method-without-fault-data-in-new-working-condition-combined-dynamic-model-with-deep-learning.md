---
title: A bearing fault diagnosis method without fault data in new working
  condition combined dynamic model with deep learning
date: 2023-11-01 16:33:38
tags: 
- IFD
- zero-shot
- GAN
categories: IFG
thumbnail: /images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.12.png
journal: Advanced Engineering Informatics (IF:8.8)
---





# 引言

1. 转子轴承重要，然后故障数据往往不够充足。

2. 介绍基于机制的方法。然而，上述方法的机理知识复杂，模型构建费时费力，不适合在工业场景中进行快速准确的诊断。

3. 首先介绍基于信号处理的方法。然后介绍了数据驱动方法：*数据驱动故障诊断方法是近年来的研究热点之一，它可以基于采集的数据和人工智能算法实现轴承故障的快速诊断。*

   1. 数据驱动的不走主要有三步：1） 数据采集，2）手动特征提取和特征选择，以及3）自动故障识别

   这些人工智能方法在故障诊断任务中取得了巨大成功，但当标记的训练样本有限时，其诊断性能显著下降。因此，获取足够的故障样本对提高诊断模型的鲁棒性和适应性具有重要意义。

4. 介绍数据模拟和数据对齐。*如上所述，基于仿真模型和GAN，已经解决了小样本问题和故障样本缺失问题。然而，没有考虑模拟数据和实际数据之间的误差，导致诊断模型的准确性较差。*

5. 介绍了机械特征生成器和数据模型生成器。然而，在实际诊断场景中，数据模型通常面临数据不足的问题，机制模型中出现模拟性能不足的情况。因此，经常组合模型采用机构模型生成大量仿真数据以满足数据模型的数据要求，并采用GAN模型来解决机构模型仿真性能不足的问题。

6. 提出一个问题，由于机械最开始并没有故障样本，因此提出零样本问题。*本文着重研究了新情况下零样本诊断问题，包括交叉操作条件和无故障样本问题。*



# 创新点

1. 针对新情况下零样本的任务，提出了一种基于机制和数据相结合的故障诊断方法。与目前流行的转移方法相比，该方法在新的条件下不需要真实样本，可以实现预测在线故障诊断。
2. 轴承动态模型用于模拟各种数据，克服了故障样本采集困难的缺点。此外，GAN还弥补了模拟数据与实际数据之间的差距。建立了故障顺序转换模型，探讨了工况变化的机理。

# 方法

## 简介

### 建立轴承动力学模型，模拟不同工况下的故障数据。- 通过机械原理模拟真实的数据。

轴承动力学模型从理论上有效揭示了轴承的故障机理和个体特征。轴承系统被建模为具有四自由度（DOF）的非线性动力学模型，该模型在中提出，用于模拟振动响应。本文的仿真模型由四个主要部件组成：内座圈、滚珠、外座圈和基座作为集中质量单元进行仿真。考虑轴承几何形状、材料参数、受力分析和速度载荷信息的影响，建立了如图所示的动力学模型。

![fig.4](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.4.png)

### 提出了故障顺序转换模型，探讨了工况变化的机理。- 通过变换将生成的模拟数据转换到工况2的数据。

为了研究工况的变化机制，在文献的基础上，建立了故障阶数转换模型，将旧工况下的样本数据转换为新工况。



这里，使用模拟数据而不是真实世界的数据来找到映射Ψ（‧），该映射将故障特征从**旧条件**转换为**新条件**。与真实世界的数据相比，无噪声模拟信号更适合于训练转换模型。



### 采用参数迁移加速模型训练。- 用真实数据微调模型

![fig.10](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.10.png)

### 采用GAN模型将模拟数据转换为与真实数据相似的生成数据。 - 生成数据

![fig.11](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.11.png)



## overview

![fig.12](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.12.png)

# 实验

## 数据集

- Paderborn University

## 实验结果

![table6](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/table6.png)



![fig.19](/images/A-bearing-fault-diagnosis-method-without-fault-data-in-new-working-condition-combined-dynamic-model-with-deep-learning/fig.19.png)

# 启示

1. 了解了基于GAN的零样本学习在故障诊断领域的基本框架：
   1. 机械原理数据生成
   2. **数据概率对齐**
   3. 训练GAN
   4. 微调GAN
   5. 使用生成的GAN生成数据
   6. 使用生成的数据训练分类器
   7. 使用分类器预测新的无标签样本
2. 了解了机械数据生成的方法
3. 但是模拟数据和真实数据概率对齐的模型文章中没有具体给出
4. 文章只对比了自己的模型，没有对比新的模型
