---
title: Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning
tags:
  - Diffusion Models
categories: Diffusion Models
thumbnail: ../images/Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning/Fig.1.png
journal: None yet
date: 2024-02-24 23:44:36
---

# 创新点

![Fig.1](../images/Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning/Fig.1.png)

解构了去噪扩散模型，使其更贴近于基本的去噪自编码器。

# 方法

1. 移除类别条件：
2. 结构 VQGAN：
3. 替换噪声调度器：
4. **解构Tokenizer：探索了标准 VAE、片段式 VAE、片段式 AE 和片段式 PCA 编码器。最终发现标准的PCA效果反而很好。**



# 实验

![Fig.6](../images/Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning/Fig.6.png)

![Fig.7](../images/Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning/Fig.7.png)

![Table.3](../images/Deconstructing-Denoising-Diffusion-Models-for-Self-Supervised-Learning/Table.3.png)



# 总结

化繁为简，很有意思的idea。

但是PCA效果为什么好，似乎没有深入的讨论，是一个方向。
