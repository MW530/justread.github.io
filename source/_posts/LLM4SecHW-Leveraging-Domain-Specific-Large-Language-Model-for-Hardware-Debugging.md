---
title: >-
  LLM4SecHW:-Leveraging-Domain-Specific-Large-Language-Model-for-Hardware-Debugging
tags:
  - Hardware
  - LLM
categories: Hardware
thumbnail: ../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/LLM4SECHW.png
journal: AsianHOST(含金量有，但不是CCF)
date: 2024-08-02 15:17:58
---

# 创新点

1. 提出了一种利用版本控制数据汇编开源硬件设计缺陷及其修复步骤的独特数据集的新方法。(貌似是上一篇文章中有的？会议扩展？)
2. LLM4SECHW 根据构建的数据集对 70 亿个参数 LLM 进行微调，从而能够识别和纠正硬件设计中的错误。该框架是将 LLMs 应用于自动化硬件错误检测和纠正的开创性方法。此外，它还为其他研究领域微调特定领域 LLM 的实际应用提供了可参考的工作流程。



# 方法

![LLM4SECHW](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/LLM4SECHW.png)



# 实验

![LLM_used](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/LLM_used.png)



![test-result](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/test-result.png)



## 问答实际情况

![fituned-response](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/fituned-response.png)

![original-Falcon](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/original-Falcon.png)

![original-LLama2](../images/LLM4SecHW Leveraging Domain-Specific Large Language Model for Hardware Debugging/original-LLama2.png)

# 总结

主要贡献是收集了一个数据集、同时微调了几个主要的大模型。
