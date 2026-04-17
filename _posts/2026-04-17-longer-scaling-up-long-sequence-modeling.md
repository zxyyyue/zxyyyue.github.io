---
title: "[LONGER] Scaling Up Long Sequence Modeling in Industrial Recommenders"
date: 2026-04-17
categories: [notes]
tags: [long-sequence-modeling, paper-reading, recommender-systems]
permalink: /notes/longer-scaling-up-long-sequence-modeling/
author_profile: true
classes: note-page
---


> 论文地址：https://arxiv.org/pdf/2505.04421
>
> 公司：字节跳动（ReySys'25)

## 背景

以往长序列建模方法：

- 两阶段检索：GSU从原始超长序列中选和当前候选item最相关的top-k（k通常为100）个item，之后ESU端到端短序列建模，代表作SIM和TWIN
- 预训练用户embedding：预训练整个超长序列然后压缩用户user embedding（UE），然后转到下游的推荐模型。预训练的序列长度可以达到1000
- 记忆增强模型：多通道用户兴趣记忆网络(MIMN)为用户序列记忆提供了一种基于神经图灵机和内存归纳单元的结构，大型记忆网络(LMN)提出了一种基于乘积量化的分解的轻量级结构。内存增强推荐模型(MARM)提出了一种内存计算权衡范式，它将中间结果从计算密集型模块缓存。

HSTU用一堆相同自注意力层组成，残差连接进行长序列建模。Wukong用一种堆叠因子分解机和基于线性压缩块的交互架构，验证了推荐的scaling law。本文提出了用于GPU高效推进器的长序列优化器**LONGER**。将输入组织为global tokens和raw sequences，用一种inner-transformer的token合并方法来减少计算预算；应对用户超长序列中存在噪声，用一种混合注意力策略提高计算效率（减少约50%的FLOPS）并保持模型性能；工程优化方面（混合精度和激活重新计算的同步训练和服务框架，KV缓存服务策略）

## 方法![截屏2026-04-17 14.33.53](/Users/yue/Library/Application Support/typora-user-images/截屏2026-04-17 14.33.53.png)

### 构造输入

非序列特征以token的方式插入，序列特征分层建模了

- **Global Token**：<span style="color: blue;">促进全局特征提取和锚定</span>，可包括目标item表征token、可学习CLS token、用户id embedding、用户-item高阶交互特征。(a) global token从整个序列中聚合上下文信号，增强用户历史、上下文属性、候选item间的特征交互。(b)<span style="color: blue;">稳定了长序列中注意力</span>动态，尤其是稀疏注意力下，StreamLLM中说明少量的global token可以缓解"attention sink"效应，更深的注意力层不成比例的关注早期token，而不是覆盖全序列。这些token作为一个**减少注意力坍塌，保持注意力多样性，并支持远程依赖的的anchor**
- **行为Token Merge**：**把相邻行为token分组压缩成更短的序列，用较小的信息损失换取大幅计算提升。**相邻tokens分组，每K个行为token合并成一个新的token——如何生成？**用InnerTrans增强**，每个组内部进行轻量模块建模组内交互

为什么InnerTrans重要，简单merge会丢失信息？如果一个group里有4个相邻行为，[看了篮球鞋，浏览运动毛巾，加购跑鞋]，如果concat/平均，模型不一定意识到行为是*连续运动消费意图&顺序本身是有意义&组内token有交互结构*

token-merge的核心**trade-off：between model efficiency and representational fidelity** 序列变短，计算下降&压缩了信息损失

### token merge压缩序列长度

用户序列L，维度d，直接用transformer的时间复杂度是 $O(L^2d)$，工业界一般L=2000，d=32，注意力复杂度主要被序列长度平方主导。如果截断，很早但是强相关行为丢失，长期偏好丢失，因此不是“截断”而是**“压缩”**。
$$
\text{FLOPs}_{\text{vanilla trans}} = 24Ld^2 + 4L^2d
$$
第一部分对应投影、FFN；第二部分是attention长序列核心痛点。merge后变成L/K，attention开销明显下降，复杂度比值为：
$$
\frac{\text{FLOPs}_{\text{Merge Token}}}{\text{FLOPs}_{\text{vanilla}}}
=
\frac{6dK + L/K}{6d + L}
$$
（对于L=2048，d=32，K=4时，有42.8%的reduction）--相邻token4合1，计算量减一半

**InnerTrans做法**：x1,x2,x3,x4,x5,x6,x7,x8 每组大小4，分成第一组[x1,x2,x3,x4],第二组[x5,x6,x7,x8]，每一组压成一个token，得到M1,M2 eik是第i个group的k个token embedding
$$
M_i = \text{TransformerBlock}([e_i^1, \dots, e_i^K])
$$
最后变成$[M_1,M_2,...,M_{L/K}]$，相比于普通attention看的长度2048，这个每次看4个token

InnerTrans在局部小窗口处理细节-->送给主干长序列模型

### 混合注意力压缩后的长序列

> we use a hybrid attention mechanism that combines both cross-attention and self-attention layers.    先进行Cross-Causal Attention，再self-casual attention

1. <span style="color: blue;">捕获用户行为序列中时间动态</span>，用额外的positional side info，结合两种位置编码（a）<span style="color: blue;">绝对时间差</span>，交互到目标item的**时间距离用作side info**和item embedding拼接； （b）learnable absolute positional embedding[<span style="color: blue;">绝对位置编码</span>]，编码序列中每个token的位置，加到item embedding

2. 位置编码后，token都过一个MLP去生成输入表征
   $$
   R \in \mathbb{R}^{(m+L)\times d} = [G \in \mathbb{R}^{m\times d}; H \in \mathbb{R}^{L\times d}]
   $$
   其中G是global token，m是token个数；H是sequence token，L是行为序列长度。可以理解为[UID,target,CLS,h1,h2,...,hL]

   采样：Hs从H中采样一小部分token（是完整历史H的压缩版query子集）。最终的query matrix O：
   $$
   O = [G;H_s]
   $$

3. **Cross-Causal Attention：**

   > **Cross-causal attention 的计算本质，是用压缩后的 query 集** O=[G;H_s] **去检索完整输入** R=[G;H]**，在 causal mask 约束下，把完整长序列的信息聚合到较短的 token 集合中，得到长度为** m+k **的压缩上下文化表示，为后续多层 self-attention 提供输入。**

   query输入不是R，而是上一步的O；key和value是全量的R。
   $$
   Q=OW_Q,\quad K=RW_K,\quad V=RW_V
   $$

   $$
   \text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt d}+M\right)V
   $$

   把mask M加到attention score里：
   $$
   M_{i,j}=
   \begin{cases}
   0,& j\ge i\\
   -\infty,& \text{otherwise}
   \end{cases}
   $$
   表明attention单向可见，之后“computing the attention, the result is passed through a feed-forward network (FFN)”本质上也是一个标准transformer block风格

   用少量query token去访问全量历史信息，优点在于复杂度低，从(m+L)方到(m+k)(m+L)。相当于不压缩检索的知识库，只压缩发起检索的query数量。**最终目的是LONGER把近期行为和全局信息保留在query里，是第一层压缩机制**

4. **Self- Casual Attention *N**：得到第一层压缩后的长度是m+k，**这些压缩后的token彼此做多层self-attention**，本质是<span style="color: blue;">在压缩后的token集合上，进一步学习token之间更复杂高阶的表示</span>

![image-20260417165620620](/Users/yue/Library/Application Support/typora-user-images/image-20260417165620620.png)

sample 40% sequence就能保留95%收益，在query端采样40%

**系统工程落地：**

> **dense 参数和 sparse 参数都在 GPU 机器上同步更新，不依赖外部 Parameter Server。**

**![截屏2026-04-17 17.43.43](/Users/yue/Library/Application Support/typora-user-images/截屏2026-04-17 17.43.43.png)**

- 训练框架：与以往传统rec中不同的，之前把sparse放到单独参数服务器，worker远程拉取更新。缺点：通信延迟、系统复杂度高等。LONGER使用统一参数存储+同步更新+尽量colocate到GPU机器。优点：提高吞吐、提升训练稳定性。

  层次化存储：高频特征放GPU HBM，中频特征放CPU内存，低频特征放本地SSD

  <span style="color: blue;">解决了：大规模sparse+dense推荐系统怎么组织训练系统</span>

- Mixed Precision + Recompute

  <span style="color: blue;">解决了：长序列怎么在有效显存下训的动</span>

- KV cache serving：线上推理怎么做

  传统方法对于每一个candidate，做full attention对于candidate和user sequence，整套序列相关计算来一遍，太贵

  serving step1:把用户历史部分算好存下来--user sequence的KV

  serving step2:每个candidate拿自己的query/global token去和缓存的user sequence KV做attention

  -- 此优化把 throughput degradation 从最高 **-40%** 降到 **-6.8%**

  <span style="color: blue;">解决了：同一用户、多candidate打分时怎么避免重复算</span>

  ![截屏2026-04-17 18.09.32](/Users/yue/Library/Application Support/typora-user-images/截屏2026-04-17 18.09.32.png)



**思考的一些Q&A：**

为什么不只用global tokens做query？可以做压缩，但是太全局化，会忽略部分局部行为片段

采样Hs的几种策略：（a）recent k：直接最近k个历史token，论文中实验效果最好（b）整条历史均匀采样（c）k learnable tokens：类似Perceiver/Q-Former做法，不用真实历史token做query，学一组latent query tokens。压缩更强，rec场景不如recent k贴近真实局部行为

第一层压缩后输出的token语义是什么？query token在读取完整长序列后得到的上下文化表示 [后续层不会直接访问完整长序列memory]

## 实验

任务：抖音广告CVR

数据：24.10.16-25.2.23，130天，52亿样本；前123天训练，7天离线评估

baseline：

![截屏2026-04-17 18.55.48](/Users/yue/Library/Application Support/typora-user-images/截屏2026-04-17 18.55.48.png)

训练资源：48*A100

**消融实验** 做了3个方面:

- **Token Merge和InnerTrans的影响:** 增加TokenMerge后FLOPs显著下降, 同时AUC有明显提升, 而增加InnerTrans后, FLOPs有一些上涨, 同时离线AUC指标进一步提升
- **Query Number的影响:** 从效果上是多多益善, 但对应的开销也会更大, 作者这里做了折衷, 取Query Number=100
- **不同Query样本选择策略:** 发现直接取最近的100个Item效果最佳

**scaling analysis**：LONGER 随着序列长度、参数量、FLOPs 增加，效果怎么变化？ 
$$
y=\alpha x^\beta+\gamma
$$
y是AUC/logloss，x是序列长度/参数/FLOPs，通过三个维度上都存在的scaling行为，证明其是一个可扩展架构

## 总结

**可能的优化点**

- query侧：目前LONGER第一层核心是O=[G;Hs]对于Hs的选择论文发现recent k最好。这是一个比较粗糙的query选择策略，可以尝试query从三个层次的组合：recent token+和target最相关的token+长期兴趣代表token



