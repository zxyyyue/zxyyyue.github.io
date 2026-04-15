---
title: "HSTU: High-Performance Sequence Modeling (Generative Perspective)"
date: 2026-04-15
categories: [notes]
tags: [Generative Recommendation, Paper Reading]
permalink: /notes/hstu-reading/
---

## 1. Problem
长序列建模在推荐系统中面临计算复杂度和显存占用的双重挑战，HSTU 尝试通过优化 Attention 算子解决这些问题。

## 2. Core Idea
HSTU 针对推荐系统的长序列特征进行了轻量化设计，摒弃了标准 Transformer 中冗余的模块。从生成式视角看，它为大规模行为序列的预测提供了高效的底层支撑。

## 3. Why It Works
使用了更贴合序列依赖关系的注意力机制，大幅降低了推理延迟。

## 4. My Thoughts
高效的序列建模是生成式推荐的基础。在处理超长历史时，HSTU 的性能优势非常明显。