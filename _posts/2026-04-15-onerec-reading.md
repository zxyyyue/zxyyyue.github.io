---
title: "OneRec: Generative Recommendation using LLMs"
date: 2026-04-15
categories: [notes]
tags: [Generative Recommendation, Paper Reading]
permalink: /notes/onerec-reading/
---

## 1. Problem
传统推荐范式难以充分利用深度语义联系，OneRec 尝试将推荐统一转化为生成式任务。

## 2. Core Idea
利用 LLM 的建模能力，直接生成推荐结果或用户兴趣描述。

## 3. Why It Works
打破了传统的“检索+排序”限制，能够更灵活地捕捉用户的潜在意图。

## 4. My Thoughts
目前的挑战在于线上推理的延迟以及如何缓解大模型的幻觉问题。