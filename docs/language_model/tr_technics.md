---
layout: default
title: ↗️ Technics
parent: Transformers
grand_parent: Language Model
nav_order: 2
---

# 트랜스포머(Transformer)에 적용된 기술들
{: .no_toc }

트랜스포머(transformer)가 좋은 성능을 내는 데는 [Self Attention]() 말고도 다양한 기법들이 적용됐기 때문입니다. 이 글에서는 Self Attention을 제외한 트랜스포머의 주요 요소들을 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---



## 피드포워드 뉴럴네트워크

중간에 activation 있음, wx+b 여러번 하면 의미 없음

## **그림1** 피드포워드 뉴럴네트워크(Feedforward Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/8nElvRQ.jpg" width="800px" title="source: imgur.com" />

## **그림2** ReLU
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3acxUWy.png" width="400px" title="source: imgur.com" />


## 잔차 연결

## **그림3** 잔차 연결(Residual Connections)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/UHVuX1X.jpg" width="800px" title="source: imgur.com" />


## 레이어 정규화

## **그림4** 레이어 정규화(Layer Normalization)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NY82BPy.png" width="200px" title="source: imgur.com" />


## 드롭아웃

과적합을 방지하기 위해서 노드의 일부를 0으로 대치하여 계산에서 제외하는 기법

## **그림5** 드롭아웃(Dropout)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pKUE70B.png" width="300px" title="source: imgur.com" />


## Adam Optimizer


## Warm-up Scheduling


---


## 참고 문헌

- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)


---
