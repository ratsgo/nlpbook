---
layout: default
title: Transformers
parent: Language Model
nav_order: 2
---

# 트랜스포머(Transformer)
{: .no_toc }

최근 트랜스포머(transformer) 기반 언어모델이 각광받고 있습니다. 그 성능이 좋기 때문인데요. 왜 성능이 좋은지, 핵심 동작 원리는 무엇인지 이 글에서 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 모델 전체 구조


## **그림1** 전체 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/F0qY4ny.png" width="500px" title="source: imgur.com" />


## **그림2** 1개 레이어
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NSmVlit.png" width="200px" title="source: imgur.com" />


---

## 셀프 어텐션


## **그림3** 리커런트 뉴럴 네트워크(Recurrenct Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5eAs55K.png" width="800px" title="source: imgur.com" />


## **그림4** 컨볼루션 뉴럴 네트워크(Convolutional Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3pn5n0C.png" width="800px" title="source: imgur.com" />
<img src="https://i.imgur.com/tk2s2eR.png" width="800px" title="source: imgur.com" />
<img src="https://i.imgur.com/QZ7QV6v.png" width="800px" title="source: imgur.com" />


## **그림5** 기계 번역에 쓰인 어텐션(Attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kRjEkPm.png" width="800px" title="source: imgur.com" />


## **그림6** 셀프 어텐션(Self Attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YUcm8xS.png" width="800px" title="source: imgur.com" />


---

### 쿼리, 키, 밸류


## **그림7** 예시 문장의 셀프 어텐션(Self Attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ydO8rUt.jpg" width="200px" title="source: imgur.com" />


## **그림8** 예시 문장의 셀프 어텐션(Self Attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/QgIcjoJ.jpg" width="200px" title="source: imgur.com" />


---


### ↗️ 셀프 어텐션 내부 동작


## **그림11** 셀프 어텐션(self attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Fvr3gPU.png" width="800px" title="source: imgur.com" />


## **그림12** 셀프 어텐션(self attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZUtND3X.png" width="800px" title="source: imgur.com" />


## **그림13** 셀프 어텐션(self attention) (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SODWeSi.png" width="800px" title="source: imgur.com" />


## **코드1** 셀프 어텐션(self attention) 튜토리얼 (1)
{: .no_toc .text-delta }
```python
import torch

x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

keys = x @ w_key
querys = x @ w_query
values = x @ w_value

print(keys)
# tensor([[0., 1., 1.],
#         [4., 4., 0.],
#         [2., 3., 1.]])

print(querys)
# tensor([[1., 0., 2.],
#         [2., 2., 2.],
#         [2., 1., 3.]])

print(values)
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])

attn_scores = querys @ keys.T
# tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
#         [ 4., 16., 12.],  # attention scores from Query 2
#         [ 4., 12., 10.]]) # attention scores from Query 3

from torch.nn.functional import softmax

attn_scores_softmax = softmax(attn_scores, dim=-1)
# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)

weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]

# tensor([[[0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000]],
# 
#         [[1.0000, 4.0000, 0.0000],
#          [2.0000, 8.0000, 0.0000],
#          [1.8000, 7.2000, 0.0000]],
# 
#         [[1.0000, 3.0000, 1.5000],
#          [0.0000, 0.0000, 0.0000],
#          [0.2000, 0.6000, 0.3000]]])

outputs = weighted_values.sum(dim=0)

# tensor([[2.0000, 7.0000, 1.5000],  # Output 1
#         [2.0000, 8.0000, 0.0000],  # Output 2
#         [2.0000, 7.8000, 0.3000]]) # Output 3
```

## **그림14** 멀티-헤드 어텐션(Multi-Head Attention)
{: .no_toc .text-delta }
<img src="https://nlpinkorean.github.io/images/transformer/transformer_attention_heads_weight_matrix_o.png" width="800px" title="source: imgur.com" />


---

## ↗️ 트랜스포머에 적용된 기술들


### 피드포워드 뉴럴네트워크

중간에 activation 있음, wx+b 여러번 하면 의미 없음

## **그림15** 피드포워드 뉴럴네트워크(Feedforward Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/8nElvRQ.jpg" width="800px" title="source: imgur.com" />

## **그림16** ReLU
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3acxUWy.png" width="400px" title="source: imgur.com" />


### 잔차 연결

## **그림17** 잔차 연결(Residual Connections)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/UHVuX1X.jpg" width="800px" title="source: imgur.com" />


### 레이어 정규화

## **그림18** 레이어 정규화(Layer Normalization)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NY82BPy.png" width="200px" title="source: imgur.com" />



---

## 인코더와 디코더


## **그림4** 인코더-디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qaMh3TR.png" width="400px" title="source: imgur.com" />


## **그림7** 소스 문장의 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ydO8rUt.jpg" width="200px" title="source: imgur.com" />


## **그림9** 타겟 문장의 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i2GTz6h.jpg" width="200px" title="source: imgur.com" />


## **그림10** 소스-타겟 문장 간 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tQLi4Wb.jpg" width="200px" title="source: imgur.com" />



---

## 모델 입력과 출력


## **그림19** 모델 입력
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YDMYiLP.png" width="200px" title="source: imgur.com" />

## **그림20** 입력 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/a4AXjcw.jpg" width="800px" title="source: imgur.com" />

## **그림21** 모델 출력
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LOr9QS6.png" width="200px" title="source: imgur.com" />

## **그림22** Masked Attention (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6jQImF5.jpg" width="300px" title="source: imgur.com" />

## **그림23** Model Update (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kZ5JM2k.jpg" width="800px" title="source: imgur.com" />

## **그림22** Masked Attention (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/86S3yyI.jpg" width="300px" title="source: imgur.com" />

## **그림23** Model Update (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/h6SN72R.jpg" width="800px" title="source: imgur.com" />

## **그림22** Masked Attention (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/CJKtexJ.jpg" width="300px" title="source: imgur.com" />

## **그림23** Model Update (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Kd1TIRo.jpg" width="800px" title="source: imgur.com" />

---


## 참고 문헌

- []()


---
