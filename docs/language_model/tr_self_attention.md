---
layout: default
title: ↗️ Self Attention
parent: Transformers
grand_parent: Language Model
nav_order: 1
---

# 셀프 어텐션(Self Attention)
{: .no_toc }

트랜스포머(transformer)의 핵심 구성요소는 셀프 어텐션(self attention)입니다. 이 글에서는 셀프 어텐션의 내부 동작 원리에 대해 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

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


## 쿼리, 키, 밸류

셀프 어텐션은 쿼리(query), 키(key), 밸류(value)


## **그림7** 예시 문장의 셀프 어텐션(Self Attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ydO8rUt.jpg" width="200px" title="source: imgur.com" />


## **그림8** 예시 문장의 셀프 어텐션(Self Attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/QgIcjoJ.jpg" width="200px" title="source: imgur.com" />


---

## 인코더에서 수행하는 셀프 어텐션


## **그림4** 인코더-디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qaMh3TR.png" width="400px" title="source: imgur.com" />


## **그림7** 소스 문장의 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ydO8rUt.jpg" width="200px" title="source: imgur.com" />


---

## 디코더에서 수행하는 셀프 어텐션


## **그림9** 타겟 문장의 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i2GTz6h.jpg" width="200px" title="source: imgur.com" />


## **그림10** 소스-타겟 문장 간 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tQLi4Wb.jpg" width="200px" title="source: imgur.com" />


---

## 셀프 어텐션 내부 동작

- 그림 예시를 어제, 카페, 갔었어 로 대체
- 인코더와 디코더 차이 설명 : query, key가 인코더 출력 + attention mask + 튜토리얼
- Multi-Head Attention : 설명 + 튜토리얼


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

---

## 멀티 헤드 어텐션

그림9 트랜스포머 인코더 블록의 멀티-헤드 어텐션은 **셀프 어텐션(self attention)**이라는 기법을 여러 번 수행한 걸 가리킵니다. 하나의 헤드(head)가 셀프 어텐션을 1회 수행하고 이를 여러 개 헤드가 독자적으로 각각 계산한다는 이야기라는 말입니다.

## **그림14** 멀티-헤드 어텐션(Multi-Head Attention)
{: .no_toc .text-delta }
<img src="https://nlpinkorean.github.io/images/transformer/transformer_attention_heads_weight_matrix_o.png" width="800px" title="source: imgur.com" />


---


## 참고 문헌

- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)


---
