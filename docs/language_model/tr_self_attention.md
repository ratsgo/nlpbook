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

셀프 어텐션을 이해하려면 먼저 입력부터 살펴봐야 합니다. 그림1은 트랜스포머 모델의 전체 구조를, 그림2는 그림1에서 인코더 입력만을 떼어서 나타낸 그림입니다. 그림2와 같이 모델 입력을 만드는 역할을 계층(layer)을 입력층(input layer)이라고 부릅니다. 

## **그림1** Transformer 전체 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Rk5wkBQ.png" width="600px" title="source: imgur.com" />

## **그림2** 인코더 입력
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YDMYiLP.png" width="200px" title="source: imgur.com" />

그림2에서 확인할 수 있듯 인코더 입력은 입력 임베딩(input embedding)에 위치 정보(positional encoding)을 더해서 만듭니다. 

한국어에서 영어로 기계 번역을 수행하는 트랜스포머 모델을 구축하다고 가정해 봅시다. 이 경우 인코더 입력은 소스 언어 문장의 토큰 인덱스(index) 시퀀스가 됩니다. 앞서 우리는 전처리 과정에서 입력 문장을 토큰화한 뒤 이를 인덱스로 변환한 적이 있는데요. 토큰화 및 인덱싱과 관련해서는 [3-4장 Tokenization Tutorial](https://ratsgo.github.io/nlpbook/docs/tokenization/encode)을 참고하면 좋을 것 같습니다.

어쨌든 소스 언어의 토큰 시퀀스가 `어제`, `카페`, `갔었어`라면 인코더 입력층의 직접적인 입력값은 이들 토큰들에 대응하는 인덱스 시퀀스가 되며 인코더 입력은 그림3과 같은 방식으로 만들어집니다. 

## **그림3** 인코더 입력 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/a4AXjcw.jpg" width="800px" title="source: imgur.com" />

그림3의 왼편 행렬(matrix)은 소스 언어의 각 어휘에 대응하는 단어 수준 임베딩인데요. 단어 수준 임베딩 행렬에서 현재 입력의 각 토큰 인덱스에 대응하는 벡터를 참조(lookup)한 것이 그림2의 입력 임베딩(input embedding)입니다. 단어 수준 임베딩은 트랜스포머의 다른 요소들처럼 소스 언어를 타겟 언어로 번역하는 태스크를 수행하는 과정에서 같이 업데이트(학습)됩니다.

입력 임베딩에 더해지는, 위치 정보는 해당 토큰이 문장 내에서 몇 번째 위치인지 정보를 나타냅니다. 그림3 예시에서는 `어제`가 첫번째, `카페`가 두번째, `갔었어`가 세번째 토큰인데요. 위치 정보는 단어 수준 임베딩 같이 번역 태스크를 수행하는 과정에서 업데이트되게끔 만들 수도 있고 학습 내내 고정된 값을 유지하는 경우도 있습니다. 

트랜스포머 모델은 이같은 방식으로 소스 언어의 토큰 시퀀스를 이에 대응하는 벡터 시퀀스로 변환해 인코더 입력을 만듭니다. 디코더 입력 역시 만드는 방식이 동일합니다. 

그림4는 그림1에서 인코더와 디코더 블록만을 떼어 그린 그림인데요. 인코더 입력층(그림2)에서 만들어진 벡터 시퀀스가 최초 인코더 블록의 입력이 되며, 최초 인코더 블록의 출력 벡터 시퀀스가 두번째 인코더 블록의 입력이 됩니다. 마찬가지로 디코더 입력층에서 만들어진 벡터 시퀀스가 최초 디코더 블록의 입력이 되며, 최초 디코더 블록의 출력 벡터 시퀀스가 두번째 디코더 블록의 입력이 됩니다.

## **그림4** 인코더-디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qaMh3TR.png" width="400px" title="source: imgur.com" />

그림5는 그림1에서 모델의 최종 출력층(output layer)만을 떼어 그린 그림입니다. 이 출력층의 입력은 디코더 마지막 블록의 출력 벡터 시퀀스입니다. 출력층의 출력은 타겟 언어의 어휘 수만큼의 차원을 갖는 확률 벡터가 됩니다. 소스 언어의 어휘가 총 3만개라고 가정해봅시다. 그렇다면 디코더 출력은 3만 차원의 확률 벡터입니다. 이 벡터의 요소 값 3만개는 모두 확률이므로 3만개 요소값을 다 더하면 그 합은 1이 됩니다.

## **그림5** 디코더 출력
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LOr9QS6.png" width="200px" title="source: imgur.com" />

트랜스포머의 학습(train)은 인코더와 디코더 입력이 주어졌을 때 모델 최종 출력에서 정답에 해당하는 단어의 확률 값을 높이는 방식으로 수행됩니다.

---

## 셀프 어텐션 내부 동작

## **수식1** 쿼리, 키, 밸류 만들기
{: .no_toc .text-delta }

$$
\mathbf{Q}=\mathbf{X} \cdot { \mathbf{W} }_{ \text{Q} } \\
\mathbf{K}=\mathbf{X} \cdot { \mathbf{W} }_{ \text{K} } \\
\mathbf{V}=\mathbf{X} \cdot { \mathbf{W} }_{ \text{V} } \\
$$

## **수식2** 입력 벡터 시퀀스 X
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}
$

## **수식3** '쿼리' 만들기 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 1 & 0 & 2 \end{bmatrix}
$

## **수식4** '쿼리' 만들기 (2)
{: .no_toc .text-delta }

$
\begin{bmatrix} 0 & 2 & 0 & 2 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 2 & 2 & 2 \end{bmatrix}
$

## **수식5** '쿼리' 만들기 (3)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 2 & 1 & 3 \end{bmatrix}
$

## **수식6** '쿼리' 만들기 (4)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 1 & 0 & 2 \\\\ 2 & 2 & 2 \\\\ 2 & 1 & 3 \end{bmatrix}
$

## **수식7** '키' 만들기
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 0 & 0 & 1 \\\\ 1 & 1 & 0 \\\\ 0 & 1 & 0 \\\\ 1 & 1 & 0 \end{bmatrix}=\begin{bmatrix} 0 & 1 & 1 \\\\ 4 & 4 & 0 \\\\ 2 & 3 & 1 \end{bmatrix}
$

## **수식8** '밸류' 만들기
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 0 & 2 & 0 \\\\ 0 & 3 & 0 \\\\ 1 & 0 & 3 \\\\ 1 & 1 & 0 \end{bmatrix}=\begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 8 & 0 \\\\ 2 & 6 & 3 \end{bmatrix}
$

## **수식9** 셀프 어텐션
{: .no_toc .text-delta }

$$
\text{Attention} (\mathbf{Q},\mathbf{K},\mathbf{V})= \text{softmax} (\frac { \mathbf{Q} { \mathbf{K} }^{ \top } }{ \sqrt { { d }_{ \text{K} } }  } ) \mathbf{V}
$$

## **수식10** 셀프 어텐션 계산 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 2 \end{bmatrix}\times \begin{bmatrix} 0 & 4 & 2 \\\\ 1 & 4 & 3 \\\\ 1 & 0 & 1 \end{bmatrix}=\begin{bmatrix} 2 & 4 & 4 \end{bmatrix}
$

## **수식11** 셀프 어텐션 계산 (2)
{: .no_toc .text-delta }

$$
\text{softmax} ([ 2, 4, 4 ])= [ 0.0, 0.5, 0.5 ]
$$

## **수식12** 셀프 어텐션 계산 (3)
{: .no_toc .text-delta }

$$
0.0 * [1, 2, 3] = [0.0, 0.0, 0.0] \\
0.5 * [2, 8, 0] = [1.0, 4.0, 0.0] \\
0.5 * [2, 6, 3] = [1.0, 3.0, 1.5]
$$

## **그림6** 셀프 어텐션(self attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Fvr3gPU.png" width="800px" title="source: imgur.com" />


## **그림7** 셀프 어텐션(self attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZUtND3X.png" width="800px" title="source: imgur.com" />


## **그림8** 셀프 어텐션(self attention) (3)
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


## 인코더에서 수행하는 셀프 어텐션

## **그림5** 트랜스포머 인코더 블록
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NSmVlit.png" width="150px" title="source: imgur.com" />


## **그림6** 예시 문장의 셀프 어텐션(Self Attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ydO8rUt.jpg" width="200px" title="source: imgur.com" />


## **그림7** 예시 문장의 셀프 어텐션(Self Attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/QgIcjoJ.jpg" width="200px" title="source: imgur.com" />


---

## 디코더에서 수행하는 셀프 어텐션


## **그림4** 인코더-디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qaMh3TR.png" width="400px" title="source: imgur.com" />

## **그림9** 타겟 문장의 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i2GTz6h.jpg" width="200px" title="source: imgur.com" />


## **그림10** 소스-타겟 문장 간 셀프 어텐션(Self Attention) 계산 일부
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tQLi4Wb.jpg" width="200px" title="source: imgur.com" />

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

- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)


---
