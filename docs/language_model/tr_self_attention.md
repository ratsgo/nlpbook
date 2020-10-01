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

그러면 트랜스포머 모델 핵심인 셀프 어텐션 기법이 내부에서 어떻게 동작하는지 살펴보겠습니다. 셀프 어텐션은 트랜스포머의 인코더와 디코더 블록 모두에서 수행되는데요. 이 글에서는 인코더의 셀프 어텐션을 살펴보도록 합니다. 그림4를 보면 인코더에서 수행되는 셀프 어텐션의 입력은 이전 인코더 블록의 출력 벡터 시퀀스입니다. 

그림3의 단어 임베딩 차원수($d$)가 4이고, 인코더에 입력된 단어 갯수가 3일 경우 셀프 어텐션 입력은 수식1의 $\mathbf{X}$과 같은 형태가 됩니다. 4차원짜리 단어 임베딩이 3개 모여있음을 확인할 수 있습니다. 수식1의 $\mathbf{X}$의 요소값이 모두 정수(integer)인데요. 이는 예시일 뿐, 실제 계산에서는 거의 대부분이 실수(real number)입니다.

## **수식1** 입력 벡터 시퀀스 X
{: .no_toc .text-delta }

$
\mathbf{X}=\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}
$

셀프 어텐션은 쿼리(query), 키(key), 밸류(value) 세 개 요소 사이의 다이내믹스가 핵심입니다. 입력 벡터 시퀀스($\mathbf{X}$)에 쿼리, 키, 밸류를 만들어주는 행렬($\mathbf{W}$)을 각각 곱해서 만들어 줍니다. 수식2와 같습니다. 수식1처럼 입력 벡터 시퀀스가 3개라면 수식2 적용 후 쿼리, 키, 밸류는 각각 3개씩 총 12개의 벡터가 나옵니다.

수식2에서 $\times$ 기호는 행렬 곱셈(matrix multiplication)을 가리키는 연산자인데요. 해당 기호를 생략하는 경우도 있습니다. 행렬 곱셈이 익숙하지 않은 분들은 [이 글](https://ko.wikipedia.org/wiki/%ED%96%89%EB%A0%AC_%EA%B3%B1%EC%85%88)을 참고하시면 좋겠습니다. 

## **수식2** 쿼리, 키, 밸류 만들기
{: .no_toc .text-delta }

$$
\mathbf{Q}=\mathbf{X} \times { \mathbf{W} }_{ \text{Q} } \\
\mathbf{K}=\mathbf{X} \times { \mathbf{W} }_{ \text{K} } \\
\mathbf{V}=\mathbf{X} \times { \mathbf{W} }_{ \text{V} } \\
$$

수식3은 수식1의 입력 벡터 시퀀스 가운데 첫번째 입력 벡터($\mathbf{X}_1$)로 쿼리를 만들어보는 예시입니다. 수식3 좌변의 첫번째가 바로 $\mathbf{X}_1$입니다. 

그리고 좌변 두번째가 수식2의 ${\mathbf{W}}_{\text{Q}}$에 대응하며 이 행렬은 학습 과정에서 태스크(예컨대 기계 번역)를 가장 잘 수행하는 방향으로 업데이트됩니다.

## **수식3** '쿼리' 만들기 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 1 & 0 & 2 \end{bmatrix}
$

수식1의 입력 벡터 시퀀스 가운데 두번째 입력 벡터($\mathbf{X}_2$)로 쿼리를 만드는 식은 수식4, 세번째($\mathbf{X}_3$)로 쿼리를 만드는 과정은 수식5와 같습니다. 쿼리 만드는 방식은 수식3과 동일합니다.

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

수식6은 입력 벡터 시퀀스 $\mathbf{X}$를 통째로 한꺼번에 쿼리 벡터 시퀀스로 변환하는 식입니다. 입력 벡터 시퀀스에서 하나씩 떼어서 쿼리로 바꾸는 수식3, 수식4, 수식5와 비교했을 때 그 결과가 완전히 동일함을 확인할 수 있습니다. 실제 쿼리 벡터 구축은 수식6과 같은 방식으로 이뤄집니다.

## **수식6** '쿼리' 만들기 (4)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 1 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix} 1 & 0 & 2 \\\\ 2 & 2 & 2 \\\\ 2 & 1 & 3 \end{bmatrix}
$

수식7은 입력 벡터 시퀀스 $\mathbf{X}$를 통째로 한꺼번에 키 벡터 시퀀스로 변환하는 걸 나타내고 있습니다. 수식7 좌변에서 입력 벡터 시퀀스에 곱해지는 행렬은 수식2의 ${\mathbf{W}}_{\text{Q}}$에 대응합니다. 이 행렬은 태스크(예컨대 기계번역)를 가장 잘 수행하는 방향으로 학습 과정 중 업데이트됩니다.

## **수식7** '키' 만들기
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 0 & 0 & 1 \\\\ 1 & 1 & 0 \\\\ 0 & 1 & 0 \\\\ 1 & 1 & 0 \end{bmatrix}=\begin{bmatrix} 0 & 1 & 1 \\\\ 4 & 4 & 0 \\\\ 2 & 3 & 1 \end{bmatrix}
$

수식8은 입력 벡터 시퀀스 $\mathbf{X}$를 통째로 한꺼번에 밸류 벡터 시퀀스로 변환하는 걸 나타내고 있습니다. 수식8 좌변에서 입력 벡터 시퀀스에 곱해지는 행렬은 수식2의 ${\mathbf{W}}_{\text{V}}$에 대응합니다. 이 행렬은 태스크(예컨대 기계번역)를 가장 잘 수행하는 방향으로 학습 과정 중 업데이트됩니다.

## **수식8** '밸류' 만들기
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 2 & 0 & 2 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}\times \begin{bmatrix} 0 & 2 & 0 \\\\ 0 & 3 & 0 \\\\ 1 & 0 & 3 \\\\ 1 & 1 & 0 \end{bmatrix}=\begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 8 & 0 \\\\ 2 & 6 & 3 \end{bmatrix}
$

셀프 어텐션을 계산하기 위한 준비가 모두 끝났습니다! 수식9는 셀프 어텐션의 정의입니다. 쿼리와 키를 행렬곱한 뒤 해당 행렬의 모든 요소값을 키 차원수의 제곱근 값으로 나눠주고, 이 행렬을 행(row) 단위로 소프트맥스(softmax)를 취해 스코어 행렬을 만들어줍니다. 이 스코어 행렬에 밸류를 행렬곱해 줘서 셀프 어텐션 계산을 마칩니다.

## **수식9** 셀프 어텐션
{: .no_toc .text-delta }

$$
\text{Attention} (\mathbf{Q},\mathbf{K},\mathbf{V})= \text{softmax} (\frac { \mathbf{Q} { \mathbf{K} }^{ \top } }{ \sqrt { { d }_{ \text{K} } }  } ) \mathbf{V}
$$

이해를 돕기 위해 수식6의 쿼리 벡터 세 개 가운데 첫번째 쿼리만 가지고 수식9에 정의된 셀프 어텐션 계산을 수행해보겠습니다(수식10\~수식12). 수식10은 첫번째 쿼리 벡터와 모든 키 벡터들에 전치(transpose)를 취한 행렬($\mathbf{K}^{\top}$)을 행렬곱한 결과입니다. 여기에서 전치란 원래 행렬의 행(row)과 열(column)을 교환해 주는 걸 뜻합니다. 

수식10 우변에 있는 벡터의 첫번째 요소값은 첫번째 쿼리 벡터와 첫번째 키 벡터 사이의 다이내믹스가 녹아든 결과입니다. 두번째 요소값은 첫번째 쿼리 벡터와 두번째 쿼리 벡터 사이의 다이내믹스, 세번째 요소값은 첫번째 쿼리와 세번째 쿼리 벡터 사이의 다이내믹스가 포함돼 있습니다.

## **수식10** 첫번째 쿼리 벡터에 관한 셀프 어텐션 계산 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 1 & 0 & 2 \end{bmatrix}\times \begin{bmatrix} 0 & 4 & 2 \\\\ 1 & 4 & 3 \\\\ 1 & 0 & 1 \end{bmatrix}=\begin{bmatrix} 2 & 4 & 4 \end{bmatrix}
$

수식11은 수식10의 결과에 키 벡터의 차원수($d_{\text{K}}=3$)의 제곱근으로 나눠준 뒤 소프트맥스를 취해 만든 확률 벡터입니다. 수식11에서 확인할 수 있듯 소프트맥스는 계산 대상 벡터의 모든 요소값을 0\~1 사이로 바꾸는 한편 모든 요소값의 합을 1로 만들어 확률 속성을 만족하도록 합니다. 다만 우리는 정확한 계산보다는 셀프 어텐션 이해에 초점을 맞추고 있으므로 수식11의 결과가 $[0.0, 0.5, 0.5]$가 되었다고 가정하고 이후 계산을 진행하겠습니다.

## **수식11** 첫번째 쿼리 벡터에 관한 셀프 어텐션 계산 (2)
{: .no_toc .text-delta }

$$
\text{softmax} ([ \frac{2}{\sqrt{3}}, \frac{4}{\sqrt{3}}, \frac{4}{\sqrt{3}} ]) = [ 0.1361, 0.4319, 0.4319 ]
$$

첫번째 쿼리 벡터에 대한 셀프 어텐션 계산의 마지막은 수식12와 같습니다. 수식11의 확률 벡터와 수식8의 밸류 벡터들을 행렬곱해서 계산을 수행합니다. 이는 확률 벡터의 각 요소값에 대응하는 밸류 벡터들을 가중합(weighted sum)한 결과와 동치입니다. 다시 말해 수식12는 $0.0 * [1, 2, 3] + 0.5 * [2, 8, 0] + 0.5 * [2, 6, 3]$와 동일한 결과라는 이야기입니다.

## **수식12** 첫번째 쿼리 벡터에 관한 셀프 어텐션 계산 (3)
{: .no_toc .text-delta }

$
\begin{bmatrix} 0.0 & 0.5 & 0.5 \end{bmatrix}\times \begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 8 & 0 \\\\ 2 & 6 & 3 \end{bmatrix}=\begin{bmatrix} 2.0 & 7.0 & 1.5 \end{bmatrix}
$

그림6은 첫번째 쿼리 벡터에 관한 셀프 어텐션 계산(수식6\~수식12)을 종합적으로 시각화해서 나타낸 그림입니다. 우선 입력 벡터 시퀀스를 각각 쿼리, 키, 밸류로 변환합니다. 첫번째 쿼리 벡터(그림6에서 붉은색 벡터)에 대해 소프트맥스 확률(score)을 만들고 이 확률과 밸류 벡터들을 가중합해서 셀프 어텐션 출력 벡터(output)를 계산합니다.

## **그림6** 첫번째 쿼리 벡터에 관한 셀프 어텐션(self attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Fvr3gPU.png" width="800px" title="source: imgur.com" />

이번에는 수식6의 두번째 쿼리 벡터에 대해 셀프 어텐션 계산을 해보겠습니다. 수식13은 두번째 쿼리 벡터와 모든 키 벡터들에 전치(transpose)를 취한 행렬($\mathbf{K}^{\top}$)을 행렬곱한 결과입니다. 여기에서 전치란 원래 행렬의 행(row)과 열(column)을 교환해 주는 걸 뜻합니다. 

수식13 우변에 있는 벡터의 첫번째 요소값은 두번째 쿼리 벡터와 첫번째 키 벡터 사이의 다이내믹스가 녹아든 결과입니다. 두번째 요소값은 두번째 쿼리 벡터와 두번째 쿼리 벡터 사이의 다이내믹스, 세번째 요소값은 두번째 쿼리와 세번째 쿼리 벡터 사이의 다이내믹스가 포함돼 있습니다.

## **수식13** 두번째 쿼리 벡터에 관한 셀프 어텐션 계산 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 2 & 2 & 2 \end{bmatrix}\times \begin{bmatrix} 0 & 4 & 2 \\\\ 1 & 4 & 3 \\\\ 1 & 0 & 1 \end{bmatrix}=\begin{bmatrix} 4 & 16 & 12 \end{bmatrix}
$

수식14는 수식13의 결과에 키 벡터의 차원수($d_{\text{K}}=3$)의 제곱근으로 나눠준 뒤 소프트맥스를 취해 만든 확률 벡터입니다. 우리는 정확한 계산보다는 셀프 어텐션 이해에 초점을 맞추고 있으므로 수식11의 결과가 $[0.0, 1.0, 0.0]$가 되었다고 가정하고 이후 계산을 진행하겠습니다.

## **수식14** 두번째 쿼리 벡터에 관한 셀프 어텐션 계산 (2)
{: .no_toc .text-delta }

$$
\text{softmax} ([ \frac{4}{\sqrt{3}}, \frac{16}{\sqrt{3}}, \frac{12}{\sqrt{3}} ])= [ 0.0009, 0.9088, 0.0903 ]
$$

두번째 쿼리 벡터에 대한 셀프 어텐션 계산의 마지막은 수식15와 같습니다. 수식14의 확률 벡터와 수식8의 밸류 벡터들을 행렬곱해서 계산을 수행합니다. 이는 확률 벡터의 각 요소값에 대응하는 밸류 벡터들을 가중합(weighted sum)한 결과와 동치입니다.

## **수식15** 두번째 쿼리 벡터에 관한 셀프 어텐션 계산 (3)
{: .no_toc .text-delta }

$
\begin{bmatrix} 0.0 & 1.0 & 0.0 \end{bmatrix}\times \begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 8 & 0 \\\\ 2 & 6 & 3 \end{bmatrix}=\begin{bmatrix} 2.0 & 8.0 & 0.0 \end{bmatrix}
$

그림7은 두번째 쿼리 벡터에 관한 셀프 어텐션 계산(수식13\~수식15)을 종합적으로 시각화해서 나타낸 그림입니다. 우선 입력 벡터 시퀀스를 각각 쿼리, 키, 밸류로 변환합니다. 두번째 쿼리 벡터(그림7에서 붉은색 벡터)에 대해 소프트맥스 확률(score)을 만들고 이 확률과 밸류 벡터들을 가중합해서 셀프 어텐션 출력 벡터(output)를 계산합니다.

## **그림7** 두번째 쿼리 벡터에 관한 셀프 어텐션(self attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZUtND3X.png" width="800px" title="source: imgur.com" />

수식6의 마지막 세번째 쿼리 벡터에 대해 셀프 어텐션 계산을 해보겠습니다. 수식16은 세번째 쿼리 벡터와 모든 키 벡터들에 전치(transpose)를 취한 행렬($\mathbf{K}^{\top}$)을 행렬곱한 결과입니다. 여기에서 전치란 원래 행렬의 행(row)과 열(column)을 교환해 주는 걸 뜻합니다. 

수식16 우변에 있는 벡터의 첫번째 요소값은 세번째 쿼리 벡터와 첫번째 키 벡터 사이의 다이내믹스가 녹아든 결과입니다. 두번째 요소값은 세번째 쿼리 벡터와 두번째 쿼리 벡터 사이의 다이내믹스, 세번째 요소값은 세번째 쿼리와 세번째 쿼리 벡터 사이의 다이내믹스가 포함돼 있습니다.

## **수식16** 세번째 쿼리 벡터에 관한 셀프 어텐션 계산 (1)
{: .no_toc .text-delta }

$
\begin{bmatrix} 2 & 1 & 3 \end{bmatrix}\times \begin{bmatrix} 0 & 4 & 2 \\\\ 1 & 4 & 3 \\\\ 1 & 0 & 1 \end{bmatrix}=\begin{bmatrix} 4 & 12 & 10 \end{bmatrix}
$

수식17은 수식16의 결과에 키 벡터의 차원수($d_{\text{K}}=3$)의 제곱근으로 나눠준 뒤 소프트맥스를 취해 만든 확률 벡터입니다. 우리는 정확한 계산보다는 셀프 어텐션 이해에 초점을 맞추고 있으므로 수식16의 결과가 $[0.0, 0.9, 0.1]$가 되었다고 가정하고 이후 계산을 진행하겠습니다.

## **수식17** 세번째 쿼리 벡터에 관한 셀프 어텐션 계산 (2)
{: .no_toc .text-delta }

$$
\text{softmax} ([ \frac{4}{\sqrt{3}}, \frac{12}{\sqrt{3}}, \frac{10}{\sqrt{3}} ])= [ 0.0074, 0.7547, 0.2378 ]
$$

세번째 쿼리 벡터에 대한 셀프 어텐션 계산의 마지막은 수식18과 같습니다. 수식17의 확률 벡터와 수식8의 밸류 벡터들을 행렬곱해서 계산을 수행합니다. 이는 확률 벡터의 각 요소값에 대응하는 밸류 벡터들을 가중합(weighted sum)한 결과와 동치입니다.

## **수식18** 세번째 쿼리 벡터에 관한 셀프 어텐션 계산 (3)
{: .no_toc .text-delta }

$
\begin{bmatrix} 0.0 & 0.9 & 0.1 \end{bmatrix}\times \begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 8 & 0 \\\\ 2 & 6 & 3 \end{bmatrix}=\begin{bmatrix} 2.0 & 7.8 & 0.3 \end{bmatrix}
$

그림8은 세번째 쿼리 벡터에 관한 셀프 어텐션 계산(수식13\~수식15)을 종합적으로 시각화해서 나타낸 그림입니다. 우선 입력 벡터 시퀀스를 각각 쿼리, 키, 밸류로 변환합니다. 세번째 쿼리 벡터(그림7에서 붉은색 벡터)에 대해 소프트맥스 확률(score)을 만들고 이 확률과 밸류 벡터들을 가중합해서 셀프 어텐션 출력 벡터(output)를 계산합니다.

## **그림8** 세번째 쿼리 벡터에 관한 셀프 어텐션(self attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SODWeSi.png" width="800px" title="source: imgur.com" />

지금까지는 손 계산으로 셀프 어텐션을 살펴봤는데요. 파이토치를 활용해 코드로도 확인해 보겠습니다. 우선 입력 벡터 시퀀스 $\mathbf{X}$와 쿼리, 키, 밸류 구축에 필요한 행렬들을 앞선 예시 그대로 정의합니다. 코드1과 같습니다.

## **코드1** 변수 정의
{: .no_toc .text-delta }
```python
import torch
x = torch.tensor([
  [1.0, 0.0, 1.0, 0.0],
  [0.0, 2.0, 0.0, 2.0],
  [1.0, 1.0, 1.0, 1.0],  
])
w_key = torch.tensor([
  [0.0, 0.0, 1.0],
  [1.0, 1.0, 0.0],
  [0.0, 1.0, 0.0],
  [1.0, 1.0, 0.0]
])
w_query = torch.tensor([
  [1.0, 0.0, 1.0],
  [1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 1.0, 1.0]
])
w_value = torch.tensor([
  [0.0, 2.0, 0.0],
  [0.0, 3.0, 0.0],
  [1.0, 0.0, 3.0],
  [1.0, 1.0, 0.0]
])
```

코드2는 수식2를 계산해 입력 벡터 시퀀스로 쿼리, 키, 밸류 벡터들을 만드는 파트입니다. `@`는 행렬곱을 뜻하는 연산자입니다.

## **코드2** 쿼리, 키, 밸류 만들기
{: .no_toc .text-delta }
```python
keys = x @ w_key
querys = x @ w_query
values = x @ w_value
```

코드3은 코드2에서 만든 쿼리와 키 벡터들을 행렬곱해서 어텐션 스코어를 만드는 과정입니다.

## **코드3** 어텐션 스코어 만들기
{: .no_toc .text-delta }
```python
attn_scores = querys @ keys.T
```

코드3을 수행한 결과는 다음과 같은데요. 정확히 수식10, 수식13, 수식16과 같습니다. 이들은 쿼리 벡터를 하나씩 떼어서 계산을 수행한 것인데요. 코드3처럼 쿼리 벡터들을 한꺼번에 모아서 키 벡터들과 행렬곱을 수행하여도 같은 결과를 낼 수 있음을 확인할 수 있습니다.

```
>>> attn_scores
tensor([[ 2.,  4.,  4.],
        [ 4., 16., 12.],
        [ 4., 12., 10.]])
```

코드4는 코드3의 결과에 키 벡터의 차원수의 제곱근으로 나눠준 뒤 소프트맥스를 취하는 과정을 나타내고 있습니다.

## **코드4** 소프트맥스 확률값 만들기
{: .no_toc .text-delta }
```python
import numpy as np
from torch.nn.functional import softmax
key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_scores_softmax = softmax(attn_scores / key_dim_sqrt, dim=-1)
```

코드4를 수행한 결과는 다음과 같습니다. 정확히 수식11, 수식14, 수식17과 같습니다. 이 역시 쿼리 벡터를 한꺼번에 모아서 수행해도 된다는 이야기입니다.

```
>>> attn_scores_softmax
tensor([[1.3613e-01, 4.3194e-01, 4.3194e-01],
        [8.9045e-04, 9.0884e-01, 9.0267e-02],
        [7.4449e-03, 7.5471e-01, 2.3785e-01]])
```

코드5는 코드4에서 구한 소프트맥스 확률과 밸류 벡터들을 가중합하는 과정을 수행하는 코드입니다. 다만 우리는 앞선 예시에서 설명의 편의를 위해 소프트맥스 확률값들을 간단한 다른 값으로 계산을 수행했으므로 코드4에서 구한 결과는 무시하고 새로운 값을 `attn_scores_softmax`에 넣어줍니다.

## **코드5** 소프트맥스 확률과 밸류를 가중합하기
{: .no_toc .text-delta }

```python
attn_scores_softmax = torch.tensor([
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
])
weighted_values = attn_scores_softmax @ values
```

코드5의 수행 결과는 다음과 같습니다. 정확히 수식12, 수식15, 수식18과 같습니다.

```
>>> weighted_values
tensor([[2.0000, 7.0000, 1.5000],
        [2.0000, 8.0000, 0.0000],
        [2.0000, 7.8000, 0.3000]])
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
