---
layout: default
title: Technics
parent: Transformers
grand_parent: Language Model
nav_order: 2
---

# ↗️ 트랜스포머에 적용된 기술들
{: .no_toc }

트랜스포머(transformer)가 좋은 성능을 내는 데는 [Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention/) 말고도 다양한 기법들이 적용됐기 때문입니다. 이번 절에서는 셀프 어텐션 외에 트랜스포머의 주요 요소들을 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 트랜스포머 블록

그림1은 트랜스포머 모델에서 인코더와 디코더 블록 부분을 떼어 다시 그린 것입니다. 인코더와 디코더 블록의 구조는 디테일에서 차이가 있을 뿐 본질적으로는 크게 다르지 않습니다. 즉 멀티 헤드 어텐션, 피드포워드 뉴럴 네트워크, 잔차 연결 및 레이어 정규화 등 세 가지 구성 요소를 기본으로 합니다.

## **그림1** 인코더, 디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qaMh3TR.png" width="400px" title="source: imgur.com" />

이 가운데 Multi-Head Attention은 이전 장에서 이미 살펴본 바 있습니다. 이와 관련해서는 [3장 Transformers](https://ratsgo.github.io/nlpbook/docs/language_model/transformers), [3-1장 Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention)을 참고하시면 좋을 것 같습니다. 이 챕터에서는 나머지 구성 요소인 FeedForward, Add&Norm을 차례대로 살펴보겠습니다.

### FeedForward : 피드포워드 뉴럴네트워크

멀티 헤드 어텐션의 출력은 입력 단어들에 대응하는 벡터 시퀀스인데요. 이후 벡터 각각을 피드포워드 뉴럴네트워크에 입력합니다. 다시 말해 피드포워드 뉴럴네트워크의 입력은 현재 블록의 멀티 헤드 어텐션의의 개별 출력 벡터가 됩니다.

피드포워드 뉴럴네트워크란 신경망(neural network)의 한 종류로 그림2와 같이 입력층(input layer, $x$), 은닉층(hidden layer, $h$), 출력층(ouput layer, $y$) 3개 계층으로 구성돼 있습니다. 그림2의 각 동그라미를 뉴런(neuron)이라고 합니다.

## **그림2** 피드포워드 뉴럴네트워크
{: .no_toc .text-delta }
<img src="https://i.imgur.com/MsyHJFC.png" width="400px" title="source: imgur.com" />

그림3은 뉴런과 뉴런 사이의 계산 과정을 좀 더 자세히 그린 것입니다. 이전 뉴런 값(그림3의 $x_i$)과 그에 해당하는 가중치(그림3의 $w_i$)를 가중합(weighted sum)한 결과에 바이어스(bias, 그림3의 $b$)를 더해 만듭니다. 가중치들과 바이어스는 학습 과정에서 업데이트됩니다. 그림3의 활성 함수(activation function, $f$)는 현재 계산하고 있는 뉴런의 출력을 일정 범위로 제한하는 역할을 합니다.

## **그림3** 뉴런
{: .no_toc .text-delta }
<img src="http://i.imgur.com/euw7qQu.png" width="400px" title="source: imgur.com" />

트랜스포머에서 사용하는 피드포워드 뉴럴네트워크의 활성함수는 **ReLU(Rectified Linear Unit)**입니다. 수식1과 같이 정의되며 입력($x$)에 대해 그림4와 같은 그래프 모양을 가집니다. 다시 말해 양수 입력은 그대로 흘려보내되 음수 입력은 모두 0으로 치환해 무시합니다.

## **수식1** ReLU
{: .no_toc .text-delta }

$$
f(x)= \max{(0, x)}
$$

## **그림4** ReLU
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3acxUWy.png" width="350px" title="source: imgur.com" />

이제 피드포워드 뉴럴네트워크의 계산 과정을 살펴보겠습니다. 입력층 뉴런이 각각 $[2,1]$이고 그에 해당하는 가중치가 $[3,2]$, 바이어스(bias)가 1이라고 가정해 보겠습니다. 그러면 은닉층 첫번째 뉴런 값은 $2 \times 3 + 1 \times 2 + 1=9$가 됩니다. 이 값은 양수이므로 ReLU를 통과해도 그대로 살아납니다. 그림5와 같습니다.

## **그림5** 피드포워드 뉴럴네트워크 계산 예시 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9fmlFxA.png" width="400px" title="source: imgur.com" />

그림5와 입력이 동일하고 입력에 대한 가중치가 $[2, -3]$이라면 은닉층 두번째 뉴런 값은 $2 \times 2 + 1 \times -3 + 1=2$가 됩니다. 이 값은 양수이므로 ReLU를 통과해도 그대로 살아남습니다. 그림6과 같습니다. 

## **그림6** 피드포워드 뉴럴네트워크 계산 예시 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LJuZzfu.png" width="400px" title="source: imgur.com" />

그림5와 입력이 동일하고 입력에 대한 가중치가 $[-4, 1]$이라면 은닉층 세번째 뉴런 값은 $2 \times -4 + 1 \times 1 + 1=-6$이 됩니다. 하지만 이 값은 음수이므로 ReLU를 통과하면서 0이 됩니다. 그림7과 같습니다.

## **그림7** 피드포워드 뉴럴네트워크 계산 예시 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/dJ4EdNu.png" width="400px" title="source: imgur.com" />

은닉층 처리를 마치고 이제 출력층을 계산할 차례입니다. 은닉층 뉴런이 각각 $[9,2,0]$이고 그에 대응하는 가중치가 $[-1,1,3]$이라면 출력층 첫번째 뉴런 값은 $9 \times -1 + 2 \times 1 + 0 \times 3 -1=-8$이 됩니다. 그림8과 같습니다.

## **그림8** 피드포워드 뉴럴네트워크 계산 예시 (4)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/FSPzKfl.png" width="400px" title="source: imgur.com" />

그림7과 은닉층 뉴런값이 동일하고 그에 대한 가중치가 $[1,2,1]$이라면 출력층 두번째 뉴런 값은 $9 \times 1 + 2 \times 2 + 0 \times 1 -1=12$가 됩니다. 그림9와 같습니다.

## **그림9** 피드포워드 뉴럴네트워크 계산 예시 (5)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/dJ4EdNu.png" width="400px" title="source: imgur.com" />

이번엔 위의 내용을 코드로 확인해보겠습니다. 코드1은 지금까지 예시를 변수로 정의하는 코드입니다. `x`는 입력이고 `w1`은 입력층-은닉층을 연결하는 가중치, `b1`은 입력층-은닉층을 연결하는 바이어스를 가리킵니다. `w2`는 은닉층-출력층을 연결하는 가중치, `b2`는 은닉층-출력층을 연결하는 바이어스를 가리킵니다.

## **코드1** 피드포워드 뉴럴네트워크 계산 예시 (1)
{: .no_toc .text-delta }
```python
import torch
x = torch.tensor([2,1])
w1 = torch.tensor([[3,2,-4],[2,-3,1]])
b1 = 1
w2 = torch.tensor([[-1, 1], [1,2], [3,1]])
b2 = -1
```

코드2는 실제 계산을 수행하는 코드입니다. 입력 `x`와 `w1`를 행렬곱한 뒤 `b1`을 더한 것이 `h_preact`입니다. 여기에 ReLU를 취해 은닉층 `h`를 만듭니다. 마지막으로 `h`와 `w2`를 행렬곱한 뒤 `b2`를 더해 출력층 `y`를 계산합니다. 행렬곱 연산이 익숙치 않은 분들은 [이 글](https://ko.wikipedia.org/wiki/%ED%96%89%EB%A0%AC_%EA%B3%B1%EC%85%88)을 추가로 참고하시면 좋을 것 같습니다.

## **코드2** 피드포워드 뉴럴네트워크 계산 예시 (2)
{: .no_toc .text-delta }
```python
h_preact = torch.matmul(x, w1) + b1
h = torch.nn.functional.relu(h_preact)
y = torch.matmul(h, w2) + b2
```

그림10은 코드2 수행 결과를 파이썬 콘솔에서 확인한 결과입니다. `h_preact`와 `h`는 그림5\~그림7에 이르는 은닉층 손 계산 예시와 동일한 결과임을 알 수 있습니다. `y`는 그림8과 그림9에 해당하는 출력층 손 계산 예시와 같은 결과입니다.

## **그림10** 피드포워드 뉴럴네트워크 계산 예시
{: .no_toc .text-delta }

```
>>> h_preact
tensor([ 9,  2, -6])
>>> h
tensor([9, 2, 0])
>>> y
tensor([-8, 12])
```

피드포워드 뉴럴네트워크의 학습 대상은 가중치와 바이어스입니다. 코드 예시에서는 `w1`, `b1`, `w2`, `b2`가 학습 대상이 됩니다. 이들은 태스크(예: 기계 번역)를 가장 잘 수행하는 방향으로 학습 과정에서 업데이트됩니다.

한편 트랜스포머에서는 은닉층의 뉴런 갯수(즉 은닉층의 차원수)를 입력층의 네 배로 설정하고 있습니다. 예컨대 피드포워드 뉴럴네트워크의 입력 벡터가 768차원일 경우 은닉층은 2048차원까지 늘렸다가 출력층에서 이를 다시 768차원으로 줄입니다.

### Add : 잔차 연결

트랜스포머 블록의 Add는 잔차 연결(residual connection)을 가리킵니다. 잔차 연결이란 그림11처럼 블록(block) 계산을 건너뛰는 경로를 하나 두는 것을 말합니다. 입력을 $\mathbf{x}$, 이번 계산 대상 블록을 $\mathbb{F}$라고 할 때 잔차 연결은 $\mathbb{F}(\mathbf{x})+\mathbf{x}$로 간단히 실현합니다.

## **그림11** 잔차 연결 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/7DUovTQ.png" width="300px" title="source: imgur.com" />

동일한 블록 계산이 계속될 때 잔차 연결을 두는 것은 제법 큰 효과가 있습니다. 그림12의 좌측처럼 블록 연산을 세 번 수행하고 블록과 블록 사이에 잔차 연결을 모두 적용했다고 가정해 봅시다. 그렇다면 모델은 사실상 그림12 우측처럼 계산하는 형태가 됩니다. 

그림12 우측을 보면 잔차 연결을 두지 않았을 때는 $f_1$, $f_2$, $f_3$을 연속으로 수행하는 경로 한 가지만 존재하였으나, 잔차 연결을 블록마다 설정해둠으로써 모두 8가지의 새로운 경로가 생겼습니다. 다시 말해 모델이 다양한 관점에서 블록 계산을 수행하게 된다는 이야기입니다.

## **그림12** 잔차 연결 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/UHVuX1X.jpg" width="800px" title="source: imgur.com" />

딥러닝 모델은 레이어가 많아지면 학습이 어려운 경향이 있습니다. 모델을 업데이트하기 위한 신호(그래디언트)가 전달되는 경로가 길어지기 때문입니다. 잔차 연결은 모델 중간에 블록을 건너뛰는 경로를 설정함으로써 학습을 용이하게 하는 효과까지 거둘 수 있습니다.

### Norm : 레이어 정규화

레이어 정규화(layer normalization)란 미니 배치의 인스턴스($\mathbf{x}$)별로 평균을 빼주고 표준편차로 나눠줘 정규화(normalization)을 수행하는 기법입니다. 레이어 정규화를 수행하면 학습이 안정되고 그 속도가 빨라지는 등의 효과가 있다고 합니다. 수식2와 같습니다. 수식2에서 $\beta$와 $\gamma$는 학습 과정에서 업데이트되는 가중치이며, $\epsilon$은 분모가 0이 되는 걸 방지하기 위해 더해주는 고정 값(보통 1e-5로 설정)입니다.

## **수식2** 레이어 정규화
{: .no_toc .text-delta }

$$
\mathbf{y}=\frac { \mathbf{x} - \mathop{\mathbb{E}} \left[ \mathbf{x} \right]  }{ \sqrt { \mathop{\mathbb{V}} \left[ \mathbf{x} \right] +\epsilon  }  } *\gamma +\beta 
$$

레이어 정규화는 미니배치의 인스턴스별로 수행합니다. 그림13은 배치 크기가 3인 경우 레이어 정규화 수행 과정의 일부를 나타낸 그림입니다. 배치의 첫번째 데이터($\mathbf{x}=[1,2,3]$)의 평균($\mathbb{E}\left[ \mathbf{x} \right]$)과 표준편차($\sqrt{\mathbb{V}\left[ \mathbf{x} \right]}$)는 각각 2, 0.8164인데요. 이 값들을 바탕으로 수식2를 계산하게 됩니다.

## **그림13** 레이어 정규화(Layer Normalization)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/axo9eTU.png" width="200px" title="source: imgur.com" />

그러면 코드로도 확인해보겠습니다. 코드3은 파이토치로 레이어 정규화를 수행하는 역할을 합니다. 입력(`input`)의 모양은 배치 크기(2) $\times$ 피처의 차원수(3)가 되는데요. `torch.nn.LayerNorm(input.shape[-1])`이라는 말은 피처 대상으로 레이어 정규화를 수행한다는 의미가 됩니다.

## **코드3** 레이어 정규화 예시
{: .no_toc .text-delta }
```python
import torch
input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
m = torch.nn.LayerNorm(input.shape[-1])
output = m(input)
```

배치의 첫 번째 데이터를 수식2에 따라 정규화하면 그 결과는 [-1.2247, 0.0, 1.2247]이 됩니다. 같은 방식으로 두번째 데이터를 정규화하면 [0.0, 0.0, 0.0]이 됩니다. 그림14는 코드14의 `output`을 파이썬 콘솔에서 확인한 결과입니다.

## **그림14** 레이어 정규화 예시
{: .no_toc .text-delta }

```
>>> output
tensor([[-1.2247,  0.0000,  1.2247],
        [ 0.0000,  0.0000,  0.0000]], grad_fn=<NativeLayerNormBackward>)
```

수식2를 자세히 보면 평균을 빼주고 표준편차로 나눠준 결과에 $\gamma$를 곱하고 마지막으로 $\beta$를 더해주는 걸 알 수 있습니다. 그런데 그림14를 보면 이 계산을 생략한 것 같은 인상을 주는군요. 하지만 그림15처럼 확인해 보면 이 의문이 풀립니다. 

## **그림15** 레이어 정규화 예시
{: .no_toc .text-delta }

```
>>> m.weight
Parameter containing:
tensor([1., 1., 1.], requires_grad=True)
>>> m.bias
Parameter containing:
tensor([0., 0., 0.], requires_grad=True)
```

`m.weight`는 $\gamma$, `m.bias`는 $\beta$에 대응하는데요. 파이토치의 `LayerNorm` 객체는 이 두 값을 각각 1과 0으로 초기화합니다. 다시 말해 학습 초기 레이어 정규화 수행은 배치 인스턴스의 평균을 빼주고 표준편차로 나눠준 결과에 1을 곱하고 마지막으로 0을 더해준다는 이야기입니다. 이후 학습 과정에서는 태스크(예: 기계번역)를 가장 잘 수행하는 방향으로 이 값들을 업데이트합니다.

---

## 모델 학습 기법

여기서는 트랜스포머 모델의 학습 기법을 살펴봅니다.

### 드롭아웃

딥러닝 모델은 그 표현력이 아주 좋아서 학습 데이터 그 자체를 외워버릴 염려가 있습니다. 이를 과적합(overfitting)이라고 합니다. 드롭아웃(dropout)은 이러한 과적합 현상을 방지하고자 뉴런의 일부를 확률적으로 0으로 대치하여 계산에서 제외하는 기법입니다. 그림16과 같습니다.

## **그림16** 드롭아웃
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pKUE70B.png" width="300px" title="source: imgur.com" />

드롭아웃을 구현한 파이토치 코드는 코드4입니다. `torch.nn.Dropout` 객체는 뉴런별로 드롭아웃을 수행할지 말지를 확률적으로 결정하는 함수인데요. `p=0.2`라는 말은 드롭아웃 수행 비율이 평균적으로 20%가 되게끔 하겠다는 이야기입니다.

## **코드4** 드롭아웃
{: .no_toc .text-delta }
```python
import torch
m = torch.nn.Dropout(p=0.2)
input = torch.randn(1, 10)
output = m(input)
```

그림17은 코드4의 `input`과 `output`을 파이썬 콘솔에서 확인한 결과입니다. 드롭아웃 수행 결과 `input` 뉴런 가운데 8번째, 10번째가 0으로 대치되었음을 확인할 수 있습니다. 

## **그림17** 드롭아웃 결과
{: .no_toc .text-delta }

```
>>> input
tensor([[ 1.0573,  0.1351, -0.0124,  0.7029,  2.3283, -0.7240,  0.0716,  0.8494,  0.6496, -1.5225]])
>>> output
tensor([[ 1.3217,  0.1689, -0.0155,  0.8786,  2.9104, -0.9050,  0.0895,  0.0000,  0.8120, -0.0000]])
```

참고로 `torch.nn.Dropout`은 안정적인 학습을 위해 각 요솟값에 $1/(1-p)$를 곱하는 역할도 수행합니다. 코드 3-9에서 $p$를 0.2로 설정해 두었으므로 드롭아웃 적용으로 살아남은 요솟값 각각에 1.25를 곱하는 셈입니다. 이에 1.0573는 `torch.nn.Dropout` 수행 후 1.3217로, 0.1351은 0.1689로 변환됐습니다.

트랜스포머 모델에서 드롭아웃은 입력 임베딩과 최초 블록 사이, 블록과 블록 사이, 마지막 블록과 출력층 사이 등에 적용합니다. 드롭아웃 비율은 10%(`p=0.1`)로 설정하는 것이 일반적입니다. 드롭아웃은 학습 과정에만 적용하고, 학습이 끝나고 나서 인퍼런스 과정에서는 적용하지 않습니다.


### 아담 옵티마이저

딥러닝 모델 학습은 모델 출력과 정답 사이의 오차(error)를 최소화하는 방향을 구하고 이 방향에 맞춰 모델 전체의 파라미터(parameter)들을 업데이트하는 과정입니다. 이때 오차를 손실(loss), 오차를 최소화하는 방향을 그래디언트(gradient)라고 합니다. 오차를 최소화하는 과정을 최적화(optimization)라고 합니다.

파라미터란 행렬, 벡터, 스칼라 따위의 모델 구성 요소입니다. 이 값들은 학습 데이터로 구합니다. 예를 들어 대한민국 남성의 키를 ‘정규 분포’라는 모델로 나타낸다고 가정한다면 대한민국 남성 키의 평균과 표준편차가 이 모델의 파라미터가 됩니다.

딥러닝 모델을 학습하려면 우선 오차부터 구해야 합니다. 오차를 구하려면 현재 시점의 모델에 입력을 넣어봐서 처음부터 끝까지 계산해보고 정답과 비교해야 합니다. 오차를 구하기 위해 이같이 모델 처음부터 끝까지 순서대로 계산해보는 과정을 순전파(forward propagation)이라고 합니다.

오차를 구했다면 오차를 최소화하는 최초의 그래디언트를 구할 수 있습니다. 이는 미분(devative)으로 구합니다. 이후 [미분의 연쇄 법칙(chain rule)](https://ko.wikipedia.org/wiki/%EC%97%B0%EC%87%84_%EB%B2%95%EC%B9%99)에 따라 모델 각 가중치별 그래디언트 역시 구할 수 있습니다. 이 과정은 순전파의 역순으로 순차적으로 수행되는데요. 이를 역전파(backpropagation)라고 합니다. 그림18은 순전파와 역전파를 개념적으로 나타낸 그림입니다.

## **그림18** 순전파와 역전파
{: .no_toc .text-delta }
<img src="https://i.imgur.com/b551jfH.png" width="300px" title="source: imgur.com" />

모델을 업데이트하는 과정, 즉 학습 과정은 미니 배치 단위로 이뤄지는데요. 이는 눈을 가린 상태에서 산등성이를 한걸음씩 내려가는 과정에 비유할 수 있습니다. 내가 지금 있는 위치에서 360도 모든 방향에 대해 한발한발 내딛어보고 가장 경사가 급한 쪽으로 한걸음씩 내려가는 과정을 반복하는 것입니다.

모델을 업데이트할 때(산등성이를 내려갈 때) 중요한 것은 방향과 보폭일 겁니다. 이는 최적화 도구(optimizer)의 도움을 받는데요. 트랜스포머 모델이 쓰는 최적화 도구가 바로 아담 옵티마이저(Adam Optimizer)입니다. 아담 옵티마이저는 오차를 줄이는 성능이 좋아서 트랜스포머 말고도 널리 쓰이고 있습니다.

아담 옵티마이저의 핵심 동작 원리는 방향과 보폭을 적절하게 정해주는 겁니다. 방향을 정할 때는 현재 위치에서 가장 경사가 급한 쪽으로 내려가되, 여태까지 내려오던 관성(방향)을 일부 유지하도록 합니다. 보폭의 경우 안가본 곳은 성큼 빠르게 걸어 훑고 많이 가본 곳은 갈수록 보폭을 줄여 세밀하게 탐색하는 방식으로 정합니다. 

코드5는 아담 옵티마이저를 사용하는 파이토치 코드입니다. 최초의 보폭(러닝 레이트, learning rate)를 정해주면 아담 옵티마이저가 최적화 대상 가중치들(`model.parameters()`)에 방향과 보폭을 정해줍니다.

## **코드5** 아담 옵티마이저
{: .no_toc .text-delta }

```python
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=model.learning_rate)
```

참고로 이 책 실습에서는 pytorch-lighting 라이브러리의 lightning 모듈의 도움을 받아 task를 정의합니다. 여기엔 모델과 최적화 방법, 학습 과정 등이 포함돼 있습니다.


---
