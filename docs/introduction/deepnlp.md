---
layout: default
title: Deep NLP
parent: Introduction
nav_order: 1
---

# 딥러닝 자연어 처리 모델 만들기
{: .no_toc }

딥러닝(Deep Learning) 기반 자연어 처리 모델의 주요 특징을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 딥러닝 모델의 학습이란

자연어 처리 모델은 일종의 함수(function)로 이해할 수 있습니다. 예컨대 우리가 영화 리뷰의 감성(sentiment)을 맞추는 자연어 처리 모델을 만든다고 가정해 봅시다. 그러면 우리가 만든 감성 분석 모델은 아래처럼 함수 $f$로 써볼 수 있습니다. 

이 $f$는 영화 리뷰를 입력 받으면 복잡한 내부 계산 과정을 거친 후 감성 점수, 즉 $\begin{bmatrix} \text{positive} & \text{neutral} & \text{negative} \end{bmatrix}$을 출력하는 역할을 수행합니다. 예컨대 수식1과 같습니다.


## **수식1** 감성 분석 모델
{: .no_toc .text-delta }

$$
\begin{align*}
f\left( \text{재미가 없는 편인 영화에요} \right) = [ 0.0, 0.3, 0.7 ] \\
f\left( \text{단언컨대 이 영화 재미 있어요} \right) = [ 1.0, 0.0, 0.0 ]
\end{align*}
$$

이 모델을 만들려면 무엇을 해야 할까요? 우선 데이터부터 준비해야 합니다. 아래처럼 각 문장에 '감성'이라는 꼬리표 혹은 레이블을 달아놓은 자료가 필요합니다. 이를 학습 데이터라고 부릅니다.

## **표1** 감성 학습 데이터
{: .no_toc .text-delta }

|문장|긍정|중립|부정|
|---|---|---|---|
|단언컨대 이 영화 재미 있어요|1|0|0|
|단언컨대 이 영화 재미 없어요|0|0|1|
|...|...|...|...|


그 다음은 모델이 데이터의 패턴(pattern)을 스스로 익히게 해야 합니다. 이를 **학습(train)**이라고 합니다. `단언컨대 이 영화 재미 있어요` 문장을 학습하는 상황이라고 가정해 봅시다. 학습 초기 $f$는 `단언컨대 이 영화 재미 있어요`를 입력 받으면 아래처럼 출력할 겁니다. 문장이 어떤 감성인지 전혀 모르는 상황입니다.


## **그림1** 감성 분석 모델의 출력 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YUyysLA.jpg" width="250px" title="source: imgur.com" />

그런데 우리는 단언컨대 이 영화 재미 있어요이라는 문장의 감성, 즉 정답이 $\begin{bmatrix} 1 & 0 & 0 \end{bmatrix}$임을 알고 있습니다. 그런데 현재 모델의 출력(아래 그림의 회색 막대)과 정답을 비교해 보면 중립/부정 점수가 높네요. **깎아야 합니다.** 긍정 점수는 낮네요. **높여 줍니다.** $f$가 `단언컨대 이 영화 재미 있어요`라는 입력을 받았을 때 긍정 점수는 높아지고, 중립/부정 점수는 낮아지도록 모델 전체를 업데이트합니다. 

## **그림2** 감성 분석 모델의 출력 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nTBwj6u.jpg" width="250px" title="source: imgur.com" />

업데이트를 한번 했는데도 현재 모델의 출력(아래 그림의 회색 막대)과 정답을 비교했을 때 여전히 긍정 점수는 낮습니다. 한번 더 **높여 줍니다.** 중립/부정 점수는 여전히 높네요. 또 **깎아 줍니다.** $f$가 `단언컨대 이 영화 재미 있어요`라는 입력을 받았을 때 긍정 점수는 높아지고, 중립/부정 점수는 낮아지도록 모델 전체를 업데이트해 줍니다.

## **그림3** 감성 분석 모델의 출력 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/e87KFzd.jpg" width="250px" title="source: imgur.com" />

이런 업데이트를 여러번 수행하면 종국에는 $f$가 아래 그림처럼 정답에 가까운 출력을 낼 수 있습니다. 이렇게 모델을 업데이트하는 과정 전체를 학습(train)이라고 부릅니다. 모델이 입력-출력 사이의 패턴을 스스로 익히는 과정입니다.

## **그림4** 감성 분석 모델의 출력 (4)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5Dzq7Qz.jpg" width="250px" title="source: imgur.com" />


---