---
layout: default
title: Overview
parent: Document Classification
nav_order: 1
---

# 전체적으로 훑어보기
{: .no_toc }

이 장에서는 문서 분류 과제를 실습합니다. 모델 아키텍처, 입/출력 등 전반을 조망합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 과제 소개

문서 분류(document classification)란 문서가 주어졌을 때 해당 문서의 범주를 분류하는 과제입니다. 뉴스를 입력으로 하고 범주(정치, 경제, 연예 등)를 맞추거나, 영화 리뷰가 어떤 극성(polarity, 긍정/부정 등)을 가지는지 분류하는 작업이 대표적인 예시가 되겠습니다.

이번 튜토리얼에서 사용할 데이터는 박은정 님이 공개하신 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)인데요. 우리가 만들 문서 분류 모델은 다음과 같은 입/출력 형태를 갖습니다. 문장(영화 리뷰)을 입력으로 하고요, 해당 문장이 속한 범주 확률(`긍정`, `부정`)를 출력으로 합니다.

- 진짜 짜증나네요 목소리 → [0.02, 0.98]
- 너무재밓었다그래서보는것을추천한다 → [0.99, 0.01]

문서 분류 모델의 출력은 확률입니다. 적당한 후처리(post processing) 과정을 거쳐 사람이 보기에 좋은 형태로 가공해 줍니다. 이와 같이 문장의 극성을 분류하는 과제를 감성 분석(Sentiment Analysis)이라고 합니다.

- 진짜 짜증나네요 목소리 → [0.02, 0.98] → 부정(negative)
- 너무재밓었다그래서보는것을추천한다 → [0.99, 0.01] → 긍정(negative)


---

## 모델 구조

우리 책에서 사용하는 문서 분류 모델은 그림 4-1과 같은 구조입니다. 입력 문장을 토큰화한 뒤 문장 시작과 끝을 알리는 스페셜 토큰 `CLS`와 `SEP`를 각각 원래 토큰 시퀀스 앞뒤에 붙입니다. 이를 BERT 모델에 입력하고 BERT 모델 마지막 레이어 출력 가운데 `CLS`에 해당하는 벡터를 뽑습니다. 이 벡터에 작은 추가 모듈을 덧붙여 모델 전체의 출력이 [해당 문장이 긍정일 확률, 해당 문장이 부정일 확률] 형태가 되도록 합니다.

## **그림 4-1** 문서 분류
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5lpkDEB.png" width="350px" title="source: imgur.com" />


---

## 태스크 모듈


`CLS` 벡터 뒤에 붙는 추가 모듈의 구조는 그림 4-2와 같습니다. 우선 `CLS` 벡터(그림 4-2에서 $\mathbf{x}$)에 드롭아웃(dropout)을 적용합니다.

## **팁**: 드롭아웃(dropout)
{: .no_toc .text-delta }
드롭아웃을 적용한다는 의미는 그림 4-2에서 입력 벡터 $\mathbf{x}$의 768개 각 요소값 가운데 일부를 랜덤으로 0으로 바꾸어서 이후 계산에 포함되지 않도록 하는 것입니다. 드롭아웃 관련 자세한 내용은 [3-2장 Techincs](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하시면 좋을 것 같습니다.
{: .fs-3 .ls-1 .code-example }

그 다음 가중치 행렬(weight matrix)을 곱해줘 `CLS`를 분류해야할 범주 수만큼의 차원을 갖는 벡터로 변환합니다(그림 4-2 $\mathbf{x}$에서 $\text{net}$). 만일 `CLS` 벡터가 768차원이고 분류 대상 범주 수가 2개(`긍정`, `부정`)라면 가중치 행렬 크기는 768 $\times$ 2가 됩니다. 

## **그림 4-2** 문서 분류 태스크 모듈
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NwcI0H2.png" width="350px" title="source: imgur.com" />

이후 여기에 소프트맥스(softmax) 함수를 취해 준 것이 모델의 최종 출력이 됩니다. 소프트맥스 함수는 입력 벡터의 모든 요소(element) 값의 범위를 0\~1로 하고 모든 요소값의 총합을 1이 되게끔 합니다. 다시 말해 어떤 입력 벡터든 소프트맥스 함수를 적용하면 해당 벡터가 확률로 변환된다는 이야기입니다. 

이렇게 만든 모델의 최종 출력과 정답 레이블을 비교해 모델 출력이 정답 레이블과 최대한 같아지도록 BERT 레이어 전체를 포함한 모델 전체를 업데이트합니다. 이를 학습(train)이라고 합니다. 딥러닝 모델의 학습과 관련해서는 [1장 Deep NLP](https://ratsgo.github.io/nlpbook/docs/introduction/deepnlp/) 챕터를 참고하시면 좋을 것 같습니다.


---
