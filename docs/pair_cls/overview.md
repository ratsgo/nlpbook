---
layout: default
title: Overview
parent: Pair Classification
nav_order: 1
---

# 전체적으로 훑어보기
{: .no_toc }

이 장에서는 자연어 추론(NLI) 과제를 실습합니다. 모델 아키텍처, 입/출력 등 전반을 조망합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## 과제 소개

문장 쌍 분류(Sentence Pair Classification)이란 문장 두 개가 주어졌을 때 해당 문장 사이의 관계가 어떤 범주일지 분류하는 과제(task)입니다. 문장 쌍 분류의 대표 예시로 자연어 추론(Natural Langugage Inference; NLI)이 있습니다. 두 개 문장이 참(entailment), 거짓(contradiction), 중립(neutral)인지 가려내는 것입니다. 예컨대 다음과 같습니다.

- 나 출근했어 + 난 백수야 → 거짓(contradiction)

이번 튜토리얼에서 사용할 데이터는 카카오브레인에서 공개한 [KorNLI](https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorNLI) 데이터셋입니다. 영어 NLI 데이터셋을 번역한 데이터로 다음과 같이 구성되어 있습니다. 아래에서 살펴볼 수 있듯 전제(premise)에 대한 가설(hypothesis)이 참(`entailment`)인지, 거짓(`contradiction`)인지, 중립 혹은 판단불가(`neutral`)인지 정보가 레이블(label)로 주어져 있습니다. 여기에서 `entailment`는 함의, `contradiction`은 모순으로 번역되기도 합니다.

- **전제(premise)**: 나는 정보가 부족해요.
- **가설(hypothesis)**: 내게 필요할 정보는 다 갖고 있어요.
- **레이블(label)**: `contradiction`

- **전제(premise)**: 나는 정보가 부족해요.
- **가설(hypothesis)**: 나는 어떤 차를 구입할지 결정하려면 더 많은 정보가 필요하다.
- **레이블(label)**: `neutral`

- **전제(premise)**: 나는 정보가 부족해요.
- **가설(hypothesis)**: 나는 이 일에 대한 정보를 더 모으고 싶어요.
- **레이블(label)**: `entailment`


우리가 만들 NLI 과제 수행 모델은 전제와 가설 두 개 문장을 입력으로 하고, 두 문장의 관계가 어떤 범주일지 확률(`entailment`, `contradiction`, `neutral`)을 출력으로 합니다. 예컨대 다음과 같습니다.

- 나는 정보가 부족해요 + 내게 필요할 정보는 다 갖고 있어요 → [0.02, 0.97, 0.01]
- 나는 정보가 부족해요 + 나는 이 일에 대한 정보를 더 모으고 싶어요 → [0.98, 0.01, 0.01]

NLI 모델의 출력은 확률입니다. 적당한 후처리(post processing) 과정을 거쳐 사람이 보기에 좋은 형태로 가공해 줍니다. 

- 나는 정보가 부족해요 + 내게 필요할 정보는 다 갖고 있어요 → [0.02, 0.97, 0.01] → `contradiction`
- 나는 정보가 부족해요 + 나는 이 일에 대한 정보를 더 모으고 싶어요 → [0.98, 0.01, 0.01] → `entailment`


---


## 모델 구조


우리 책에서 사용하는 문장 쌍 분류 모델은 그림1과 같은 구조입니다. 전제와 가설 두 문장을 각각 토큰화한 뒤 `[CLS] + 전제 + [SEP] + 가설 + [SEP]` 형태로 이어 붙입니다. 여기에서 `CLS`는 문장 시작을 알리는 스페셜 토큰, `SEP`는 전제와 가설을 서로 구분해주는 의미의 스페셜 토큰입니다. 

이를 BERT 모델에 입력하고 BERT 모델 마지막 레이어 출력 가운데 `CLS`에 해당하는 벡터를 뽑습니다. 이 벡터에 작은 추가 모듈을 덧붙여 모델 전체의 출력이 [전제에 대해 가설이 참(entailment)일 확률, 전제에 대해 가설이 거짓(contradiction)일 확률, 전제에 대해 가설이 중립(neutral)일 확률] 형태가 되도록 합니다.


## **그림1** 문장 쌍 분류
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Wd9UK1s.png" width="500px" title="source: imgur.com" />


---

## 태스크 모듈

`CLS` 벡터 뒤에 붙는 추가 모듈의 구조는 그림 4-2와 같습니다. 우선 `CLS` 벡터(그림 4-2에서 $\mathbf{x}$)에 드롭아웃(dropout)을 적용합니다. 그 다음 가중치 행렬(weight matrix)을 곱해줘 `CLS`를 분류해야할 범주 수만큼의 차원을 갖는 벡터로 변환합니다(그림2 $\mathbf{x}$에서 $\text{net}$). 만일 `CLS` 벡터가 768차원이고 분류 대상 범주 수가 3개(`참`, `거짓`, `중립`)라면 가중치 행렬 크기는 768 $\times$ 3이 됩니다. 


## **그림2** 문장 쌍 분류 태스크 모듈
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qub0zK2.png" width="350px" title="source: imgur.com" />

이후 여기에 소프트맥스(softmax) 함수를 취해 준 것이 모델의 최종 출력이 됩니다. 소프트맥스 함수는 입력 벡터의 모든 요소(element) 값의 범위를 0\~1로 하고 모든 요소값의 총합을 1이 되게끔 합니다. 다시 말해 어떤 입력 벡터든 소프트맥스 함수를 적용하면 해당 벡터가 확률로 변환된다는 이야기입니다. 

그림2의 문장 쌍 분류 태스크 모듈은 [4-1장 문서 분류 태스크 모듈](https://ratsgo.github.io/nlpbook/docs/doc_cls/overview)과 거의 유사한 모습인 걸 확인할 수 있습니다. 4-1장 문서 분류 과제를 세 개 범주(예컨대 긍정, 부정, 중립)를 분류하는 태스크로 상정한다면 완전히 동일한 모듈 구조를 갖습니다. 다만 차이는 이 모듈의 입력($\mathbf{x}$)이 됩니다. 태스크 모듈의 입력으로 문장 하나의 임베딩이 주어진다면 문서 분류, 두 개의 임베딩이 주어진다면 문장 쌍 분류 과제가 됩니다.

이렇게 만든 모델의 최종 출력과 정답 레이블을 비교해 모델 출력이 정답 레이블과 최대한 같아지도록 BERT 레이어 전체를 포함한 모델 전체를 업데이트합니다. 이를 학습(train)이라고 합니다. 딥러닝 모델의 학습과 관련해서는 [1장 Deep NLP](https://ratsgo.github.io/nlpbook/docs/introduction/deepnlp/) 챕터를 참고하시면 좋을 것 같습니다.


---