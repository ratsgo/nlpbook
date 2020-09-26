---
layout: default
title: Transfer Learning
parent: Introduction
nav_order: 2
---

# Transfer Learning
{: .no_toc }

본 사이트에서 소개하는 자연어 처리 모델 학습 방법은 트랜스퍼 러닝(Transfer Learning)이라는 기법을 씁니다. 이 장에서는 프리트레인(pretrain), 파인튜닝(finetuning), 퓨샷 러닝(few-shot learning), 제로샷 러닝(zero-shot learning) 등 이와 관련된 개념을 설명합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 트랜스퍼 러닝 (Transfer Learning)

**트랜스퍼 러닝(transfer learning)**이란 특정 태스크를 학습한 모델을 다른 태스크 수행에 재사용하는 기법을 가리킵니다. 그림1처럼 `Task2`를 수행하는 모델을 만든다고 가정해 봅시다. 이 경우 트랜스퍼 러닝이 꽤 도움이 될 수 있습니다. 모델이 `Task2`를 배울 때 `Task1`을 수행해봤던 경험을 재활용하기 때문입니다. 비유하자면 사람이 새로운 지식을 배울 때 그가 평생 쌓아왔던 지식을 요긴하게 다시 써먹는 것과 같습니다.

## **그림1** 트랜스퍼 러닝 (Transfer Learning)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/cNvHXxj.png" width="400px" title="source: imgur.com" />

트랜스퍼 러닝을 적용하면 모델의 학습 속도가 빨라지고 새로운 태스크(`Task2`)를 이전보다 잘 수행하는 경향이 있습니다. 이 때문에 트랜스퍼 러닝은 최근 각광받고 있는 테크닉으로 널리 쓰이고 있습니다. BERT(Bidirectional Encoder Representations from Transformers)나 GPT(Generative Pre-trained Transformer) 등이 바로 이 기법을 쓰고 있습니다.

트랜스퍼 대상 모델을 처음 학습하는 과정, 즉 그림1의 `Task1`을 수행하는 과정을 **프리트레인(pretrain)**이라고 합니다. `Task2`를 본격적으로 수행하기에 앞서(pre) 학습(train)한다는 의미에서 이런 용어가 붙은 것 같습니다.

트랜스퍼 대상 모델을 새로운 태스크(`Task2`)에 학습하는 과정은 여러 가지 용어로 불리고 있습니다. 그 방식이 참으로 다양하기 때문인데요. **파인튜닝(finetuning)**, **제로샷 러닝(zero-shot learning)**, **원샷 러닝(one-shot learning)**, **퓨샷 러닝(few-shot learning)** 등이 바로 그것입니다. 이 장의 마지막 챕터에서 다시 살펴보겠습니다.

한편 그림2의 `Task2`는 **다운스트림(downstream) 태스크**라고 부릅니다. 문서 분류, 개체명 인식 등 우리가 풀고 싶은 자연어 처리의 구체적 문제들을 가리킵니다. `Task1`은 **업스트림(upstream) 태스크**라고 불리는 데요. 비유컨대 업스트림과 다운스트림 태스크의 관계는 '윗물(up)이 맑아야 아랫물(down)이 맑을' 수 있는 것과 같습니다. 다시 말해 `Task1`을 잘 수행하는 모델이 `Task2`도 잘 수행할 수 있다는 것이지요.


---


## 프리트레인 (pretrain)



## **그림2** 언어 모델 (Language Model)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/r8s1POC.png" width="200px" title="source: imgur.com" />

## **그림3** 마스크 언어 모델 (Masked Language Model)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kfkf6bw.png" width="220px" title="source: imgur.com" />


---

## 다운스트림 태스크 (Downstream Tasks)

## **그림1** 문서 분류
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5lpkDEB.png" width="350px" title="source: imgur.com" />

## **그림2** 개체명 인식
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I0Fdtfe.png" width="350px" title="source: imgur.com" />

## **그림3** 질의/응답
{: .no_toc .text-delta }
<img src="https://i.imgur.com/eHKCry2.png" width="500px" title="source: imgur.com" />

## **그림4** 문서 검색
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ur0NrHC.png" width="800px" title="source: imgur.com" />

## **그림5** 문장 생성
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Ui71883.png" width="250px" title="source: imgur.com" />

---


## 파인튜닝, 제로샷 러닝, 퓨샷 러닝...


---

## References

- []()


---