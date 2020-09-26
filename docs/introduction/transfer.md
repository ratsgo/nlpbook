---
layout: default
title: Transfer Learning
parent: Introduction
nav_order: 2
---

# 트랜스퍼 러닝 (Transfer Learning)
{: .no_toc }

이 책에서 소개하는 자연어 처리 모델 학습 방법은 트랜스퍼 러닝(Transfer Learning)이라는 기법을 씁니다. 이 장에서는 프리트레인(pretrain), 파인튜닝(finetuning), 퓨샷 러닝(few-shot learning), 제로샷 러닝(zero-shot learning) 등 이와 관련된 개념을 설명합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 트랜스퍼 러닝

**트랜스퍼 러닝(transfer learning)**이란 특정 태스크를 학습한 모델을 다른 태스크 수행에 재사용하는 기법을 가리킵니다. 그림1처럼 `Task2`를 수행하는 모델을 만든다고 가정해 봅시다. 이 경우 트랜스퍼 러닝이 꽤 도움이 될 수 있습니다. 모델이 `Task2`를 배울 때 `Task1`을 수행해봤던 경험을 재활용하기 때문입니다. 비유하자면 사람이 새로운 지식을 배울 때 그가 평생 쌓아왔던 지식을 요긴하게 다시 써먹는 것과 같습니다.

## **그림1** 트랜스퍼 러닝 (Transfer Learning)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/cNvHXxj.png" width="400px" title="source: imgur.com" />

트랜스퍼 러닝을 적용하면 모델의 학습 속도가 빨라지고 새로운 태스크(`Task2`)를 이전보다 잘 수행하는 경향이 있습니다. 이 때문에 트랜스퍼 러닝은 최근 널리 쓰이고 있습니다. BERT(Bidirectional Encoder Representations from Transformers)나 GPT(Generative Pre-trained Transformer) 등이 바로 이 기법을 쓰고 있습니다.

그림2의 `Task1`은 **업스트림(upstream) 태스크**라고 불립니다. `Task2`는 이와 대비된 개념으로 **다운스트림(downstream) 태스크**라고 합니다. `Task2`는 문서 분류, 개체명 인식 등 우리가 풀고 싶은 자연어 처리의 구체적 문제들을 가리킵니다.

한편 업스트림 태스크를 학습하는 과정을 **프리트레인(pretrain)**이라고 합니다. 다운스트림 태스크를 본격적으로 수행하기에 앞서(pre) 학습(train)한다는 의미에서 이런 용어가 붙은 것 같습니다.

다운스트림 태스크를 학습하는 과정은 여러 가지 용어로 불리고 있습니다. 그 방식이 참으로 다양하기 때문인데요. **파인튜닝(finetuning)**, **제로샷 러닝(zero-shot learning)**, **원샷 러닝(one-shot learning)**, **퓨샷 러닝(few-shot learning)** 등이 바로 그것입니다. 이 장의 마지막 챕터에서 다시 살펴보겠습니다.


---


## 업스트림 태스크

트랜스퍼 러닝이 주목받게 된 것은 업스트림 태스크와 프리트레인 덕분입니다. 이 단계에서 해당 자연어의 풍부한 문맥(context)을 모델에 내재화하고 이 모델을 다양한 다운스트림 태스크에 적용해 그 성능을 대폭 끌어올리게 된 것이죠.

대표적인 업스트림 태스크 가운데 하나가 **다음 단어 맞추기**입니다. GPT 계열 모델이 바로 이 태스크로 프리트레인을 수행합니다. 그림2처럼 `티끌 모아`라는 문맥이 주어져 있고 학습 데이터 말뭉치에 `티끌 모아 태산`이라는 구(phrase)가 많이 있을 경우 모델은 이를 바탕으로 다음 단어를 `티끌`로 분류하도록 학습됩니다.

모델이 대규모 말뭉치를 가지고 그림2 과정을 반복 수행하게 되면 이전 문맥과 다음에 올 단어를 비교해 어떤 단어가 오는 것이 자연스러운지 아닌지 알 수 있게 됩니다. 다시 말해 해당 언어의 풍부한 문맥을 이해할 수 있게 된다는 것이죠. 이같이 '다음 단어 맞추기'로 업스트림 태스크를 수행한 모델을 **언어 모델(Language Model)**이라고 합니다.

## **그림2** 다음 단어 맞추기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/r8s1POC.png" width="200px" title="source: imgur.com" />

언어모델을 학습하는 것은 [딥러닝 모델의 학습](https://ratsgo.github.io/nlpbook/docs/introduction/deepnlp/#%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8%EC%9D%98-%ED%95%99%EC%8A%B5) 챕터에서 언급한 감성 분석 모델의 학습 과정과 별반 다르지 않습니다. 감성 분석 예시에서는 분류해야 할 범주의 수가 3개(긍정, 중립, 부정)뿐이었는데요. 언어모델에서는 분류 대상 범주 수가 학습 대상 언어의 어휘 수(보통 수만 개 이상)가 됩니다. 예컨대 `티끌 모아` 다음 단어의 정답이 `태산`이라면 `태산`이라는 단어에 해당하는 확률은 높이고 나머지 단어에 관계된 확률은 낮추는 방향으로 모델 전체를 업데이트합니다. 그림3과 같습니다.

## **그림3** 언어 모델 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TlR365S.jpg" width="300px" title="source: imgur.com" />


또다른 업스트림 태스크로는 **빈칸 채우기**가 있습니다. BERT 계열 모델이 바로 이 태스크로 프리트레인을 수행합니다. 그림3처럼 `티끌 모아 태산`이라는 문장의 `모아`를 빈칸으로 만들고 해당 빈칸에 들어갈 단어가 무엇일지 맞추는 과정에서 모델이 학습됩니다.

모델이 다량의 데이터를 가지고 그림3을 반복 학습하게 되면 빈칸 앞뒤의 문맥을 보고 빈칸에 올 단어가 무엇이 되는 것이 자연스러운지 알 수 있게 됩니다. 이 태스크를 수행한 모델 역시 언어 모델과 마찬가지로 해당 언어의 풍부한 문맥을 내재화할 수 있게 됩니다. 이같이 '빈칸 채우기'로 업스트림 태스크를 수행한 모델을 **마스크 언어 모델(Masked Language Model)**이라고 합니다.

## **그림4** 빈칸 채우기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kfkf6bw.png" width="220px" title="source: imgur.com" />

마스크 언어모델 학습 역시 언어모델과 다르지 않습니다. `티끌 [빈칸] 태산`에서 빈칸 정답이 `모아`라면 `모아`라는 단어에 해당하는 확률은 높이고 나머지 단어에 관계된 확률은 낮추는 방향으로 모델 전체를 업데이트합니다. 그림5와 같습니다.

## **그림5** 마스크 언어 모델 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YHa2Vva.jpg" width="300px" title="source: imgur.com" />

요컨대 언어모델, 마스크 언어모델 둘 모두 그 본질은 **분류(classification)** 과제를 수행하는 모델이라는 점입니다. 두 모델은 자연어 입력을 받아 전체 어휘 각각에 확률값을 부여합니다. 모델이 출력한 결과는 태스크에 관련된 값들입니다. 언어모델이라면 입력 다음에 올 단어로써 얼마나 자연스러운지를, 마스크 언어모델이라면 빈칸에 올 단어로써 얼마나 그럴듯한지를 나타내는 지표가 됩니다.

---

## 다운스트림 태스크

우리가 모델을 업스트림 태스크로 프리트레인한 근본 이유는 다운스트림 태스크를 잘 하기 위해서입니다. 앞서 설명했듯 다운스트림 태스크는 우리가 풀어야할 자연어 처리의 구체적 과제들을 가리킵니다. 보통 다운스트림 태스크는 프리트레인을 마친 모델을 구조 변경 없이 그대로 사용하거나, 여기에 작은 추가 모듈을 덧붙인 형태로 수행합니다.

다운스트림 태스크 역시 업스트림 태스크와 마찬가지로 그 본질은 **분류(classification)**입니다. 다시 말해 자연어 입력을 받아 해당 입력이 어떤 범주에 해당하는지 확률 형태로 반환한다는 이야기입니다. 문장 생성을 제외한 대부분의 과제에서는 프리트레인을 마친 마스크 언어 모델(BERT 계열)을 사용합니다. 각 다운스트림 태스크를 차례대로 살펴보겠습니다.

그림6은 문서 분류를 수행하는 모델을 도식적으로 나타낸 것입니다. 이 문서 분류 모델은 자연어 입력(문서 혹은 문장)을 받아 해당 입력이 어떤 범주(긍정, 중립, 부정 따위)에 속하는지 그 확률값을 반환합니다. 구체적으로는 프리트레인을 마친 마스크 언어모델(그림6에서 노란색 박스 이하 모듈) 위에 작은 모듈을 하나 더 쌓아 문서 전체의 범주를 분류하는 방식입니다. 문서 분류 과제 튜토리얼은 [4장 Document Classification](https://ratsgo.github.io/nlpbook/docs/classification)을 참고하세요.

한편 그림6 이후에 나오는 `CLS`, `SEP`는 각각 문장의 시작과 끝에 붙이는 특수한 토큰(token)입니다. 자세한 내용은 [3장 Vocab & Tokenization](https://ratsgo.github.io/nlpbook/docs/preprocess)을 보세요.

## **그림6** 문서 분류
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5lpkDEB.png" width="350px" title="source: imgur.com" />

그림7은 개체명 인식을 수행하는 모델을 나타냈습니다. 이 모델은 자연어 입력(문서 혹은 문장)을 받아 각 단어별로 어떤 개체명 범주(기관명, 인명, 지명 등)에 속할지 그 확률값을 리턴합니다. 구체적으로는 프리트레인을 마친 마스크 언어모델(그림7에서 노란색 박스 이하 모듈) 위에 입력 문장 전체의 각 단어별로 작은 모듈을 각각 더 쌓아 단어별로 개체명 범주를 분류하는 방식입니다. 개체명 인식 과제 튜토리얼은 [5장 Named Entity Recognition](https://ratsgo.github.io/nlpbook/docs/ner)을 참고하세요.

## **그림7** 개체명 인식
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I0Fdtfe.png" width="350px" title="source: imgur.com" />

그림8은 질의에 알맞은 답을 하는 질의/응답 과제를 수행한 모델입니다. 질의/응답 모델은 자연어 입력(질문, 문서)을 받아 문서의 각 단어가 정답의 시작인지 아닌지, 끝인지 아닌지 관련된 확률값을 반환합니다. 구체적으로는 프리트레인을 마친 마스크 언어모델(그림8에서 노란색 박스 이하 모듈) 위에 문서의 각 단어별로 작은 모듈을 각각 더 쌓아 문서에 속한 단어별로 정답의 시작/끝을 분류하는 방식입니다. 질의/응답 과제 튜토리얼은 [6장 Question Answering](https://ratsgo.github.io/nlpbook/docs/qa)을 참고하세요.

## **그림8** 질의/응답
{: .no_toc .text-delta }
<img src="https://i.imgur.com/eHKCry2.png" width="500px" title="source: imgur.com" />

그림9는 질의에 알맞는 문서를 찾는 검색 과제를 수행한 모델을 도식적으로 나타냈습니다. 검색 모델은 자연어 입력(질문, 문서)을 받아 질문과 문서가 관련이 있는지 없는지 관련된 확률값을 출력합니다. 구체적으로는 프리트레인을 마친 2개의 마스크 언어모델에 질문과 문서를 각각 입력하고, 그 위에 작은 모듈 하나를 더 쌓아 질문과 문서가 관련성 있는지/없는지를 분류하는 방식입니다. 문서 검색 과제 튜토리얼은 [7장 Search](https://ratsgo.github.io/nlpbook/docs/search)를 참고하세요.

## **그림9** 문서 검색
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ur0NrHC.png" width="800px" title="source: imgur.com" />

그림10은 문장 생성 태스크를 수행하는 모델을 나타냈습니다. 다른 과제와 달리 문장 생성은 언어모델(GPT 계열)이 마스크 언어모델보다 품질이 좋아서 언어모델이 자주 쓰이는 편입니다. 문장 생성 모델은 자연어 입력(문장)을 받아 어휘 전체에 대해 확률값을 부여합니다. 이 확률값은 입력에 대해 다음 단어로 적절한지에 관련된 스코어입니다. 구체적으로는 프리트레인을 마친 언어모델을 구조 변경 없이 그대로 사용해, 문맥에 이어지는 적절한 다음 단어를 분류하는 방식입니다. 문장 생성 과제 튜토리얼은 [8장 Sentence Generation](https://ratsgo.github.io/nlpbook/docs/generation)을 참고하세요.

## **그림10** 문장 생성
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Ui71883.png" width="250px" title="source: imgur.com" />

---


## 파인튜닝, 제로샷/원샷/퓨샷 러닝

다운스트림 태스크를 학습하는 방식은 다양합니다. 크게 다음 네 가지 경우가 있습니다.

- **파인튜닝(finetuning)** : 다운스트림 태스크에 해당하는 데이터 전체를 사용합니다. 모델 전체를 다운스트림 데이터에 맞게 업데이트합니다.
- **제로샷러닝(zero-shot learning)** : 다운스트림 태스크 데이터를 전혀 사용하지 않습니다. 모델이 바로 다운스트림 태스크를 수행합니다.
- **원샷러닝(one-shot learning)** : 다운스트림 태스크 데이터를 한 건만 사용합니다. 모델 전체를 1건의 데이터에 맞게 업데이트합니다. 업테이트 없이 수행하는 원샷러닝도 있습니다. 모델이 1건의 데이터가 어떻게 수행되는지 참고한 뒤 바로 다운스트림 태스크를 수행합니다.
- **퓨샷러닝(few-shot learning)** : 다운스트림 태스크 데이터를 몇 건만 사용합니다. 모델 전체를 몇 건의 데이터에 맞게 업데이트합니다. 업데이트 없이 수행하는 퓨삿러닝도 있습니다. 모델이 몇 건의 데이터가 어떻게 수행되는지 참고한 뒤 바로 다운스트림 태스크를 수행합니다.

---

## 참고 문헌

- [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.](https://arxiv.org/abs/1810.04805)
- [Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 9.](https://www.ceid.upatras.gr/webpages/faculty/zaro/teaching/alg-ds/PRESENTATIONS/PAPERS/2019-Radford-et-al_Language-Models-Are-Unsupervised-Multitask-%20Learners.pdf)
- [Brown, Tom B.; Mann, Benjamin; Ryder, Nick; Subbiah, Melanie; Kaplan, Jared; Dhariwal, Prafulla; Neelakantan, Arvind; Shyam, Pranav; Sastry, Girish; Askell, Amanda; Agarwal, Sandhini; Herbert-Voss, Ariel; Krueger, Gretchen; Henighan, Tom; Child, Rewon; Ramesh, Aditya; Ziegler, Daniel M.; Wu, Jeffrey; Winter, Clemens; Hesse, Christopher; Chen, Mark; Sigler, Eric; Litwin, Mateusz; Gray, Scott; Chess, Benjamin; Clark, Jack; Berner, Christopher; McCandlish, Sam; Radford, Alec; Sutskever, Ilya; Amodei, Dario (July 22, 2020). "Language Models are Few-Shot Learners". arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

---