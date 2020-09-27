---
layout: default
title: BERT & GPT
parent: Language Model
nav_order: 3
---

# BERT와 GPT
{: .no_toc }

이 글에서는 트랜스포머 아키텍처를 기본 뼈대로 하는 BERT와 GPT 모델의 공통점과 차이점을 중심으로 살펴봅니다. 마지막으로는 트랜스포머 계열 언어모델의 최근 경향도 설명합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## BERT와 GPT

GPT(Generative Pre-trained Transformer)는 언어모델(Language Model)입니다. 이전 단어들이 주어졌을 때 다음 단어가 무엇인지 맞추는 과정에서 프리트레인(pretrain)합니다. 문장 시작부터 순차적으로 계산한다는 점에서 일방향(unidirectional)입니다.

BERT(Bidirectional Encoder Representations from Transformers)는 마스크 언어모델(Masked Language Model)입니다. 문장 중간에 빈칸을 만들고 해당 빈칸에 어떤 단어가 적절할지 맞추는 과정에서 프리트레인합니다. 빈칸 앞뒤 문맥을 모두 살필 수 있다는 점에서 양방향(bidirectional) 성격을 가집니다.

이 때문에 GPT는 문장 생성에, BERT는 문장의 의미를 추출하는 데 강점을 지닌 것으로 알려져 있습니다. 그림1은 GPT와 BERT의 프리트레인 방식을 도식적으로 나타낸 것입니다.


## **그림1** GPT vs BERT
{: .no_toc .text-delta }
<img src="https://i.imgur.com/S0equuG.png" width="300px" title="source: imgur.com" />


한편 BERT는 트랜스포머에서 인코더(encoder), GPT는 트랜스포머에서 디코더(decoder)만 취해 사용한다는 점 역시 다른 점입니다. 구조상 차이에 대해서는 각 모델 챕터에서 설명하겠습니다.


---

## GPT

그림2는 GPT 구조를 나타낸 것입니다. 트랜스포머에서 인코더를 제외하고 디코더만 사용합니다. 그림2 오른쪽 디코더 블록을 자세히 보면 인코더 쪽에서 보내오는 정보를 받는 모듈(Multi-Head Attention) 역시 제거돼 있음을 확인할 수 있습니다.

## **그림2** GPT 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Q7IS78n.png" width="300px" title="source: imgur.com" />

그림3은 GPT의 셀프 어텐션(Masked Multi-Head Attention)을 도식적으로 나타낸 것입니다. 입력 단어 시퀀스가 `어제 카페 갔었어 거기 사람 많더라`이고 이번이 `카페`를 맞춰야 하는 상황이라고 가정해 보겠습니다. 이 경우 GPT는 정답 단어 `카페`를 맞출 때 `어제`라는 단어만 참고할 수 있습니다. 

따라서 정답 단어 이후의 모든 단어(`카페`\~`많더라`)를 볼 수 없도록 처리해 줍니다. 구체적으로는 밸류 벡터들을 가중합할 때 참고 불가인 단어들에 곱해지는 스코어가 0이 되도록 합니다. 이와 관련해 자세한 내용은 [3-2-1장 Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention)을 참고하시기 바랍니다.

## **그림3** GPT의 셀프 어텐션 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/oPXpfWk.jpg" width="300px" title="source: imgur.com" />

`어제`라는 단어에 대응하는 GPT 마지막 레이어의 출력 결과에 어떤 계산을 수행해 학습 대상 언어의 어휘 수만큼의 확률 벡터가 되도록 합니다. 이번 시점의 정답인 `카페`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다. 그림4와 같습니다.

## **그림4** GPT의 학습 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/mnzuvVg.jpg" width="500px" title="source: imgur.com" />

이번에는 `갔었어`를 맞춰야 하는 상황입니다. 이 경우 GPT는 정답 단어 `갔었어`를 맞출 때 `어제`와 `카페`라는 단어를 참고할 수 있습니다. 

## **그림5** GPT의 셀프 어텐션 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1R0Cbbk.jpg" width="300px" title="source: imgur.com" />

`카페`라는 단어에 대응하는 GPT 마지막 레이어의 출력 결과에 어떤 계산을 수행해 학습 대상 언어의 어휘 수만큼의 확률 벡터가 되도록 합니다. 이번 시점의 정답인 `갔었어`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다. 그림6과 같습니다.

## **그림6** GPT의 학습 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3h5Y3TX.jpg" width="500px" title="source: imgur.com" />

`거기`를 맞춰야 하는 상황이라면 모델은 `어제`, `카페`, `갔었어` 세 단어를 참고할 수 있습니다(그림7).

## **그림7** GPT의 셀프 어텐션 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/wXGfHBQ.jpg" width="300px" title="source: imgur.com" />

`갔었어`라는 단어에 대응하는 GPT 마지막 레이어의 출력 결과에 어떤 계산을 수행해 학습 대상 언어의 어휘 수만큼의 확률 벡터가 되도록 합니다. 이번 시점의 정답인 `거기`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다(그림8).

## **그림8** GPT의 학습 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/79lXt7h.jpg" width="500px" title="source: imgur.com" />


---

## BERT

그림9는 BERT 구조를 나타낸 것입니다. 트랜스포머에서 디코더를 제외하고 인코더만 사용합니다.

## **그림9** BERT 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ekmm63h.png" width="300px" title="source: imgur.com" />

그림10은 BERT의 셀프 어텐션(Multi-Head Attention)을 도식적으로 나타낸 것입니다. 입력 단어 시퀀스가 `어제 카페 갔었어 [MASK] 사람 많더라`라고 가정해 보겠습니다. 그림10에서 확인할 수 있듯 BERT는 마스크 토큰 앞뒤 문맥을 모두 참고할 수 있습니다. 앞뒤 정보를 준다고 해서 정답을 미리 알려주는 것이 아니기 때문입니다.

## **그림10** BERT의 셀프 어텐션
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pdBIXTT.jpg" width="300px" title="source: imgur.com" />

`MASK`라는 단어에 대응하는 BERT 마지막 레이어의 출력 결과에 어떤 계산을 수행해 학습 대상 언어의 어휘 수만큼의 확률 벡터가 되도록 합니다. 빈칸의 정답인 `거기`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다. 그림11과 같습니다.

## **그림11** BERT의 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/79lXt7h.jpg" width="500px" title="source: imgur.com" />

---

## 최근 트렌드

트랜스포머 계열 언어모델의 최근 트렌드를 살펴보겠습니다. 우선 모델 크기 증가 추세가 눈에 띕니다. GPT3가 대표적입니다. 표1을 보면 파라메터 수 기준 GPT3는 GPT1 대비 1400배, GPT2 대비 117배 증가했습니다. OpenAI에 따르면 모델 크기 증가는 언어모델 품질은 물론 각종 다운스트림 태스크의 성능 개선에 큰 도움이 된다고 합니다.

## **표1** GPT 모델 크기 비교
{: .no_toc .text-delta }

|모델명|사이즈|
|---|---|
|GPT1|0.125B(=125M)
|GPT2|1.5B|
|GPT3|175B|

이와 별개로 모델 성능을 최대한 유지하면서 크기를 줄이려는 시도도 계속되고 있습니다(그림12). 디스틸레이션(Distillation), 퀀타이제이션(Quantization), 프루닝(Pruning), 파라메터 공유(Weight Sharing) 등이 바로 그것입니다.

## **그림12** 모델 크기 줄이기 (Distillation, Quantization, Pruning, Weight Sharing)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/l5wCe0v.png" width="300px" title="source: imgur.com" />

트랜스포머 모델은 입력 시퀀스 사이의 관계를 모두 고려하기 때문에 계산량이 많은 편입니다. 계산량을 줄이거나(그림13) 계산 대상 시퀀스 범위를 넓히려는(그림14) 연구 역시 활발한 분야입니다. 

## **그림13** 계산량 줄이기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/B2NG8JW.png" width="350px" title="source: imgur.com" />
<img src="https://i.imgur.com/pMnVqOq.png" width="350px" title="source: imgur.com" />

## **그림14** 좀 더 긴 문맥을 살피기 (Transformer-XL)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Cks0i1R.png" width="350px" title="source: imgur.com" />

트랜스포머 아키텍처에서 벗어나 새로운 구조의 언어모델도 고안되고 있습니다. 그림15는 GAN 방식을 차용한 언어모델, 일렉트라(electra)를 도식적으로 나타낸 그림입니다.

## **그림15** 모델 구조 개선 (Electra)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/n93wqBt.png" width="350px" title="source: imgur.com" />

트랜스포머 입력 역시 다양하게 실험되고 있습니다. 이종 언어, 이종 데이터(텍스트, 이미지, 음성 등)를 입력으로 한 모델들이 바로 그것입니다. 그림16과 같습니다.

## **그림16** 다양한 소스의 데이터 사용하기 (XLM, Multimodal Transformer)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5S8Ueyj.png" width="550px" title="source: imgur.com" />
<img src="https://i.imgur.com/uFav742.png" width="550px" title="source: imgur.com" />

---


## 참고 문헌

- []()


---
