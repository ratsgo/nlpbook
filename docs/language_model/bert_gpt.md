---
layout: default
title: BERT & GPT
parent: Language Model
nav_order: 3
---

# BERT와 GPT
{: .no_toc }

이 글에서는 트랜스포머 아키텍처를 기본 뼈대로 하는 BERT 모델을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## BERT와 GPT 비교

BERT(Bidirectional Encoder Representations from Transformers)는 트랜스포머에서 인코더만 사용합니다. GPT(Generative Pre-trained Transformer(는 트랜스포머에서 디코더만 사용합니다. BERT는 문장의 의미를 추출하는 데 강점을 지니며 GPT는 문장 생성에 장점이 있습니다.

## **그림1** 트랜스포머 전체 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/F0qY4ny.png" width="500px" title="source: imgur.com" />

## **그림1** GPT 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Q7IS78n.png" width="500px" title="source: imgur.com" />

## **그림2** GPT (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/oPXpfWk.jpg" width="300px" title="source: imgur.com" />

## **그림2** GPT (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/mnzuvVg.jpg" width="500px" title="source: imgur.com" />

## **그림3** GPT (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1R0Cbbk.jpg" width="300px" title="source: imgur.com" />

## **그림3** GPT (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3h5Y3TX.jpg" width="500px" title="source: imgur.com" />

## **그림4** GPT (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/wXGfHBQ.jpg" width="300px" title="source: imgur.com" />

## **그림4** GPT (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/79lXt7h.jpg" width="500px" title="source: imgur.com" />

## **그림1** BERT 구조
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ekmm63h.png" width="500px" title="source: imgur.com" />

## **그림5** BERT
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pdBIXTT.jpg" width="300px" title="source: imgur.com" />

## **그림5** BERT
{: .no_toc .text-delta }
<img src="https://i.imgur.com/79lXt7h.jpg" width="500px" title="source: imgur.com" />


## **그림5** GPT vs BERT
{: .no_toc .text-delta }
<img src="https://i.imgur.com/S0equuG.png" width="300px" title="source: imgur.com" />


++ 단어의 출력 확률 업데이트 그림 보여주기

---

## BERT의 발전


## **그림1** Permutation Language Model(XLNet)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/iWGjfFR.png" width="250px" title="source: imgur.com" />
<img src="https://i.imgur.com/fJoFcNT.png" width="250px" title="source: imgur.com" />

## **그림1** 좀 더 긴 문맥을 살피기 (Transformer-XL)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Cks0i1R.png" width="350px" title="source: imgur.com" />

## **그림1** 학습 태스크 개선 (Electra)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/n93wqBt.png" width="350px" title="source: imgur.com" />

## **그림1** 다양한 소스의 데이터 사용하기 (XLM, Multimodal Transformer)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5S8Ueyj.png" width="550px" title="source: imgur.com" />
<img src="https://i.imgur.com/uFav742.png" width="550px" title="source: imgur.com" />

## **그림1** 계산량 줄이기 (Power-BERT)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/B2NG8JW.png" width="350px" title="source: imgur.com" />
<img src="https://i.imgur.com/pMnVqOq.png" width="350px" title="source: imgur.com" />

## **그림1** 모델 크기 줄이기 (Distillation, Quantization, Pruning, Weight Sharing)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/l5wCe0v.png" width="300px" title="source: imgur.com" />
<img src="https://i.imgur.com/D1yotJR.png" width="300px" title="source: imgur.com" />
<img src="https://i.imgur.com/mPvOIyn.png" width="300px" title="source: imgur.com" />
<img src="https://i.imgur.com/j2STiRI.png" width="300px" title="source: imgur.com" />



---

## GPT의 발전

BERT 쪽 아키텍처에 비해 그 변화량이 적은 편입니다. 하지만 모델 사이즈를 대폭 키워 성능이 큰 폭으로 개선되었습니다.


---


## 참고 문헌

- []()


---
