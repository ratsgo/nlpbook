---
layout: default
title: Vector Semantics
parent: Language Model
nav_order: 1
---

# Vector Semantics
{: .no_toc }

컴퓨터가 자연어를 연산 가능하게 하려면 자연어를 숫자로 바꿔줘야 합니다. 숫자에 자연어의 의미를 함축시키려면 어떤 정보를 반영해야 할까요. 크게 Bag of words, Language Model 등의 가정이 있습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 언어 모델의 정의

정의1 : 좁게는 단어 시퀀스가 나타날 결합 확률 P(w1, w2, .. wn)
정의2 : 좁게는 컨텍스트가 나타났을 때 다음 단어가 나타날 조건부 확률 P(wn\|w1,w2,...wn-1)

==> 사실 조건부 확률 정의를 다시 쓰면 정의1과 정의2가 사실상 동일

정의3 : 제일 넓게는 컨텍스트가 나타났을 때 타겟 단어가 나타날 조건부 확률 P(wt\|context).. 여기에서 컨텍스트 단어가 하나라면 skip-gram, 컨텍스트 단어가 여러개이고 타겟 단어가 마스킹이라면 Masked Language Model


---

## 빈도를 세기

전통적인 통계 기반의 언어 모델
n-gram... 하지만 안나오는 패턴 다수


---

## 주변에 등장하는지 여부 맞추기

Skip-Gram Model
조건부 확률 P(wt\|context) 학습, 뉴럴네트워크


## **그림1** 분포 정보 배우기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DlhHAJe.png" width="400px" title="source: imgur.com" />

## **그림1** 분포 정보 배우기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RTaDutG.jpg" width="180px" title="source: imgur.com" />

++ 벡터 의미
++ 공간에서 가까워지는 의미

---

## 다음 단어 맞추기

autoregressive


## **그림2** left-to-right
{: .no_toc .text-delta }
<img src="https://i.imgur.com/4dv6TNZ.png" width="400px" title="source: imgur.com" />

## **그림2** right-to-left
{: .no_toc .text-delta }
<img src="https://i.imgur.com/VHB5dsR.png" width="400px" title="source: imgur.com" />



---

## 빈칸 맞추기

autoencoding
bidirectional
encoding에 특화


## **그림3** bidirectional
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YJjh69r.png" width="400px" title="source: imgur.com" />


## **그림3** kcbert
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nv9A8i2.png" width="400px" title="source: imgur.com" />


---

## 언어 모델의 유용성

언어 모델 자체의 쓰임새
Machine Translaon:
• P(high winds tonite) > P(large winds tonite) 
• Spell!Correcon
• The!office!is!about!fiIeen!minuets!from!my!house!
• P(about!fiIeen!minutes!from)!>!P(about!fiIeen!minuets!from)
• Speech!Recognion!
• P(I!saw!a!van)!>>!P(eyes!awe!of!an)!
• Summarizaon,!queson,answering,!etc.,!etc.!!


++ 언어 모델이 왜 잘되나?

하지만 더 중요한 것은 자연어의 풍부한 문맥을 학습 (심지어 지식도 간단한 계산도 외운다)
언어 모델이 자연어 의미를 포착할 수 있는 이유 : 주변 단어와 함께 나타나는지, 얼마나 자주 나타나는지, 어떤 순서로 나타나는지 등 포괄적인 정보가 모두 함축되어 있음

딥러닝은 모델 크기가 커질 수록 표현력이 풍부해지는 반면 다량의 데이터가 필요... 언어모델로 학습을 하면 레이블 없이도 다량의 학습데이터를 만들어낼 수 있다+적은 양의 다운스트림 데이터(레이블링)만으로도 높은 성능을 가진다, 이때문에 트랜스퍼 러닝 대상으로 언어모델이 각광받는 것

++ 프리트레인된 언어모델 자체로 훌륭한 임베딩/리프레젠테이션을 만들 수 있다
빈도 세기 말고는 모두 딥러닝 계열 모델이다, 딥러닝 계열 모델은 모델의 최종 혹은 중간 출력값으로 임베딩을 만들 수 있다.
 


---

## 참고문헌

- [한국어 임베딩](http://www.yes24.com/Product/Goods/78569687)
- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Language Modeling](http://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)
- [Language Model - Wikipedia](https://en.m.wikipedia.org/wiki/Language_model)

---

