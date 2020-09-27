---
layout: default
title: Transformers
parent: Language Model
nav_order: 2
has_children: true
has_toc: false
---

# 트랜스포머(Transformer)
{: .no_toc }

트랜스포머(transformer)는 2017년 구글이 제안한 시퀀스-투-시퀀스(sequence-to-sequence) 모델입니다. 최근 자연어 처리에서는 BERT나 GPT 같은 트랜스포머 기반 언어모델이 각광받고 있습니다. 그 성능이 좋기 때문인데요. 왜 성능이 좋은지, 핵심 동작 원리는 무엇인지 이 글에서 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 시퀀스-투-시퀀스

트랜스포머(Transformer)란 기계 번역(machine translation) 등 시퀀스-투-시퀀스(sequence-to-sequence) 과제를 수행하기 위한 모델입니다. 여기에서 시퀀스란 단어(word) 같은 무언가(something)의 나열을 의미하는데요. '시퀀스-투-시퀀스'는 특정 속성을 지닌 시퀀스를 다른 속성의 시퀀스로 변환하는 작업을 가리킵니다.

기계 번역을 예시로 '시퀀스-투-시퀀스'가 어떤 태스크인지 알아봅시다. 기계 번역이란 어떤 언어(소스 언어, source language)의 단어 시퀀스를 다른 언어(대상 언어, target language)의 단어 시퀀스로 변환하는 과제입니다. 그림1과 그림2에서는 소스 언어와 번역 대상 문장 각각에 형태소 분석을 실시한 결과를 나타내고 있습니다.

## **그림1** 소스 언어의 문장과 그 시퀀스
{: .no_toc .text-delta }
```
어제 카페 갔었어 거기 사람 많더라 > 어제, 카페, 갔었어, 거기, 사람, 많더라
```

## **그림2** 대상 언어의 문장과 그 시퀀스
{: .no_toc .text-delta }
```
i went to the cafe there were many people there > i, went, to, the, cafe, there, were, many, people, there
```

그림3은 기계 번역에서의 '시퀀스-투-시퀀스' 결과를 예시로 나타낸 것입니다. 자세히 살펴보면 소스의 시퀀스 길이(단어 6개)와 대상 시퀀스의 길이(10개)가 다르다는 점을 알 수 있습니다. 이처럼 '시퀀스-투-시퀀스' 태스크는 소스와 대상의 길이가 달라도 해당 과제를 수행하는 데 문제가 없어야 합니다.

## **그림3** 기계 번역에서의 '시퀀스-투-시퀀스'
{: .no_toc .text-delta }
```
어제, 카페, 갔었어, 거기, 사람, 많더라 > i, went, to, the, cafe, there, were, many, people, there
```

---

## 인코더와 디코더

트랜스포머는 '시퀀스-투-시퀀스' 과제 수행에 특화된 모델입니다. 임의의 시퀀스를 해당 시퀀스와 속성이 다른 시퀀스로 변환하는 작업이라면 꼭 기계 번역이 아니더라도 수행이 가능합니다. 예컨대 필리핀 앞바다의 한 달치 기온 데이터를 가지고 앞으로 1주일간 태풍이 발생할지 여부를 맞추는 과제(기온의 시퀀스 > 태풍 발생 여부의 시퀀스) 역시 트랜스포머가 할 수 있는 일입니다.

'시퀀스-투-시퀀스' 과제를 수행하는 모델은 대개 인코더(encoder)와 디코더(decoder) 두 개 파트로 구성되어 있습니다(그림4). 인코더는 소스 시퀀스의 정보를 압축해 디코더로 보내주는 역할을 담당합니다. 인코더가 소스 시퀀스 정보를 압축하는 과정을 인코딩(encoding)이라고 합니다.

## **그림4** 인코더, 디코더
{: .no_toc .text-delta }
<img src="https://i.imgur.com/l0RJkOv.png" width="400px" title="source: imgur.com" />

디코더는 인코더가 보내준 소스 시퀀스 정보를 받아서 타겟 시퀀스를 생성합니다. 디코더가 타겟 시퀀스를 생성하는 과정을 디코딩(decoding)이라고 합니다. 기계번역에서는 인코더가 한국어 문장을 압축해 디코더에 보내고, 디코더는 이를 받아 영어로 번역합니다.

트랜스포머 역시 인코더와 디코더 구조를 따르고 있습니다. 그림5와 같습니다. 그림5 왼편이 인코더이고 오른편이 디코더입니다. 인코더의 입력(그림5에서 `Inputs`)은 소스 시퀀스입니다. 디코더의 입력(그림5에서 `Outputs`)은 타겟 시퀀스의 일부입니다. 

## **그림5** Transformer
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Rk5wkBQ.png" width="600px" title="source: imgur.com" />

---

## 모델 학습과 인퍼런스

그림3을 예시로 트랜스포머가 어떻게 학습되는지 살펴보겠습니다. 예컨대 이번 학습은 `i`를 맞춰야 하는 차례라고 가정해 봅시다. 그림6과 같습니다. 이 경우 인코더 입력은 `어제, 카페, 갔었어, 거기, 사람, 많더라` 같이 소스 시퀀스 전체가 되고 디코더 입력은 `<s>`가 됩니다. 여기에서 `<s>`는 타겟 시퀀스의 시작을 뜻하는 스페셜 토큰입니다. 인코더는 소스 시퀀스를 압축해 디코더로 보내고, 디코더는 인코더에서 보내온 정보와 현재 디코더 입력을 모두 감안해 다음 토큰을 맞춥니다. 

## **그림6** 인코더-디코더 입력 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Eboj504.jpg" width="400px" title="source: imgur.com" />

트랜스포머의 최종 출력, 즉 디코더 출력(그림5에서 `Output Probabilities`)은 타겟 언어의 어휘 수만큼의 차원을 갖는 확률 벡터(random vector)가 됩니다. 확률 벡터란 요소(element) 값이 모두 확률인 벡터를 의미합니다. 소스 언어의 어휘가 총 3만개라고 가정해봅시다. 그렇다면 디코더 출력은 3만 차원의 확률 벡터입니다. 이 벡터의 요소 값 3만개는 모두 확률이므로 3만개 요소값을 다 더하면 그 합은 1이 됩니다. 

**트랜스포머의 학습(train)은 인코더와 디코더 입력이 주어졌을 때 정답에 해당하는 단어의 확률 값을 높이는 방식으로 수행**됩니다. 이와 관련해 그림7을 봅시다. 모델은 이번 시점의 정답인 `i`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다.

## **그림7** 트랜스포머의 학습 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/7BLqgT4.jpg" width="500px" title="source: imgur.com" />

이번에는 타겟 시퀀스 가운데 `went`를 맞출 차례입니다. 그림8과 같습니다. 인코더 입력은 소스 시퀀스 전체, 디코더 입력은 `<s> i`가 됩니다. 여기에서 특이한 점이 하나 있습니다. 학습 중의 디코더 입력과 학습을 마친 후 모델을 실제 기계 번역에 사용할 때(인퍼런스)의 디코더 입력이 다르다는 점입니다.

학습 과정에서는 디코더 입력에 맞춰야할 단어(`went`) 이전의 타겟 시퀀스 전체(`<s> i`), 즉 정답 타겟 시퀀스를 넣어줍니다. 하지만 학습 종료 후 인퍼런스 때는 현재 디코더 입력에 직전 디코딩 결과를 사용합니다. 예를 들어 모델 학습이 약간 잘못 되어 인퍼런스 때 그림6과 같은 인코더, 디코더 입력의 결과로 `i` 대신 `you`라는 단어가 나왔다고 가정해 봅시다. 이 경우 디코더 입력은 `<s> you`가 됩니다.

## **그림8** 인코더-디코더 입력 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/C8d8jB7.jpg" width="400px" title="source: imgur.com" />

학습 과정 중 그림8과 같은 상황에서 인코더, 디코더 입력 모델은 이번 시점의 정답인 `went`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다. 그림9와 같습니다.

## **그림9** 트랜스포머의 학습 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qq9fQdD.jpg" width="500px" title="source: imgur.com" />

이번에는 타겟 시퀀스 가운데 `to`를 맞출 차례입니다. 그림10과 같습니다. 인코더 입력은 소스 시퀀스 전체입니다. 학습 과정 중 디코더 입력은 정답인 `<s> i went`, 인퍼런스할 때 디코더 입력은 직전 디코딩 히스토리가 됩니다. 

## **그림10** 인코더-디코더 입력 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Oh4zPK1.jpg" width="400px" title="source: imgur.com" />

학습 과정 중 그림10과 같은 상황에서 인코더, 디코더 입력 모델은 이번 시점의 정답인 `to`에 해당하는 확률은 높이고 나머지 단어의 확률은 낮아지도록, 모델 전체를 업데이트합니다. 그림11과 같습니다.

## **그림11** 트랜스포머의 학습 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/K6DQPwR.jpg" width="500px" title="source: imgur.com" />

이와 같은 방식으로 말뭉치 전체를 반복 학습합니다. 이렇게 학습을 마친 모델은 한국어-영어 기계 번역을 성공적으로 수행할 수 있게 됩니다.

---

## 트랜스포머 블록

그림12는 트랜스포머의 인코더 가운데 반복되는 요소를 떼어내 다시 그린 것입니다. 그림12 같은 구조를 블록(block) 혹은 레이어(layer)라고 부릅니다. 트랜스포머의 인코더는 이같은 블록을 6\~24개를 쌓아서 구성합니다. 

## **그림12** 트랜스포머 인코더 블록
{: .no_toc .text-delta }
<img src="https://i.imgur.com/NSmVlit.png" width="150px" title="source: imgur.com" />

그림12에서 확인할 수 있듯 인코더 블록은 다음과 같은 세 가지 요소로 구성돼 있습니다. 각 챕터에서 자세한 내용을 참고하시면 좋을 것 같습니다.

- **Multi-Head Attention** : [3-2-1장 Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention)
- **FeedForward** : [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)
- **Add & Norm** : [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)

디코더 쪽 블록의 구조도 그림9의 인코더 블록과 본질적으로는 다르지 않습니다. 다만 마스크(mask)를 적용한 멀티헤드 어텐션이 인코더 쪽과 다른 점이고, 인코더 쪽에서 넘어온 정보와 디코더 입력에 멀티헤드 어텐션을 수행하는 모듈이 추가됐습니다.

## **그림13** 트랜스포머 디코더 블록
{: .no_toc .text-delta }
<img src="https://i.imgur.com/jmNALxv.png" width="150px" title="source: imgur.com" />

디코더 블록의 구성 요소도 각 챕터에서 자세한 내용을 참고하시면 좋을 것 같습니다.

- **Masked Multi-Head Attention** : [3-2-1장 Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention)
- **Multi-Head Attention** : [3-2-1장 Self Attention](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention)
- **FeedForward** : [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)
- **Add & Norm** : [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)

멀티헤드 어텐션은 셀프 어텐션(self attention)이라고도 불립니다. 트랜스포머 경쟁력의 원천은 셀프 어텐션에 있다고 많은 분들이 언급하고 있는데요. 셀프 어텐션 파트만 별도로 이 장에서 좀 더 살펴보겠습니다. 


---

## 셀프 어텐션

어텐션(attention)은 시퀀스 입력에 수행하는 기계학습 방법의 일종인데요. **어텐션은 시퀀스 요소들 가운데 태스크 수행에 중요한 요소에 집중하고 그렇지 않은 요소는 무시해 태스크 수행 성능을 끌어 올립니다.** 어텐션은 기계 번역 과제에 처음 도입됐습니다.

기계 번역에 어텐션을 도입한다면 타겟 언어를 디코딩할 때 소스 언어의 단어들(시퀀스) 가운데 디코딩에 도움되는 단어들 위주로 취사 선택해 번역 품질을 끌어 올리게 됩니다. 요컨대 어텐션은 타겟 시퀀스(target sequence) 디코딩시 소스 시퀀스(source sequence)를 참조해 소스 시퀀스 가운데 중요한 요소들만 추립니다.

셀프 어텐션이란 말 그대로 자기 자신에 수행하는 어텐션 기법입니다. 입력 시퀀스 가운데 태스크 수행에 의미 있는 녀석들 위주로 정보를 추출한다는 것이죠.

이렇게만 설명한다면 너무 알쏭달쏭하니 자연어 처리에서 자주 쓰이는 리커런트 뉴럴네트워크(Recurrenct Neural Network, RNN), 컨볼루션 뉴럴네트워크(Convolutional Neural Network, CNN) 등과 비교를 통해 셀프 어텐션이 대체 어떤 점을 목표로 하는지 살펴보도록 하겠습니다.

### CNN과 비교

CNN은 시퀀스의 지역적인 특징을 잡아내는 데 유리한 모델입니다. 자연어는 기본적으로 시퀀스(단어 혹은 형태소의 나열)이고 특정 단어 기준 주변 문맥이 의미 형성에 중요한 역할을 하기 때문에 CNN은 자연어 처리에 널리 쓰이고 있습니다. 

그림14는 CNN이 문장을 어떻게 인코딩하는지 도식적으로 나타낸 것입니다. 컨볼루션 필터(그림14에서 붉은색 네모칸)이 단어를 하나씩 슬라이딩하면서 차례대로 읽어들이는 걸 알 수 있습니다.

하지만 CNN은 컨볼루션 필터 크기를 넘어서는 문맥은 읽어내기 어렵다는 단점이 있습니다. 예컨대 필터 크기가 3(3개 단어를 처리)이라면 5칸 이상 떨어져 있는 단어 사이의 의미는 캐치하기 어렵다는 것입니다.

## **그림14** 컨볼루션 뉴럴 네트워크(Convolutional Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3pn5n0C.png" width="500px" title="source: imgur.com" />
<br>
<img src="https://i.imgur.com/tk2s2eR.png" width="500px" title="source: imgur.com" />
<br>
<img src="https://i.imgur.com/QZ7QV6v.png" width="500px" title="source: imgur.com" />


### RNN과 비교

RNN 역시 시퀀스 정보를 압축하는 데 강점이 있는 구조입니다. 소스 언어 시퀀스, `어제, 카페, 갔었어, 거기, 사람, 많더라`를 인코딩해야 한다고 가정해 봅시다. 그렇다면 RNN은 그림15와 같이 소스 시퀀스를 순차적으로 처리하게 됩니다.

하지만 RNN은 시퀀스 길이가 길어질 수록 정보 압축에 문제가 생기게 됩니다. 오래 전에 입력된 단어는 잊어버리거나, 특정 단어 정보를 과도하게 반영해 전체 정보를 왜곡하는 경우가 자주 생긴다는 것입니다.

## **그림15** 리커런트 뉴럴 네트워크(Recurrenct Neural Network)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5eAs55K.png" width="500px" title="source: imgur.com" />

기계 번역시 소스 언어의 문장을 인코딩할 때 RNN을 사용했다고 칩시다. 이 경우 인코더가 디코더로 넘기는 정보는 소스 시퀀스의 마지막인 `많더라`라는 단어의 의미가 많이 반영될 수밖에 없습니다. RNN은 입력 정보를 순차적으로 처리하기 때문입니다.

### 어텐션과 비교

그림16을 보면 `cafe`에 대응하는 소스 언어의 단어는 `카페`이고 이는 소스 시퀀스의 초반부에 등장한 상황입니다. `cafe`라는 단어를 디코딩해야 하는 경우 `카페`를 반드시 참조해야 하는데요. 단순 RNN을 사용하면 워낙 오래 전에 입력된 단어라 모델이 잊었을 가능성이 높고 이 때문에 번역 품질이 낮아질 수 있습니다.

## **그림16** 기존 어텐션(Sequence-to-Sequence Attention)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kRjEkPm.png" width="500px" title="source: imgur.com" />

어텐션은 RNN의 이러한 문제점을 해결하기 위해 제안된 기법입니다. 어텐션은 대상 언어 디코딩시 소스 시퀀스 전체에서 어떤 요소에 주목해야 할지 알려주기 때문에 `카페`가 시퀀스 초반에 등장하더라도, 소스 시퀀스의 길이가 길어지더라도 번역 품질 하락을 막을 수 있게 됩니다. 그림16의 예시에서는 어텐션 기법으로 주목되는 단어에 좀 더 짙고 굵은 실선을 그려 놓았습니다.

### 특징 및 장점

셀프 어텐션은 자기 자신에 수행하는 어텐션입니다. 그림17을 봅시다. 입력 시퀀스가 `어제, 카페, 갔었어, 거기, 사람, 많더라`일 때 `거기`라는 단어가 어떤 의미를 가지는지 계산하는 상황입니다. 

잘 학습된 셀프 어텐션 모델이라면 `거기`에 대응하는 장소는 `카페`라는 사실을 알아챌 수 있을 것입니다. 뿐만 아니라 `거기`는 `갔었어`와도 연관되어 있음을 확인할 수 있습니다. 트랜스포머 인코더 블록 내부에서는 이처럼 `거기`라는 단어를 인코딩할 때 `카페`, `갔었어`라는 단어의 의미를 강조해 반영합니다.

## **그림17** 셀프 어텐션(Self Attention) (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/l7JogVh.png" width="500px" title="source: imgur.com" />

그림18은 입력 시퀀스가 입력 시퀀스가 `어제, 카페, 갔었어, 거기, 사람, 많더라`일 때 `카페`라는 단어가 어떤 의미를 가지는지 계산하는 상황입니다. 트랜스포머 인코더 블록은 `카페`라는 단어를 인코딩할 때 `거기`, `갔었어`라는 단어의 의미를 다른 단어들보다 더 강하게 반영합니다.

## **그림18** 셀프 어텐션(Self Attention) (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YUcm8xS.png" width="500px" title="source: imgur.com" />

셀프 어텐션 수행 대상은 입력 시퀀스 전체입니다. `거기`(그림17)와 `카페`(그림18)만을 예로 들었지만 실제로는 `어제`-전체 입력 시퀀스, `갔었어`-전체 입력 시퀀스, `사람`-전체 입력 시퀀스, `많더라`-전체 입력 시퀀스 모두 어텐션 계산을 합니다. 

개별 단어와 전체 입력 시퀀스를 대상으로 어텐션 계산을 수행해 문맥 전체를 고려하기 때문에 지역적인 문맥만 보는 CNN 대비 강점이 있습니다. 아울러 모든 경우의 수를 고려(단어들 서로가 서로를 1대 1로 바라보게 함)하기 때문에 시퀀스 길이가 길어지더라도 정보를 잊거나 왜곡할 염려가 없습니다. 이는 RNN의 단점을 극복한 지점입니다.

한편 어텐션과 셀프 어텐션이 다른 지점이 있다면 기존 어텐션은 RNN 구조를 보정하기 위해 제안된 아키텍처라면, 셀프 어텐션은 RNN을 제거하고 아예 어텐션으로만 인코더 디코더 구조를 만들었다는 점입니다.
{: .fs-4 .ls-1 .code-example }

### 계산 예시

셀프 어텐션은 쿼리(query), 키(key), 밸류(value) 세 가지 요소가 서로 영향을 주고 받는 구조입니다. 트랜스포머 블록에는 문장 내 각 단어가 벡터(vector) 형태로 입력되는데요. 여기서 벡터란 숫자의 나열 정도로 일단 이해해 두시면 좋을 것 같습니다.

각 단어 벡터는 블록 내에서 어떤 계산 과정을 거쳐 쿼리, 키, 밸류 세 가지로 변환됩니다. 만일 트랜스포머 블록에 입력되는 문장이 그림19처럼 여섯 개 단어로 구성돼 있다면 이 블록의 셀프 어텐션 계산 대상은 쿼리 벡터 6개, 키 벡터 6개, 밸류 백터 6개 등 모두 18개가 됩니다.

그림19는 그림18을 좀 더 세부적으로 그린 것입니다. 셀프 어텐션은 쿼리 단어 각각에 대해 모든 키 단어와 얼마나 유기적인 관계를 맺고 있는지 그 합이 1인 확률값으로 나타냅니다. 그림19를 보면 `카페`라는 쿼리 단어와 가장 관련이 높은 키 단어는 `거기`라는 점(0.4)을 확인할 수 있습니다.

## **그림19** 셀프 어텐션 계산 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/QgIcjoJ.jpg" width="200px" title="source: imgur.com" />

셀프 어텐션 모듈은 그림19와 같은 결과에 밸류 벡터들을 가중합하는 방식으로 계산을 마무리합니다. 수식1과 같습니다. 새롭게 만들어지는 카페 벡터($\mathbf{Z}_{\text{카페}}$)는 문장에 속한 모든 단어 쌍 사이의 관계가 녹아 있습니다.

## **수식1** 셀프 어텐션 계산 예시
{: .no_toc .text-delta }

$$
\begin{align*}
\mathbf{Z}_{\text{카페}} =0.1 \times \mathbf{V}_{\text{어제}} + 0.1 \times \mathbf{V}_{\text{카페}} + 0.1 \times \mathbf{V}_{\text{갔었어}} \\ 
+ 0.4 \times \mathbf{V}_{\text{거기}} + 0.2 \times \mathbf{V}_{\text{사람}} + 0.1 \times \mathbf{V}_{\text{많더라}}
\end{align*}
$$

지금은 `카페`에 대해서만 계산 예시를 들었는데요. 이같은 방식으로 나머지 입력 시퀀스 전체(`어제`, `갔었어`, `거기`, `사람`, `많더라`)에 대해서도 셀프 어텐션을 각각 수행합니다. 트랜스포머 모델 전체적으로 보면 이같은 셀프 어텐션을 블록(레이어) 수만큼 반복하는 셈이 됩니다.

---


## 참고 문헌

- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)


---
