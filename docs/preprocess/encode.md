---
layout: default
title: Tokenization Tutorial
parent: Preprocess
nav_order: 4
---

# 토큰화하기
{: .no_toc }

이 장에서는 문장을 토큰화하고 해당 토큰들을 모델의 입력으로 만드는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

1. TOC
{:toc}

---

## 실습 환경 만들기


이번 실습은 웹 브라우저에서 다음 주소에 접속하면 코랩 환경에서 수행할 수 있습니다. 이전 실습과 마찬가지로 코랩에서 '내 드라이브에 복사'와 '하드웨어 가속기 사용 안함(None)'으로 설정합니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/basic/tokenization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


---

## 의존성 패키지 설치하기

다음 코드를 실행해 의존성 있는 패키지를 설치합니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

---

## 구글 드라이브 연동하기

이번 실습에서는 [이전 실습](https://ratsgo.github.io/nlpbook/docs/preprocess/vocab/)에서 미리 구축해 놓은 어휘 집합(vocabulary)을 바탕으로 실습합니다. 이전 실습에서 어휘 집합을 구글 드라이브에 저장해 두었기 때문에 자신의 구글 드라이브를 코랩 노트북과 연결해야 합니다. 코드2를 실행하면 됩니다.

## **코드2** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```


---


## GPT 입력값 만들기

GPT 입력값을 만들려면 토크나이저부터 준비해야 합니다. 코드3을 수행하면 GPT 모델이 사용하는 토크나이저를 초기화할 수 있습니다. 먼저 자신의 구글 드라이브 경로(`/gdrive/My Drive/nlpbook/bbpe`)에는 [이전 실습](https://ratsgo.github.io/nlpbook/docs/preprocess/vocab)에서 만든 바이트 기준 BPE 어휘 집합(`vocab.json`)과 바이그램 쌍의 병합 우선순위(`merge.txt`)가 있어야 합니다.

## **코드3** GPT 토크나이저 선언
{: .no_toc .text-delta } 
```python
from transformers import GPT2Tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained("/gdrive/My Drive/nlpbook/bbpe")
tokenizer_gpt.pad_token = "[PAD]"
```

다음 코드는 예시 문장 세 개를 바이트 수준 BPE 토크나이저로 토큰화합니다. 그 결과는 표1과 같습니다.

## **코드4** GPT 토크나이저로 토큰화하기
{: .no_toc .text-delta } 
```python
sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]
tokenized_sentences = [tokenizer_gpt.tokenize(sentence) for sentence in sentences]
```

표1을 보면 토큰들이 알 수 없는 문자열로 구성돼 있음을 확인할 수 있습니다. 그 이유는 앞에서도 설명했듯이 GPT 모델은 바이트 기준 BPE를 적용하기 때문입니다.

## **표1** GPT 토크나이저 토큰화 결과
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|토큰13|토큰14|토큰15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|ìķĦ|ĠëįĶë¹Ļ|..|Ġì§Ħì§ľ|Ġì§ľì¦ĿëĤĺ|ëĦ¤ìļĶ|Ġëª©ìĨĮë¦¬|||||||||
|문장2|íĿł|...|íı¬ìĬ¤íĦ°|ë³´ê³ł|Ġì´ĪëĶ©|ìĺģíĻĶ|ì¤Ħ|....|ìĺ¤ë²Ħ|ìĹ°ê¸°|ì¡°ì°¨|Ġê°Ģë³į|ì§Ģ|ĠìķĬ|êµ¬ëĤĺ|
|문장3|ë³Ħë£¨|Ġìĺ|Ģëĭ¤|..||||||||||||

코드4와 표1은 GPT 토크나이저의 토큰화 결과를 살짝 맛보려고 한 것이고, 실제 모델 입력값은 코드5로 만듭니다.

## **코드5** GPT 모델 입력 만들기
{: .no_toc .text-delta } 
```python
batch_inputs = tokenizer_gpt(
    sentences,
    padding="max_length", # 문장의 최대 길이에 맞춰 패딩
    max_length=12, # 문장의 토큰 기준 최대 길이
    truncation=True, # 문장 잘림 허용 옵션
)
```

코드5 실행 결과로 두 가지의 입력값이 만들어집니다. 하나는 `input_ids`입니다. `batch_inputs['input_ids']`를 코랩에서 실행해 그 결과를 출력해보면 표2와 같습니다. `input_ids`는 표1의 토큰화 결과를 가지고 각 토큰들을 인덱스(index)로 바꾼 것입니다. 어휘 집합(`vocab.json`)을 확인해 보면 각 어휘가 순서대로 나열된 확인할 수 있는데요. 이 순서가 바로 인덱스입니다. 이같이 각 토큰을 인덱스로 변환하는 과정을 **인덱싱(indexing)**이라고 합니다.

## **표2** GPT의 input_ids
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|334|2338|263|581|4055|464|3808|0|0|0|0|0|
|문장2|3693|336|2876|758|2883|356|806|422|9875|875|2960|7292|
|문장3|4957|451|3653|263|0|0|0|0|0|0|0|0|

표2를 자세히 보시면 모든 문장의 길이가 12로 맞춰진걸 볼 수 있습니다. 코드5에서 `max_length` 인자에 12를 넣었기 때문인데요. 이보다 짧은 문장1과 문장3은 뒤에 `[PAD]` 토큰에 해당하는 인덱스 `0`이 붙었습니다. `[PAD]` 토큰은 일종의 더미 토큰으로 길이를 맞춰주는 역할을 합니다. 문장2는 원래 토큰 길이가 15였는데 12로 줄었습니다. 문장 잘림을 허용하는 `truncation=True` 옵션 때문입니다.

코드5 실행 결과로 `attention_mask`도 만들어졌습니다. `attention_mask`는 일반 토큰이 자리한 곳(`1`)과 패딩 토큰이 자리한 곳(`0`)을 구분해 알려주는 장치입니다. `batch_inputs['input_ids']`를 입력해 그 결과를 출력해 보면 표3과 같습니다.

## **표3** GPT attention_mask
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|1|1|1|1|1|1|1|0|0|0|0|0|
|문장2|1|1|1|1|1|1|1|1|1|1|1|1|
|문장3|1|1|1|1|0|0|0|0|0|0|0|0|


---

## BERT 입력값 만들기

이번엔 BERT 모델의 입력값을 만들어보겠습니다. 코드6을 수행하면 BERT 모델이 사용하는 토크나이저를 초기화할 수 있습니다. 먼저 자신의 구글 드라이브 경로(`/gdrive/My Drive/nlpbook/wordpiece`)에는 BERT용 워드피스 어휘 집합(`vocab.txt`)이 있어야 합니다.

## **코드6** BERT 토크나이저 선언
{: .no_toc .text-delta } 
```python
from transformers import BertTokenizer
tokenizer_bert = BertTokenizer.from_pretrained("/gdrive/My Drive/nlpbook/wordpiece", do_lower_case=False)
```

다음 코드는 예시 문장 3개를 워드피스 토크나이저로 토큰화합니다. 그리고 그 결과는 표4와 같습니다. 토큰 일부에 있는 `##`은 해당 토큰이 어절(띄어쓰기 기준)의 시작이 아님을 나타냅니다. 예컨대 `##네요`는 이 토큰이 앞선 토큰 `짜증나`와 같은 어절에 위치하며 어절 내에서 연속되고 있음을 표시합니다.

## **코드7** BERT 토크나이저로 토큰화하기
{: .no_toc .text-delta } 
```python
sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]
tokenized_sentences = [tokenizer_bert.tokenize(sentence) for sentence in sentences]
```

## **표4** BERT 토크나이저 토큰화 결과
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|토큰13|토큰14|토큰15|토큰16|토큰17|토큰18|토큰19|토큰20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|아|더빙.|.|진짜|짜증나|##네요|목소리|
|문장2|흠|.|.|.|포스터|##보고|초딩|##영화|##줄|.|.|.|.|오버|##연기|##조차|가볍|##지|않|##구나|
|문장3|별루|였다|.|.|

코드5와 표4는 BERT 토크나이저의 토큰화 결과를 살짝 맛보기 위해 설명한 것인데요. 실제 모델 입력값은 코드8로 만듭니다.

## **코드8** BERT 모델 입력 만들기
{: .no_toc .text-delta } 
```python
batch_inputs = tokenizer_bert(
    sentences,
    padding="max_length",
    max_length=12,
    truncation=True,
)
```

코드8 실행 결과로 세 가지의 입력값이 만들어집니다. 하나는 GPT 모델과 마찬가지로 토큰 인덱스 시퀀스를 나타내는 `input_ids`입니다. `batch_inputs['input_ids']`를 입력하고 이를 출력해 보면 다음 표와 같습니다.

## **표5** BERT input_ids
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|2|622|2639|16|16|1993|3682|1990|3467|3|0|0|
|문장2|2|997|16|16|16|2596|2045|2809|1981|1225|16|3|
|문장3|2|3341|9157|16|16|3|0|0|0|0|0|0|

표5를 자세히 보시면 모든 문장 앞에 `2`, 끝에 `3`이 붙은 걸 확인할 수 있습니다. 이는 각각 `[CLS]`, `[SEP]`라는 토큰에 대응하는 인덱스인데요. BERT는 문장 시작과 끝에 이 두 개 토큰을 덧붙이는 특징이 있습니다. 그리고 `attention_mask`도 만들어집니다. BERT의 `attention_mask`는 GPT와 마찬가지로 일반 토큰이 자리한 곳(`1`)과 패딩 토큰이 자리한 곳(`0`)을 구분해 알려줍니다.


## **표7** BERT attention_mask
{: .no_toc .text-delta } 

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|문장1|1|1|1|1|1|1|1|1|1|1|0|0|
|문장2|1|1|1|1|1|1|1|1|1|1|1|1|
|문장3|1|1|1|1|1|1|0|0|0|0|0|0|

마지막으로 `token_type_ids`라는 입력값도 만들어집니다. 이는 **세그먼트(segment)**에 해당하는 것으로 모두 `0`입니다. 세그먼트 정보를 입력하는 건 BERT 모델의 특징입니다. BERT 모델은 기본적으로 문서(혹은 문장) 2개를 입력받는데요, 둘은 `token_type_ids`로 구분합니다. 첫 번째 세그먼트(문서 혹은 문장)에 해당하는 `token_type_ids`는 `0`, 두 번째 세그먼트는 `1`입니다. 이번 실습에서 우리는 문장을 하나씩 넣었으므로 `token_type_ids`가 모두 `0`으로 처리됩니다.

---

## 알아두면 좋아요

토큰화와 관련해 가장 좋은 학습 자료는 허깅페이스 토크나이저 공식 문서입니다. 허깅페이스 토 크나이저는 바이트 페어 인코딩, 워드피스 등 각종 서브워드 토큰화 기법은 물론 유니코드 정규화, 프리토크나이즈, 토큰화 후처리 등 다양한 기능을 제공합니다. 공식 문서(영어)가 꽤 상세해 학습 자료로 손색이 없습니다. 다음 링크로 접속할 수 있습니다.

- [https://huggingface.co/docs/tokenizers/python/latest](https://huggingface.co/docs/tokenizers/python/latest)

좀 더 깊게 공부하고 싶은 독자라면 논문 두 편(영어)을 추천해 드립니다. 하나는 바이트 페어 인코딩을 자연어 처리(기계 번역)에 처음 도입해 본 논문이고, 또 하나는 워드피스 기법을 제안한 논문입니다.

- **Neural Machine Translation of Rare Words with Subword Units** : [https://arxiv.org/pdf/1508.07909.pdf](https://arxiv.org/pdf/1508.07909.pdf)
- **Japanese and Korean Voice Search** : [https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf)

---