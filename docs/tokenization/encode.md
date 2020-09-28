---
layout: default
title: Batch Encoding
parent: Preprocess
nav_order: 5
---

# Batch Encoding
{: .no_toc }

자연어 처리 모델은 숫자만 입력받을 수 있습니다. 토큰화를 마친 문장을 모델 입력으로 어떻게 집어 넣는지 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

1. TOC
{:toc}

---

## 원본 문서

```
9976970	아 더빙.. 진짜 짜증나네요 목소리	0
3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
5403919	막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.	0
1927486	별루 였다..	0
```

```python
data = """9976970	아 더빙.. 진짜 짜증나네요 목소리	0
3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
5403919	막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.	0
1927486	별루 였다..	0""".split("\n")
sentences = [line.split("\t")[1] for line in data]
labels = [line.split("\t")[2] for line in data]

from transformers import BertTokenizer
pretrained_model_cache_dir="/Users/david/works/cache/kcbert-base"
tokenizer = BertTokenizer.from_pretrained(pretrained_model_cache_dir, do_lower_case=False)
first = tokenizer(
    sentences[:2],
    padding="max_length",
    max_length=12,
    truncation=True,
)
from pprint import pprint
pprint(["|".join(tokenizer.convert_ids_to_tokens(el)) for el in first['input_ids']])
last = tokenizer(
    sentences[2:],
    padding="max_length",
    max_length=12,
    truncation=True,
)
pprint(["|".join(tokenizer.convert_ids_to_tokens(el)) for el in last['input_ids']])

pprint(["|".join([str(e) for e in el]) for el in first['input_ids']])
pprint(["|".join([str(e) for e in el]) for el in first['attention_mask']])

pprint(["|".join([str(e) for e in el]) for el in last['input_ids']])
pprint(["|".join([str(e) for e in el]) for el in last['attention_mask']])
```

---

## PADDING & TRUNCATION

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|인스턴스1|[CLS]|아|더|##빙|.|.|진짜|짜증나네|##요|목소리|[SEP]|[PAD]|
|인스턴스2|[CLS]|흠|.|.|.|포|##스터|##보고|초딩|##영화|##줄|[SEP]|
|인스턴스3|[CLS]|막|걸|##음|##마|뗀|3|##세|##부터|초등학교|1|[SEP]|
|인스턴스4|[CLS]|별|##루|였|##다|.|.|[SEP]|[PAD]|[PAD]|[PAD]|[PAD]|


---


## CONVERT TO ID

|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|인스턴스1|2|2170|832|5045|17|17|7992|29734|4040|10720|3|0|
|인스턴스2|2|3521|17|17|17|3294|13069|8190|10635|13796|4006|3|
|인스턴스3|2|1294|254|4126|4168|1038|22|4066|8042|15507|20|3|
|인스턴스4|2|1558|4532|2281|4020|17|17|3|0|0|0|0|


---


## ATTENTION MASKS


|구분|토큰1|토큰2|토큰3|토큰4|토큰5|토큰6|토큰7|토큰8|토큰9|토큰10|토큰11|토큰12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|인스턴스1|1|1|1|1|1|1|1|1|1|1|1|0|
|인스턴스2|1|1|1|1|1|1|1|1|1|1|1|1|
|인스턴스3|1|1|1|1|1|1|1|1|1|1|1|1|
|인스턴스4|1|1|1|1|1|1|1|1|0|0|0|0|


---


## LABELS

|구분|레이블|
|---|---|
|인스턴스1|0|
|인스턴스2|1|
|인스턴스3|0|
|인스턴스4|0|


---