---
layout: default
title: Vocab Tutorial
parent: Preprocess
nav_order: 3
---

# Vocabulary 구축 튜토리얼
{: .no_toc }

Huggingface 라이브러리를 활용해 [바이트 페어 인코딩(Byte Pair Encoding, BPE)](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe) 기반의 토크나이저를 만들어 보는 튜토리얼입니다. BPE는 학습 대상 말뭉치에 자주 등장하는 문자열을 토큰으로 인식해, 이를 기반으로 토큰화를 수행하는 기법입니다. BPE 기반 토크나이저를 사용하려면 말뭉치 대상으로 어휘 집합(vocabulary)을 구축해야 하는데요. 전반적인 과정을 살펴봅니다.  
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 말뭉치 다운로드 및 전처리

오픈소스 파이썬 패키지 [코포라(Korpora)](https://github.com/ko-nlp/korpora)를 활용해 BPE 수행 대상 말뭉치를 내려받고 전처리합니다. 실습용 말뭉치는 박은정 님이 공개하신 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)입니다. 코드1을 수행하면 데이터를 내려받아 해당 데이터를 `nsmc`라는 변수로 읽어들입니다.

## **코드1** NSMC 다운로드
{: .no_toc .text-delta } 
```python
from Korpora import Korpora
nsmc = Korpora.load("nsmc", force_download=True)
```

코드2를 수행하면 NSMC에 포함된 영화 리뷰(순수 텍스트)들을 지정된 디렉토리(`save_path`)에 저장해 둡니다.

## **코드2** NSMC 전처리
{: .no_toc .text-delta } 
```python
import os
def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')
train_fpath = os.path.join(save_path, "train.txt")
test_fpath = os.path.join(save_path, "test.txt")
write_lines(train_fpath, nsmc.train.get_all_texts())
write_lines(test_fpath, nsmc.test.get_all_texts())
```

---


## GPT 토크나이저 구축

GPT2, GPT3 등 GPT 계열 모델이 사용하는 토크나이저 기법은 BPE입니다. 단 문자 단위가 아니라 유니코드 바이트 기준으로 어휘 집합을 구축하고 토큰화를 수행합니다. 한글의 경우 1글자가 3개의 유니코드 바이트로 나눠지게 되는데요. 예컨대 `안녕하세요`라는 문자열을 유니코드 바이트로 변환하면 다음과 같이 15개의 유니코드가 됩니다.

- `안녕하세요` > `ìķĪëħķíķĺìĦ¸ìļĶ`

유니코드 바이트 기준으로 BPE를 수행한다고 함은, 어휘 집합 구축 대상 말뭉치를 위와 같이 모두 유니코드로 변환하고 이 유니코드 대상으로 가장 자주 등장한 문자열을 병합하는 방식으로 어휘 집합을 만든다는 의미입니다. 토큰화 역시 문자열을 유니코드 바이트로 일단 변환한 뒤 수행을 하게 됩니다.

코드2를 실행하면 GPT 계열 모델이 사용하는 바이트 레벨 BPE 어휘 집합을 구축할 수 있습니다. 학습 말뭉치는 `files`라는 인자에 리스트 형태로 집어 넣으면 되고요. 어휘 집합의 크기는 `vocab_size`로 조절할 수 있습니다.

## **코드2** 바이트 레벨 BPE 어휘 집합 구축
{: .no_toc .text-delta } 
```python
from tokenizers import ByteLevelBPETokenizer
bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files=[train_fpath, test_fpath],
    vocab_size=10000,
    special_tokens=["[PAD]"]
)
bytebpe_tokenizer.save_model(save_path)
```

코드2 수행이 끝나면 `save_path`에 `vocab.json`과 `merges.txt`가 생성됩니다. 전자는 바이트 레벨 BPE의 어휘 집합이며 후자는 바이그램 쌍의 병합 우선순위입니다. 각각의 역할에 대해서는 [이전 장](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)을 참고하시면 좋을 것 같습니다. 그림1과 그림2는 수행 결과입니다.

## **그림1** 바이트 레벨 BPE 어휘 집합 구축 결과 (vocab.json)
{: .no_toc .text-delta }
```
{"[PAD]":0,"!":1,"\"":2,"#":3, ... , "Ġíķĺìłķìļ°":9997,"ìķ¼ê²łëĭ¤":9998,"ìĪĺê³ł":9999}
```

## **그림2** 바이트 레벨 BPE 어휘 집합 구축 결과 (merge.txt)
{: .no_toc .text-delta }
```
#version: 0.2 - Trained by `huggingface/tokenizers`
Ġ ì
Ġ ë
ì Ŀ
...
ìķ¼ ê²łëĭ¤
ìĪĺ ê³ł
ê°Ħ ìĿ´
```

---

## BERT 토크나이저 구축

BERT는 [워드피스](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe/#%EC%9B%8C%EB%93%9C%ED%94%BC%EC%8A%A4) 토크나이저를 사용합니다. 코드1과 코드2를 실행해 학습 말뭉치를 먼저 준비합니다. 

이후 코드3을 실행하면 BERT 모델이 사용하는 워드피스 어휘 집합을 구축할 수 있습니다. 학습 말뭉치는 `files`라는 인자에 리스트 형태로 집어 넣으면 되고요. 어휘 집합의 크기는 `vocab_size`로 조절할 수 있습니다.

## **코드3** 워드피스 어휘 집합 구축
{: .no_toc .text-delta } 
```python
from tokenizers import BertWordPieceTokenizer
wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files=[train_fpath, test_fpath],
    vocab_size=10000,
)
wordpiece_tokenizer.save_model(save_path)
```

코드3 수행이 끝나면 `save_path`에 워드피스 어휘 집합인 `vocab.txt`가 생성됩니다. 워드피스 어휘 집합의 역할에 대해서는 [이전 장](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)을 참고하시면 좋을 것 같습니다. 그림3은 워드피스 수행 결과입니다.

## **그림3** 워드피스 어휘 집합 구축 결과 (vocab.txt)
{: .no_toc .text-delta }
```
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
!
"
...
에일리언
99
very
ㅠㅠㅠㅠㅠ
간간히
```

---