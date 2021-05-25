---
layout: default
title: Vocab Tutorial
parent: Preprocess
nav_order: 3
---

# 어휘 집합 구축 튜토리얼
{: .no_toc }

허깅페이스(Huggingface) 라이브러리를 활용해 [바이트 페어 인코딩(Byte Pair Encoding, BPE)](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe) 기반의 토크나이저를 만들어 보는 튜토리얼입니다. BPE는 학습 대상 말뭉치에 자주 등장하는 문자열을 토큰으로 인식해, 이를 기반으로 토큰화를 수행하는 기법입니다. BPE 기반 토크나이저를 사용하려면 말뭉치 대상으로 어휘 집합(vocabulary)을 구축해야 하는데요. 전반적인 과정을 살펴봅니다.  
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 실습 환경 만들기


이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭하면 코랩 환경에서 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/basic/vocab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 이 튜토리얼에서는 하드웨어 가속기가 따로 필요 없습니다. 그림1과 같이 코랩 화면의 메뉴 탭에서 런타임 > 런타임 유형 변경을 클릭합니다. 이후 그림2의 화면에서 `None`을 선택합니다.

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />


---

## 구글 드라이브 연동하기

코랩 노트북은 일정 시간 사용하지 않으면 당시까지의 모든 결과물들이 날아갈 수 있습니다. 이번 튜토리얼에서 구축할 어휘 집합을 따로 저장해 두기 위해 자신의 구글 드라이브를 코랩 노트북과 연결합니다. 코드 1을 실행하면 됩니다.

## **코드1** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```


---

## 말뭉치 다운로드 및 전처리

오픈소스 파이썬 패키지 [코포라(Korpora)](https://github.com/ko-nlp/korpora)를 활용해 BPE 수행 대상 말뭉치를 내려받고 전처리합니다. 실습용 말뭉치는 박은정 님이 공개하신 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)입니다. 코드2를 수행하면 데이터를 내려받아 해당 데이터를 `nsmc`라는 변수로 읽어들입니다.

## **코드2** NSMC 다운로드
{: .no_toc .text-delta } 
```python
from Korpora import Korpora
nsmc = Korpora.load("nsmc", force_download=True)
```

코드3을 수행하면 NSMC에 포함된 영화 리뷰(순수 텍스트)들을 지정된 디렉토리에 저장해 둡니다.

## **코드3** NSMC 전처리
{: .no_toc .text-delta } 
```python
import os
def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')
write_lines("/root/train.txt", nsmc.train.get_all_texts())
write_lines("/root/test.txt", nsmc.test.get_all_texts())
```

---


## GPT 토크나이저 구축

GPT2, GPT3 등 GPT 계열 모델이 사용하는 토크나이저 기법은 BPE입니다. 단 문자 단위가 아니라 유니코드 바이트 기준으로 어휘 집합을 구축하고 토큰화를 수행합니다. 한글의 경우 1글자가 3개의 유니코드 바이트로 나눠지게 되는데요. 예컨대 `안녕하세요`라는 문자열을 유니코드 바이트로 변환하면 다음과 같이 15개의 유니코드가 됩니다.

- `안녕하세요` > `ìķĪëħķíķĺìĦ¸ìļĶ`

유니코드 바이트 기준으로 BPE를 수행한다고 함은, 어휘 집합 구축 대상 말뭉치를 위와 같이 모두 유니코드로 변환하고 이 유니코드 대상으로 가장 자주 등장한 문자열을 병합하는 방식으로 어휘 집합을 만든다는 의미입니다. 토큰화 역시 문자열을 유니코드 바이트로 일단 변환한 뒤 수행을 하게 됩니다.

우선 바이트 레벨 BPE 어휘집합 구축 결과를 저장해 둘 디렉토리를 자신의 구글 드라이브 계정 내에 만듭니다. 코드4를 수행하면 됩니다.

## **코드4** 디렉토리 만들기
{: .no_toc .text-delta } 
```python
import os
os.makedirs("/gdrive/My Drive/nlpbook/bbpe", exist_ok=True)
```

코드5를 실행하면 GPT 계열 모델이 사용하는 바이트 레벨 BPE 어휘 집합을 구축할 수 있습니다. 학습 말뭉치는 `files`라는 인자에 리스트 형태로 집어 넣으면 되고요. 어휘 집합의 크기는 `vocab_size`로 조절할 수 있습니다.

## **코드5** 바이트 레벨 BPE 어휘 집합 구축
{: .no_toc .text-delta } 
```python
from tokenizers import ByteLevelBPETokenizer
bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files=["/root/train.txt", "/root/train.txt"],
    vocab_size=10000,
    special_tokens=["[PAD]"]
)
bytebpe_tokenizer.save_model("/gdrive/My Drive/nlpbook/bbpe")
```

코드5 수행이 끝나면 `/gdrive/My Drive/nlpbook/bbpe`에 `vocab.json`과 `merges.txt`가 생성됩니다. 전자는 바이트 레벨 BPE의 어휘 집합이며 후자는 바이그램 쌍의 병합 우선순위입니다. 각각의 역할에 대해서는 [이전 장](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)을 참고하시면 좋을 것 같습니다. 그림3과 그림4는 수행 결과입니다.

## **그림3** 바이트 레벨 BPE 어휘 집합 구축 결과 (vocab.json)
{: .no_toc .text-delta }
```
{"[PAD]":0,"!":1,"\"":2,"#":3, ... , "Ġíķĺìłķìļ°":9997,"ìķ¼ê²łëĭ¤":9998,"ìĪĺê³ł":9999}
```

## **그림4** 바이트 레벨 BPE 어휘 집합 구축 결과 (merge.txt)
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

BERT는 [워드피스](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe/#%EC%9B%8C%EB%93%9C%ED%94%BC%EC%8A%A4) 토크나이저를 사용합니다. 코드2와 코드3을 실행해 학습 말뭉치를 먼저 준비합니다. 

그 다음으로, 워드피스 어휘집합 구축 결과를 저장해 둘 디렉토리를 자신의 구글 드라이브 계정 내에 만듭니다. 코드6을 수행하면 됩니다.

## **코드6** 디렉토리 만들기
{: .no_toc .text-delta } 
```python
import os
os.makedirs("/gdrive/My Drive/nlpbook/wordpiece", exist_ok=True)
```

이후 코드7을 실행하면 BERT 모델이 사용하는 워드피스 어휘 집합을 구축할 수 있습니다. 학습 말뭉치는 `files`라는 인자에 리스트 형태로 집어 넣으면 되고요. 어휘 집합의 크기는 `vocab_size`로 조절할 수 있습니다.

## **코드7** 워드피스 어휘 집합 구축
{: .no_toc .text-delta } 
```python
from tokenizers import BertWordPieceTokenizer
wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files=["/root/train.txt", "/root/train.txt"],
    vocab_size=10000,
)
wordpiece_tokenizer.save_model("/gdrive/My Drive/nlpbook/wordpiece")
```

코드3 수행이 끝나면 `save_path`에 워드피스 어휘 집합인 `vocab.txt`가 생성됩니다. 워드피스 어휘 집합의 역할에 대해서는 [이전 장](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)을 참고하시면 좋을 것 같습니다. 그림5는 워드피스 수행 결과입니다.

## **그림5** 워드피스 어휘 집합 구축 결과 (vocab.txt)
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
전락
정리
정재
조정
직장
```

---