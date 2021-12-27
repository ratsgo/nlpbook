---
layout: default
title: Vocab Tutorial
parent: Preprocess
nav_order: 3
---

# 어휘 집합 구축 튜토리얼
{: .no_toc }

[이전](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)에 살펴본 내용을 바탕으로 허깅페이스(Huggingface) 라이브러리를 활용해 바이트 페어 인코딩(Byte Pair Encoding, BPE) 기반의 토크나이저를 만들어 보겠습니다. BPE는 학습 대상 말뭉치에 자주 등장하는 문자열을 토큰으로 인식해, 이를 기반으로 토큰화를 수행하는 기법입니다. BPE 기반 토크나이저를 사용하려면 어휘 집합(vocabulary)부터 구축해야 합니다. 전반적인 과정을 살펴봅니다.  
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 실습 환경 만들기


이 실습에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭하면 코랩 환경에서 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/basic/vocab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

위 노트북은 읽기 권한만 부여돼 있으므로 실행하거나 내용을 고칠 수가 없습니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 개발환경과 관련한 자세한 내용은 [이 글](https://ratsgo.github.io/nlpbook/docs/introduction/environment/)을 참고해 주세요. 

한편 이 실습에서는 하드웨어 가속기가 따로 필요 없으므로 다음과 같이 코랩 화면의 메뉴 탭에서 [런타임 > 런타임 유형 변경]을 선택한 뒤 `None`을 선택합니다.

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />


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

코랩 노트북은 일정 시간 사용하지 않으면 모든 결과물들이 날아갈 수 있습니다. 이번 실습에서 구축할 어휘 집합을 따로 저장해 두기 위해 자신의 구글 드라이브를 코랩 노트북과 연결합니다. 다음 코드를 실행하면 됩니다.

## **코드2** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```


---

## 말뭉치 내려받기 및 전처리

오픈소스 파이썬 패키지 [코포라(Korpora)](https://github.com/ko-nlp/korpora)를 활용해 BPE 수행 대상 말뭉치를 내려받고 전처리합니다. 실습용 말뭉치는 박은정 님이 공개하신 [NAVER Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)입니다. 코드3을 수행하면 데이터를 내려받아 해당 데이터를 `nsmc`라는 변수로 읽어들입니다.

## **코드3** NSMC 다운로드
{: .no_toc .text-delta } 
```python
from Korpora import Korpora
nsmc = Korpora.load("nsmc", force_download=True)
```

코드4를 수행하면 NSMC에 포함된 영화 리뷰들을 순수 텍스트 형태로 지정된 디렉터리에 저장해 둡니다.

## **코드4** NSMC 전처리
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

GPT 계열 모델이 사용하는 토크나이저 기법은 BPE입니다. 단 [앞절](https://ratsgo.github.io/nlpbook/docs/preprocess/bpe)에서 설명한 문자 단위가 아니라 **유니코드 바이트** 수준으로 어휘 집합을 구축하고 토큰화를 수행합니다. 전세계 대부분의 글자는 유니코드로 표현할 수 있으므로 유니코드 바이트 기준 BPE를 사용하면 미등록 토큰 문제에서 비교적 자유롭습니다.

한글은 한 글자가 3개의 유니코드 바이트로 표현되는데요. 예컨대 `안녕하세요`라는 문자열을 각각의 유니코드 바이트에 대응하는 문자\*로 변환하면 다음과 같이 됩니다.

- `안녕하세요` > `ìķĪëħķíķĺìĦ¸ìļĶ`

\* 유니코드(UTF-8) 1바이트를 10진수로 표현하면 0에서 255 사이의 정수가 됩니다. 이 256개 정수 각각을 특정 문자로 매핑한 것입니다. 예컨대 0은 Ā, 255는 ÿ에 각각 대응합니다.
{: .fs-4 .ls-1 .code-example }

바이트 수준으로 BPE를 수행한다는 것은 어휘 집합 구축 대상 말뭉치를 위와 같이 변환하고 이들을 문자 취급해 가장 자주 등장한 문자열을 병합하는 방식으로 어휘 집합을 만든다는 의미입니다. 토큰화 역시 원래 문자열을 위와 같이 변환한 뒤 수행합니다.

우선 바이트 수준 BPE 어휘집합 구축 결과를 저장해 둘 디렉토리를 자신의 구글 드라이브 계정 내에 만듭니다. 코드5를 수행하면 됩니다.

## **코드5** 디렉토리 만들기
{: .no_toc .text-delta } 
```python
import os
os.makedirs("/gdrive/My Drive/nlpbook/bbpe", exist_ok=True)
```

코드6을 실행하면 GPT 계열 모델이 사용하는 바이트 레벨 BPE 어휘 집합을 구축할 수 있습니다. 어휘 집합 구축에 시간이 걸리니 코드 실행 후 잠시 기다려 주세요.

## **코드6** 바이트 레벨 BPE 어휘 집합 구축
{: .no_toc .text-delta } 
```python
from tokenizers import ByteLevelBPETokenizer
bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files=["/root/train.txt", "/root/train.txt"], # 학습 말뭉치를 리스트 형태로 넣기
    vocab_size=10000, # 어휘 집합 크기 조절
    special_tokens=["[PAD]"] # 특수 토큰 추가
)
bytebpe_tokenizer.save_model("/gdrive/My Drive/nlpbook/bbpe")
```

코드6 수행이 끝나면 `/gdrive/My Drive/nlpbook/bbpe`에 `vocab.json`과 `merges.txt`가 생성됩니다. 전자는 바이트 레벨 BPE의 어휘 집합이며 후자는 바이그램 쌍의 병합 우선순위입니다. 각각의 역할에 대해서는 [이전 장](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe)을 참고하시면 좋을 것 같습니다. 그림3과 그림4는 수행 결과 일부를 보여줍니다.

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

BERT는 [워드피스](https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/#%EC%9B%8C%EB%93%9C%ED%94%BC%EC%8A%A4) 토크나이저를 사용합니다. 먼저 앞에서 살펴본 코드1로 의존성 있는 패키지를 설치하고 코드2로 자신의 구글 드라이브를 연결한 후 코드3과 코드4를 실행해 학습 말뭉치를 준비합니다. 

그 다음으로, 워드피스 어휘집합 구축 결과를 저장해 둘 디렉토리를 자신의 구글 드라이브 계정 내에 만듭니다. 코드7을 수행하면 됩니다.

## **코드7** 디렉토리 만들기
{: .no_toc .text-delta } 
```python
import os
os.makedirs("/gdrive/My Drive/nlpbook/wordpiece", exist_ok=True)
```

이후 코드8을 실행하면 BERT 모델이 사용하는 워드피스 어휘 집합을 구축할 수 있습니다. 코드 수행에 시간이 걸리니 잠시만 기다려주세요.

## **코드8** 워드피스 어휘 집합 구축
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

코드8 수행이 끝나면 `save_path`가 가리키는 위치에 워드피스 어휘 집합인 `vocab.txt`가 생성됩니다. 그림5는 워드피스 수행 결과 일부입니다.

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
공중파
절제된
에일리언
99
very 
ᅲᅲᅲᅲᅲ 
간간히
```

---
