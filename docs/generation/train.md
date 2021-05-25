---
layout: default
title: Training
parent: Sentence Generation
nav_order: 2
---


# 문장 생성 모델 파인튜닝하기
{: .no_toc }

문장 생성 모델의 데이터 전처리 및 파인튜닝 과정을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 1단계 코랩 노트북 초기화하기

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/sentence_generation/train_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 모델을 파인튜닝하려면 하드웨어 가속기를 사용해야 계산 속도를 높일 수 있습니다. 코랩에서는 GPU와 TPU 두 종류의 가속기를 지원합니다. 그림1과 같이 코랩 화면의 메뉴 탭에서 `런타임 > 런타임 유형 변경`을 클릭합니다. 이후 그림2와 같이 GPU 혹은 TPU 둘 중 하나를 선택합니다. 

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />

`None을 선택할 경우 하드웨어 가속 기능을 사용할 수 없게 돼 파인튜닝 속도가 급격히 느려집니다. 반드시 GPU 혹은 TPU 둘 중 하나를 사용하세요!
{: .fs-3 .ls-1 .code-example }


---


## 2단계 각종 설정하기

코드1을 실행해 의존성 있는 패키지를 우선 설치합니다. 코랩 환경에서는 명령어 맨 앞에 느낌표(!)를 붙이면 파이썬이 아닌, 배쉬 명령을 수행할 수 있습니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

1단계 코랩 노트북 초기화 과정에서 하드웨어 가속기로 TPU를 선택했다면 코드1에 이어 코드2를 실행하세요. TPU 관련 라이브러리들을 설치하게 됩니다. GPU를 선택했다면 코드1만 수행하고 코드2는 실행하면 안됩니다.

## **코드2** TPU 관련 패키지 설치
{: .no_toc .text-delta }
```python
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8.1-cp37-cp37m-linux_x86_64.whl
```

코랩 노트북은 일정 시간 사용하지 않으면 당시까지의 모든 결과물들이 날아갈 수 있습니다. 모델 체크포인트 등을 저장해 두기 위해 자신의 구글 드라이브를 코랩 노트북과 연결합니다. 코드3을 실행하면 됩니다.

## **코드3** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

이번 튜토리얼에서는 SK텔레콤이 공개한 [KoGPT2 모델](https://github.com/SKT-AI/KoGPT2)을 [NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)로 파인튜닝해볼 예정입니다. 코드4를 실행하면 관련 설정을 할 수 있습니다.

## **코드4** 모델 환경 설정
{: .no_toc .text-delta }
```python
import torch
from ratsnlp.nlpbook.generation import GenerationTrainArguments
args = GenerationTrainArguments(
    pretrained_model_name="skt/kogpt2-base-v2",
    downstream_corpus_name="nsmc",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-generation",
    max_seq_length=32,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)
```

참고로 `GenerationTrainArguments`의 각 인자(argument)가 하는 역할과 의미는 다음과 같습니다.

- **pretrained_model_name** : 프리트레인 마친 언어모델의 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다), 이번 튜토리얼에서는 SK텔레콤이 공개한 KoGPT2(`skt/kogpt2-base-v2`)를 사용합니다.
- **downstream_corpus_name** : 다운스트림 데이터의 이름.
- **downstream_model_dir** : 파인튜닝된 모델의 체크포인트가 저장될 위치. `/gdrive/My Drive/nlpbook/checkpoint-generation`라고 함은 자신의 구글 드라이브의 `내 폴더` 하위의 `nlpbook/checkpoint-generation`에 모델 체크포인트가 저장됩니다.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이. 이보다 긴 문장은 `max_seq_length`로 자르고, 짧은 문장은 `max_seq_length`가 되도록 스페셜 토큰(`PAD`)을 붙여 줍니다.
- **batch_size** : 배치 크기. 하드웨어 가속기로 GPU를 선택(`torch.cuda.is_available() == True`)했다면 32, TPU라면(`torch.cuda.is_available() == False`) 4. 코랩 환경에서 TPU는 보통 8개 코어가 할당되는데 `batch_size`는 코어별로 적용되는 배치 크기이기 때문에 이렇게 설정해 둡니다.
- **learning_rate** : 러닝레이트. 1회 스텝에서 한 번에 얼마나 업데이트할지에 관한 크기를 가리킵니다. 이와 관련한 자세한 내용은 [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하세요.
- **epochs** : 학습 에폭 수. 3이라면 학습 데이터를 3회 반복 학습합니다.
- **tpu_cores** : TPU 코어 수. 하드웨어 가속기로 GPU를 선택(`torch.cuda.is_available() == True`)했다면 0, TPU라면(`torch.cuda.is_available() == False`) 8.
- **seed** : 랜덤 시드.

코드5를 실행해 랜덤 시드를 설정합니다. `args`에 지정된 시드로 고정하는 역할을 합니다.

## **코드5** 랜덤 시드 고정
{: .no_toc .text-delta }
```python
from transformers import set_seed
set_seed(args.seed)
```

코드6을 실행해 각종 로그들을 출력하는 로거를 설정합니다.

## **코드6** 로거 설정
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
nlpbook.set_logger(args)
```



---

## 3단계 말뭉치 내려받기

코드7을 실행하면 [NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc) 다운로드를 수행합니다. 데이터를 내려받는 도구로 오픈소스 패키지 `Korpora`를 사용해, corpus_name(`nsmc`)에 해당하는 말뭉치를 root_dir(`/root/Korpora`) 이하에 저장해 둡니다.

## **코드7** 말뭉치 다운로드
{: .no_toc .text-delta }
```python
from Korpora import Korpora
Korpora.fetch(
    corpus_name=args.downstream_corpus_name,
    root_dir=args.downstream_corpus_root_dir,
    force_download=args.force_download,
)
```


---

## 4단계 토크나이저 준비하기

코드8을 실행해 SK텔레콤이 공개한 KoGPT2 모델이 사용하는 토크나이저를 선언합니다. `eos_token`은 문장 마지막에 붙이는 스페셜 토큰(end of sentence)으로 SK텔레콤이 모델을 프리트레인할 때 이렇게 지정했기 때문에 우리도 같은 방식으로 사용합니다.

## **코드8** 토크나이저 준비
{: .no_toc .text-delta }
```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)
```


---

## 5단계 데이터 전처리하기

딥러닝 모델을 학습하려면 학습데이터를 배치(batch) 단위로 지속적으로 모델에 공급해 주어야 합니다. 파이토치(PyTorch)에서는 이 역할을 데이터 로더(DataLoader)가 수행하는데요. 그 개념을 도식적으로 나타내면 그림3과 같습니다.

## **그림3** DataLoader
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bD07LbT.jpg" width="350px" title="source: imgur.com" />

코드9를 수행하면 그림3의 Dataset을 만들 수 있습니다. 여기에서 `NsmcCorpus`는 NSMC 데이터를 읽어들이는 역할을 하고요. `GenerationDataset`는 그림3의 DataSet 역할을 수행합니다.

## **코드9** 학습 데이터셋 구축
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.generation import NsmcCorpus, GenerationDataset
corpus = NsmcCorpus()
train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
```

아무런 전처리를 하지 않은 NSMC 데이터는 다음과 같이 생겼습니다. `id`는 영화 리뷰의 ID, `document`는 영화 리뷰 문장, `label`은 평점을 가공해 만든 극성 레이블입니다. 0은 부정(negative), 1은 긍정(positive)이라는 의미를 가집니다.

```
id      document        label
9976970 아 더빙.. 진짜 짜증나네요 목소리        0
3819312 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나        1
10265843        너무재밓었다그래서보는것을추천한다      0
(하략)
```

`NsmcCorpus`는 위의 NSMC 데이터를 가공해 다음과 같이 처리합니다. 우리가 파인튜닝할 모델은 텍스트 왼쪽부터 오른쪽으로 순차적으로 읽어나가면서 학습하는 GPT 모델인데요. 파인튜닝시 문장 맨 앞에 극성(polarity) 정보를 부여함으로써, 인퍼런스 과정에서 임의의 극성 정보를 주었을 때 해당 극성에 맞는 문장을 생성할 수 있는 능력이 있는지 여부를 검증해보고자 합니다.

```
부정 아 더빙.. 진짜 짜증나네요 목소리
부정 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
긍정 너무재밓었다그래서보는것을추천한다
(하략)
```

그러면 `GenerationDataset` 클래스는 `NsmcCorpus`와 코드8에서 선언해 둔 토크나이저를 품고 있는데요. 이 클래스는 `NsmcCorpus`가 넘겨준 문장을 `input_ids`, `attention_mask`, `token_type_ids`, `labels` 네 자료로 변환하는 역할을 합니다. `부정 아 더빙.. 진짜 짜증나네요 목소리` 문장 하나만 예를 들면 다음과 같습니다.

우선 `input_ids`는 입력 문장(`부정 아 더빙.. 진짜 짜증나네요 목소리`)을 토큰화한 뒤 이를 인덱싱한 결과입니다. `input_ids`에 패딩 토큰(`</s>`)의 인덱스에 해당하는 `1`이 많이 붙어 있음을 확인할 수 있습니다. 분석 대상 문장의 토큰 길이가 코드4에서 정의한 `max_seq_length`보다 짧아서입니다. 이보다 긴 문장일 경우 시퀀스 길이를 32로 줄입니다.

`attention_mask`는 해당 토큰이 패딩 토큰인지(`0`) 아닌지(`1`)를 나타내며 `token_type_ids`는 세그먼트(segment) 정보로 기본값은 모두 0으로 넣습니다. 한편 `labels`는 `input_ids`와 동일합니다. 코드12에서 사용하는 `GPT2LMHeadModel`이 파인튜닝 과정에서 입력된 `labels`를 오른쪽으로 한칸씩 옮겨서 파인튜닝 태스크가 '입력 토큰의 다음 토큰을 맞추기'가 되도록 합니다.

- **input_ids** : [11775, 9050, 9267, 7700, 9705, 23971, 12870, 8262, 7055, 7098, 8084, 48213, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- **attention_mask** : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **token_type_ids** : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **labels** : [11775, 9050, 9267, 7700, 9705, 23971, 12870, 8262, 7055, 7098, 8084, 48213, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

코드10을 실행하면 학습할 때 쓰이는 데이터 로더를 만들 수 있습니다. 그림1에서 Dataset 역할을 하는 `GenerationDataset`은 학습데이터에 속한 각각의 문장을 `input_ids`, `attention_mask`, `token_type_ids`, `label` 등 네 가지로 변환한 형태로 가지고 있습니다. 그림1에서 인스턴스(instance)에 해당합니다. 데이터 로더는 Dataset이 들고 있는 전체 인스턴스 가운데 배치 크기(코드4에서 정의한 batch_size)만큼을 뽑아 배치 형태로 가공하는 역할을 수행합니다. 

## **코드10** 학습 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import DataLoader, RandomSampler
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
```

코드10을 자세히 보면 `sampler`와 `collate_fn`이 눈에 띕니다. 전자는 샘플링 방식을 정의합니다. 코드10 실행으로 만들어진 데이터 로더는 배치를 만들 때 `GenerationDataset`이 들고 있는 전체 인스턴스 가운데 batch_size 갯수만큼을 비복원(`replacement=False`) 랜덤 추출합니다. 

후자는 이렇게 뽑힌 인스턴스를 배치로 만드는 역할을 하는 함수입니다. `GenerationDataset`는 파이썬 리스트(list) 형태의 자료형인데요. 이를 파이토치가 요구하는 자료형인 텐서(tensor) 형태로 바꾸는 등의 역할을 수행합니다.

한편 코드11을 실행하면 평가용 데이터 로더를 구축할 수 있습니다. 학습용 데이터 로더와 달리 평가용 데이터 로더는 `SequentialSampler`를 사용하고 있음을 알 수 있습니다. 학습 때 배치 구성은 랜덤으로 하는 것이 좋은데요. 평가할 때는 평가용 데이터 전체를 사용하기 때문에 굳이 랜덤으로 구성할 이유가 없기 때문입니다.

## **코드11** 평가용 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import SequentialSampler
val_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
```



---


## 6단계 프리트레인 마친 모델 읽어들이기

코드12를 수행해 모델을 초기화합니다. `GPT2LMHeadModel`은 SK텔레콤이 공개한 KoGPT2가 프리트레인할 때 썼던 모델 클래스입니다. 이번 과제는 프리트레인 태스크와 파인튜닝 태스크가 '다음 단어 맞추기'로 같기 때문에 동일한 모델 클래스를 사용합니다.

## **코드12** 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    args.pretrained_model_name
)
```


---

## 7단계 모델 파인튜닝하기

[파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 태스크에는 그림2와 같이 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다.

## **그림4** Task의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="350px" title="source: imgur.com" />

코드13을 실행하면 문장 생성용 Task를 정의할 수 있습니다. 모델은 코드12에서 준비한 모델 클래스를 사용하고요, 옵티마이저는 웜업 스케줄링(Warm-up Scheduling)을 적용한 Adam을 사용합니다. 옵티마이저와 관련 자세한 내용은 [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하시면 좋을 것 같습니다.

## **코드13** Task 정의
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.generation import GenerationTask
task = GenerationTask(model, args)
```

코드14를 실행하면 트레이너(Trainer)를 정의할 수 있습니다. 이 트레이너는 GPU/TPU 설정, 로그 및 체크포인트 등 귀찮은 설정들을 알아서 해줍니다.

## **코드14** Trainer 정의
{: .no_toc .text-delta }
```python
trainer = nlpbook.get_trainer(args)
```

코드15처럼 트레이너의 `fit` 함수를 호출하면 학습이 시작됩니다. 그림5는 코랩 환경에서 학습되는 화면입니다.

## **코드15** 학습 개시
{: .no_toc .text-delta }
```python
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

## **그림5** 코랩 환경에서의 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tG2U4Wp.png" width="500px" title="source: imgur.com" />

----
