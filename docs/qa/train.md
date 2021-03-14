---
layout: default
title: Training
parent: Question Answering
nav_order: 2
---


# Training
{: .no_toc }

질의 응답 모델의 데이터 전처리 및 학습 과정을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 코랩 노트북

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/question_answering/train_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---


## 각종 설정

코드1을 실행해 의존성 있는 패키지를 우선 설치합니다. 코랩 환경에서는 명령어 맨 앞에 느낌표(!)를 붙이면 파이썬이 아닌, 배쉬 명령을 수행할 수 있습니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

코랩 노트북은 일정 시간 사용하지 않으면 당시까지의 모든 결과물들이 날아갈 수 있습니다. 모델 체크포인트 등을 저장해 두기 위해 자신의 구글 드라이브를 코랩 노트북과 연결합니다. 코드2를 실행하면 됩니다.

## **코드2** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

이번 튜토리얼에서는 이준범 님이 공개하신 `kcbert-base` 모델을 NSMC 데이터로 파인튜닝해볼 예정입니다. 코드3을 실행하면 관련 설정을 할 수 있습니다.

## **코드3** 모델 환경 설정
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import QATrainArguments
args = QATrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_root_dir="/root/Korpora",
    downstream_corpus_name="korquad-v1",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-qa",
    max_seq_length=128,
    max_query_length=32,
    doc_stride=64,
    do_eval=True,
    batch_size=32,
    epochs=5,
)
```

참고로 `QATrainArguments`의 각 인자(argument)가 하는 역할과 의미는 다음과 같습니다.

- **pretrained_model_name** : 프리트레인 마친 언어모델의 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_corpus_root_dir** : 다운스트림 데이터를 저장해 둘 위치. `/root/Korpora`라고 함은 코랩 노트북이 실행되는 환경의 루트 하위에 위치한 `Korpora` 디렉토리라는 의미입니다.
- **downstream_corpus_name** : 다운스트림 데이터의 이름.
- **downstream_task_name** : 다운스트림 태스크의 이름.
- **downstream_model_dir** : 파인튜닝된 모델의 체크포인트가 저장될 위치. `/gdrive/My Drive/nlpbook/checkpoint-cls`라고 함은 자신의 구글 드라이브의 `내 폴더` 하위의 `nlpbook/checkpoint-cls` 디렉토리에 모델 체크포인트가 저장됩니다.
- **do_eval** : 학습 중 테스트 데이터로 모델이 얼마나 잘하고 있는지 평가할지 여부. True이면 밸리데이션(validation)을 실시합니다.
- **batch_size** : 배치 크기.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이. 아무 것도 입력하지 않으면 128입니다.
- **seed** : 랜덤 시드 값. 아무 것도 입력하지 않으면 7입니다. 

코드4를 실행해 랜덤 시드를 설정합니다. `args`에 지정된 시드로 고정하는 역할을 합니다.

## **코드4** 랜덤 시드 고정
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
nlpbook.seed_setting(args)
```

코드5를 실행해 각종 로그들을 출력하는 로거를 설정합니다.

## **코드5** 로거 설정
{: .no_toc .text-delta }
```python
nlpbook.set_logger(args)
```



---

## 말뭉치 다운로드

코드6을 실행하면 NSMC 데이터 다운로드를 수행합니다. 다운로드 툴킷으로 [코포라(Korpora)](https://github.com/ko-nlp/korpora)라는 오픈소스 파이썬 패키지를 사용해, corpus_name(`nsmc`)에 해당하는 말뭉치를 root_dir(`/root/Korpora`) 이하에 저장해 둡니다.

## **코드6** 말뭉치 다운로드
{: .no_toc .text-delta }
```python
nlpbook.download_downstream_dataset(args)
```


---

## 토크나이저 준비

코드7을 실행해 이준범 님이 공개하신 `kcbert-base` 모델이 사용하는 토크나이저를 선언합니다.

## **코드7** 토크나이저 준비
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```


---

## 데이터 전처리

딥러닝 모델을 학습하려면 학습데이터를 배치(batch) 단위로 지속적으로 모델에 공급해 주어야 합니다. 파이토치(PyTorch)에서는 이 역할을 데이터 로더(DataLoader)가 수행하는데요. 그 개념을 도식적으로 나타내면 그림1과 같습니다.

## **그림1** DataLoader
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bD07LbT.jpg" width="350px" title="source: imgur.com" />

코드8을 수행하면 그림1의 Dataset을 만들 수 있습니다. 여기에서 `NsmcCorpus`는 csv 파일 형식의 NSMC 데이터를 문장(영화 리뷰) + 레이블(`긍정`, `부정`) 형태로 읽어들이는 역할을 하고요. `ClassificationDataset`는 그림1의 DataSet 역할을 수행합니다.

## **코드8** 학습 데이터셋 구축
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import KorQuADV1Corpus, QADataset
corpus = KorQuADV1Corpus()
train_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
```

그러면 `QADataset`를 좀 더 자세히 살펴보겠습니다. 이 클래스는 `NsmcCorpus`와 코드7에서 선언해 둔 토크나이저를 품고 있는데요. `NsmcCorpus`가 넘겨준 데이터(문장, 레이블)를 모델이 학습할 수 있는 형태로 가공합니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 한편, 레이블 역시 정수(integer)로 바꿔주는 역할을 합니다.

예컨대 `NsmcCorpus`가 넘겨준 데이터가 다음과 같다고 가정해 봅시다.

- **text** : 아 더빙.. 진짜 짜증나네요 목소리
- **label** : 0(부정)

그러면 ClassificationDataset은 이를 다음과 같은 정보로 변환합니다. `input_ids`, `attention_mask`, `token_type_ids`의 길이가 모두 128인 이유는 토큰 기준 최대 길이(`max_seq_length`)를 코드3의 args에서 128로 설정해 두었기 때문입니다. 

`input_ids`에 패딩 토큰에 해당하는 `0`이 많이 붙어 있음을 확인할 수 있습니다. 분석 대상 문장의 토큰 길이가 `max_seq_length`보다 짧아서입니다. 이보다 긴 문장일 경우 128로 줄입니다.

`attention_mask`는 해당 토큰이 패딩 토큰인지(`0`) 아닌지(`1`)를 나타내며 `token_type_ids`는 세그먼트(segment) 정보로 기본값은 모두 0으로 넣습니다. `label`은 정수로 변환됐습니다. 문장을 모델 입력값으로 변환하는 절차와 관련 자세한 내용은 [2장 Preprocess](https://ratsgo.github.io/nlpbook/docs/preprocess)를 참고하시면 좋을 것 같습니다.

- **input_ids** : [2, 2170, 832, 5045, 17, 17, 7992, 29734, 4040, 10720, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **attention_mask** : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **token_type_ids** : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **label** : 0

코드9를 실행하면 학습할 때 쓰이는 데이터 로더를 만들 수 있습니다. 그림1에서 Dataset 역할을 하는 `ClassificationDataset`은 학습데이터에 속한 각각의 문장을 `input_ids`, `attention_mask`, `token_type_ids`, `label` 등 네 가지로 변환한 형태로 가지고 있습니다. 그림1에서 인스턴스(instance)에 해당합니다. 데이터 로더는 Dataset이 들고 있는 전체 인스턴스 가운데 배치 크기(코드3에서 정의한 args의 batch_size)만큼을 뽑아 배치 형태로 가공하는 역할을 수행합니다. 

## **코드9** 학습 데이터 로더 구축
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

코드9를 자세히 보면 `sampler`와 `collate_fn`이 눈에 띕니다. 전자는 샘플링 방식을 정의합니다. 코드9 실행으로 만들어진 데이터 로더는 배치를 만들 때 `ClassificationDataset`이 들고 있는 전체 인스턴스 가운데 batch_size 갯수만큼을 비복원(`replacement=False`) 랜덤 추출합니다. 

후자는 이렇게 뽑힌 인스턴스를 배치로 만드는 역할을 하는 함수입니다. `ClassificationDataset`는 파이썬 리스트(list) 형태의 자료형인데요. 이를 파이토치가 요구하는 자료형인 텐서(tensor) 형태로 바꾸는 등의 역할을 수행합니다.

한편 코드10을 실행하면 평가용 데이터 로더를 구축할 수 있습니다. 학습용 데이터 로더와 달리 평가용 데이터 로더는 `SequentialSampler`를 사용하고 있음을 알 수 있습니다. 학습 때 배치 구성은 랜덤으로 하는 것이 좋은데요. 평가할 때는 평가용 데이터 전체를 사용하기 때문에 굳이 랜덤으로 구성할 이유가 없기 때문입니다.

## **코드10** 평가용 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import SequentialSampler
if args.do_eval:
    val_dataset = QADataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="val",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
else:
    val_dataloader = None
```



---


## 모델 로딩

코드11을 수행해 모델을 초기화합니다. `BertForTokenClassification`은 프리트레인을 마친 BERT 모델 위에 [4-1장](https://ratsgo.github.io/nlpbook/docs/classification/overview/)에서 설명한 문서 분류용 태스크 모듈이 덧붙여진 형태의 모델 클래스입니다.

## **코드11** 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertForQuestionAnswering
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)
model = BertForQuestionAnswering.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
```


---

## 모델 학습

[파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 태스크에는 그림2와 같이 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다.

## **그림2** Task의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="350px" title="source: imgur.com" />

코드12를 실행하면 문서 분류용 Task를 정의할 수 있습니다. 모델은 코드11에서 준비한 모델 클래스를 사용하고요, 옵티마이저는 웜업 스케줄링(Warm-up Scheduling)을 적용한 Adam을 사용합니다. 옵티마이저와 관련 자세한 내용은 [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하시면 좋을 것 같습니다.

## **코드12** Task 정의
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import QATask
task = QATask(model, args)
```

코드13을 실행하면 트레이너(Trainer)를 정의할 수 있습니다. 이 트레이너는 GPU 설정, 로그 및 체크포인트 등 귀찮은 설정들을 알아서 해줍니다.

## **코드13** Trainer 정의
{: .no_toc .text-delta }
```python
trainer = nlpbook.get_trainer(args)
```

코드14처럼 트레이너의 fit 함수를 호출하면 학습이 시작됩니다. 그림3은 코랩 환경에서 학습되는 화면입니다.

## **코드14** 학습 개시
{: .no_toc .text-delta }
```python
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

## **그림3** 코랩 환경에서의 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/MDv0WkJ.png" width="500px" title="source: imgur.com" />

----
