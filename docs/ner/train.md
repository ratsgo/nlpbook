---
layout: default
title: Training
parent: Named Entity Recognition
nav_order: 2
---

# 개체명 인식 모델 학습하기
{: .no_toc }

개체명 인식 모델의 데이터 전처리 및 학습 과정을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 코랩 노트북

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/named_entity_recognition/train_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---


## 1단계 각종 설정하기

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

이번 튜토리얼에서는 이준범 님이 공개하신 `kcbert-base` 모델을 튜토리얼 데이터로 파인튜닝해볼 예정입니다. 튜토리얼 데이터로는 [한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER)에서 공개한 데이터와 자체 제작한 데이터를 합쳐 사용합니다. 코드3을 실행하면 관련 설정을 할 수 있습니다.

## **코드3** 모델 환경 설정
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.ner import NERTrainArguments
args = NERTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_root_dir="/root/Korpora",
    downstream_corpus_name="ner",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-ner",
    do_eval=True,
    max_seq_length=64,
    batch_size=32,
    epochs=10,
)
```

참고로 `NERTrainArguments`의 각 인자(argument)가 하는 역할과 의미는 다음과 같습니다.

- **pretrained_model_name** : 프리트레인 마친 언어모델의 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_corpus_root_dir** : 다운스트림 데이터를 저장해 둘 위치. `/root/Korpora`라고 함은 코랩 노트북이 실행되는 환경의 루트 하위에 위치한 `Korpora` 디렉토리라는 의미입니다.
- **downstream_corpus_name** : 다운스트림 데이터의 이름.
- **downstream_model_dir** : 파인튜닝된 모델의 체크포인트가 저장될 위치. `/gdrive/My Drive/nlpbook/checkpoint-cls`라고 함은 자신의 구글 드라이브의 `내 폴더` 하위의 `nlpbook/checkpoint-cls` 디렉토리에 모델 체크포인트가 저장됩니다.
- **do_eval** : 학습 중 테스트 데이터로 모델이 얼마나 잘하고 있는지 평가할지 여부. True이면 밸리데이션(validation)을 실시합니다.
- **batch_size** : 배치 크기.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이. 아무 것도 입력하지 않으면 128입니다.
- **epochs** : 학습 에폭 수. 3이라면 학습 데이터를 3회 반복 학습합니다.
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

## 2단계 말뭉치 내려받기

코드6을 실행하면 튜토리얼 데이터 다운로드를 수행합니다. 데이터를 내려받는 도구로 `nlpbook`에 포함된 패키지를 사용해, corpus_name(`nsmc`)에 해당하는 말뭉치를 root_dir(`/root/Korpora`) 이하에 저장해 둡니다.

## **코드6** 말뭉치 다운로드
{: .no_toc .text-delta }
```python
nlpbook.download_downstream_dataset(args)
```


---

## 3단계 토크나이저 준비하기

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

## 4단계 데이터 전처리하기

딥러닝 모델을 학습하려면 학습데이터를 배치(batch) 단위로 지속적으로 모델에 공급해 주어야 합니다. 파이토치(PyTorch)에서는 이 역할을 데이터 로더(DataLoader)가 수행하는데요. 그 개념을 도식적으로 나타내면 그림1과 같습니다.

## **그림1** DataLoader
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bD07LbT.jpg" width="350px" title="source: imgur.com" />

코드8을 수행하면 그림1의 Dataset을 만들 수 있습니다. 여기에서 `NERCorpus`는 "튜토리얼 데이터를 원본 문장 + 개체명 태그를 레이블한 문장" 형태로 읽어들이는 역할을 하고요. `NERDataset`는 그림1의 DataSet 역할을 수행합니다.

## **코드8** 학습 데이터셋 구축
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.ner import NERCorpus, NERDataset
corpus = NERCorpus(args)
train_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
```

그러면 `NERDataset`를 좀 더 자세히 살펴보겠습니다. 이 클래스는 `NERCorpus`와 코드7에서 선언해 둔 토크나이저를 품고 있는데요. `NERCorpus`가 넘겨준 데이터(원본 문장, 레이블한 문장)를 모델이 학습할 수 있는 형태로 가공합니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 한편, 레이블한 문장을 모델이 읽어들일 수 있는 포맷으로 바꿔주는 역할을 합니다.

예컨대 `NERCorpus`가 넘겨준 데이터가 다음과 같다고 가정해 봅시다.

- **원본 문장** : ―효진 역의 김환희(14)가 특히 인상적이었다.
- **레이블한 문장** : ―<효진:PER> 역의 <김환희:PER>(<14:NOH>)가 특히 인상적이었다.

그러면 `NERDataset`은 이를 다음과 같은 정보로 변환합니다. `tokens`는 원본 문장을 토큰화한 뒤 문장 앞뒤에 각각 `[CLS]`와 `[SEP]`를 붙이고 코드3의 `max_seq_length`(=64)가 되도록 패딩 토큰을 추가한 결과입니다. 

- **tokens** : [CLS] [UNK] 효 ##진 역 ##의 김 ##환 ##희 ( 14 ) 가 특히 인상 ##적이 ##었다 . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
- **labels** : [CLS] O B-PER I-PER O O B-PER I-PER I-PER O B-NOH O O O O O O O [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

`labels`는 레이블한 문장을 tokens에 대응되도록 가공한 결과입니다. 문장 앞뒤에 각각 `[CLS]`와 `[SEP]`를 붙이고 코드3의 `max_seq_length`(=64)가 되도록 패딩 토큰을 추가하는 원칙은 tokens와 같습니다. 레이블한 문장을 보면 개체명은 총 3개가 태깅되어 있는데요. `<효진:PER>`, `<김환희:PER>`, `<14:NOH>`이 바로 그것입니다. 

`PER(인명)`으로 레이블링된 `효진`은 tokens 기준 세번째 토큰(`효`)부터 네번째 토큰(`##진`)인 걸 확인할 수 있습니다. 이에 `labels`에는 세번째 토큰과 네번째 토큰이 `PER(인명)`이 되도록 합니다. 단 여기에서 `B-`는 해당 태그의 시작(Begin), `I-`는 해당 태그의 시작이 아님(Inside)이라는 뜻을 가집니다.

`PER(인명)`으로 레이블링된 `김환희`는 일곱번째 토큰(`김`)부터 아홉번째 토큰(`##희`)인 걸 알 수 있습니다. 이에 `labels`에는 일곱번째 토큰과 아홉번째 토큰이 `PER(인명)`이 되도록 합니다. 마찬가지로 `14`의 경우 `labels`의 열한번째 토큰이 `NOH(기타 수량표현)`이 되도록 만들었습니다. 한편 `labels`에서 `O`는 outside의 약자로 개체명이 아닌 부분을 의미합니다.

이후 `NERDataset`은 여기에 인덱싱 작업을 수행하여 `input_ids`, `attention_mask`, `token_type_ids`, `labels`를 만듭니다. `input_ids`는 `tokens`에 인덱싱을 수행한 결과이며 `attention_mask`는 `tokens` 각각의 해당 토큰이 패딩인지(`0`) 아닌지(`1`)를 나타냅니다. `token_type_ids`는 세그먼트(segment) 정보로 기본값은 모두 0으로 넣습니다. 

`label_ids`은 `labels`의 각 개체명 태그(`B-PER`, `I-PER` 등)를 정수로 바꾼 결과입니다. 개체명 인식을 위한 BERT 모델의 입력은 `input_ids`, `attention_mask`, `token_type_ids`이 되며, 출력은 `labels`가 되도록 합니다. 문장을 모델 입력값으로 변환하는 절차와 관련 자세한 내용은 [2장 Preprocess](https://ratsgo.github.io/nlpbook/docs/preprocess)를 참고하시면 좋을 것 같습니다.

- **input_ids** : [2, 1, 3476, 4153, 2270, 4042, 420, 4185, 4346, 11, 11524, 12, 197, 9250, 11662, 8805, 8217, 17, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **attention_mask** : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **token_type_ids** : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- **label_ids** : [0, 4, 5, 15, 4, 4, 5, 15, 15, 4, 6, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

코드9를 실행하면 학습할 때 쓰이는 데이터 로더를 만들 수 있습니다. 그림1에서 Dataset 역할을 하는 `NERDataset`은 학습데이터에 속한 각각의 문장을 `input_ids`, `attention_mask`, `token_type_ids`, `label_ids` 등 네 가지로 변환한 형태로 가지고 있습니다. 그림1에서 인스턴스(instance)에 해당합니다. 데이터 로더는 Dataset이 들고 있는 전체 인스턴스 가운데 배치 크기(코드3에서 정의한 args의 batch_size)만큼을 뽑아 배치 형태로 가공하는 역할을 수행합니다. 

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

코드9를 자세히 보면 `sampler`와 `collate_fn`이 눈에 띕니다. 전자는 샘플링 방식을 정의합니다. 코드9 실행으로 만들어진 데이터 로더는 배치를 만들 때 `NERDataset`이 들고 있는 전체 인스턴스 가운데 batch_size 갯수만큼을 비복원(`replacement=False`) 랜덤 추출합니다. 

후자는 이렇게 뽑힌 인스턴스를 배치로 만드는 역할을 하는 함수입니다. `NERDataset`는 파이썬 리스트(list) 형태의 자료형인데요. 이를 파이토치가 요구하는 자료형인 텐서(tensor) 형태로 바꾸는 등의 역할을 수행합니다.

한편 코드10을 실행하면 평가용 데이터 로더를 구축할 수 있습니다. 학습용 데이터 로더와 달리 평가용 데이터 로더는 `SequentialSampler`를 사용하고 있음을 알 수 있습니다. `SequentialSampler`는 batch_size만큼의 갯수만큼을 인스턴스 순서대로 추출하는 역할을 합니다. 학습 때 배치 구성은 랜덤으로 하는 것이 좋은데요. 평가할 때는 평가용 데이터 전체를 사용하기 때문에 굳이 랜덤으로 구성할 이유가 없어 `SequentialSampler`를 씁니다.

## **코드10** 평가용 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import SequentialSampler
if args.do_eval:
    val_dataset = NERDataset(
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


## 5단계 모델 불러오기

코드11을 수행해 모델을 초기화합니다. `BertForTokenClassification`은 프리트레인을 마친 BERT 모델 위에 [6-1장](https://ratsgo.github.io/nlpbook/docs/ner/overview/)에서 설명한 개체명 인식을 위한 태스크 모듈이 덧붙여진 형태의 모델 클래스입니다.

## **코드11** 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertForTokenClassification
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)
model = BertForTokenClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
```


---

## 6단계 모델 학습시키기

[파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 태스크에는 그림2와 같이 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다.

## **그림2** Task의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="350px" title="source: imgur.com" />

코드12를 실행하면 개체명 인식을 위한 태스크(task)를 정의할 수 있습니다. 모델은 코드11에서 준비한 모델 클래스를 사용하고요, 옵티마이저는 웜업 스케줄링(Warm-up Scheduling)을 적용한 Adam을 사용합니다. 옵티마이저와 관련 자세한 내용은 [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하시면 좋을 것 같습니다.

## **코드12** Task 정의
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.ner import NERTask
task = NERTask(model, args)
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
<img src="https://i.imgur.com/Fo5dL08.png" width="500px" title="source: imgur.com" />

----
