---
layout: default
title: Training Pipeline
parent: Introduction
nav_order: 3
---

# 학습 파이프라인 소개
{: .no_toc }

모델 학습 전체 파이프라인을 소개합니다. 이 파이프라인은 본 사이트에서 소개하는 태스크(문서 분류, 개체명 인식, 질의/응답, 문서 검색, 문장 생성)에 상관 없이 공통적으로 적용됩니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## 각종 설정값 정하기

모델을 만들기 위해서 가장 먼저 해야 하는 것은 각종 설정 값들을 정하는 일입니다. 학습에 사용할 데이터는 무엇인지, 데이터가 어디에 위치해 있는지, 학습 결과는 어디에 저장해 둘지 등은 모델 학습 과정에서 자주 참고하는 정보이기 때문입니다.

하이퍼파라메터(hyperparameter) 역시 미리 정해두어야 하는 중요 정보입니다. 하이퍼파라메터란 모델의 환경 설정 값을 가리킵니다. 예컨대 모델 크기 등이 바로 그것입니다.

이들 설정값들은 본격적인 학습에 앞서 미리 선언해 둡니다. 예컨대 다음과 같습니다.

## **코드1** 설정값 예시
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
args = nlpbook.TrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_root_dir="/root/Korpora",
    downstream_corpus_name="nsmc",
    downstream_task_name="document-classification",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-cls",
    do_eval=True,
    batch_size=32,
)
```

---

## 데이터 내려받기

이 책에서는 프리트레인을 마친 모델을 다운스트림 데이터로 파인튜닝하는 방식의 튜토리얼을 진행합니다. 파인튜닝을 위해서는 다운스트림 데이터를 미리 내려받아 두어야 합니다.

본서에서는 상업적으로도 사용 가능한 라이센스를 가진 다운스트림 데이터를 튜토리얼에 포함했습니다. 박은정 님이 공개하신 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)가 대표적입니다.

코드2는 NSMC 다운로드를 수행합니다. corpus_name(`nsmc`)에 해당하는 말뭉치를 root_dir(`/root/Korpora`) 이하에 저장해 둡니다. 앞서 우리는 코드1에서 args를 선언해 두었으므로 가능한 일입니다.

## **코드2** 데이터 다운로드 예시
{: .no_toc .text-delta }
```python
from Korpora import Korpora
Korpora.fetch(
    corpus_name=args.downstream_corpus_name,
    root_dir=args.downstream_corpus_root_dir,
    force_download=True,
)
```

코드2에서 확인할 수 있는 것처럼 본서에서는 다운로드 툴킷으로 [코포라(Korpora)](https://github.com/ko-nlp/korpora)라는 오픈소스 파이썬 패키지를 사용합니다. 이 패키지는 다양한 한국어 말뭉치를 쉽게 내려받고 전처리할 수 있도록 도와줍니다.


---

## 프리트레인 마친 모델 준비

대규모 말뭉치를 활용한 프리트레인에는 대단히 많은 리소스가 필요합니다. 다행히 최근 많은 기업과 개인이 프리트레인을 마친 모델을 자유롭게 사용할 수 있도록 공개해주고 계셔서 그 혜택을 볼 수 있습니다.

특히 미국 자연어처리 기업 '허깅페이스(huggingface)'에서 만든 [트랜스포머(transformers)](https://github.com/huggingface/transformers)라는 오픈소스 파이썬 패키지에 주목할 필요가 있습니다. 이 책에서는 BERT, GPT 같은 트랜스포머(trasformer) 계열 모델로 튜토리얼을 진행할 예정인데요. 이 패키지를 쓰면 단 몇 줄만으로 모델을 사용할 수 있습니다.

코드3은 이준범 님이 허깅페이스 트랜스포머에 등록해 주신 [kcbert-base 모델](https://github.com/Beomi/KcBERT)을 준비하는 코드입니다. 우리는 이미 코드1 수행으로 args.pretrained_model_name에 `beomi/kcbert-base`라고 선언해 둔 상황입니다. 따라서 코드1과 코드3을 순차적으로 실행하면 `kcbert-base` 모델을 쓸 수 있는 상태가 됩니다.

## **코드3** kcbert-base 모델 준비
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertForSequenceClassification
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=2,
)
model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
```

코드3을 실행하면 `kcbert-base`가 로컬에 없다면 자동으로 내려받고 있다면 캐시 디렉토리에서 읽어옵니다. 이 모델은 네이버 댓글 데이터 12기가바이트(GB)를 프리트레인한 BERT인데요. 이토록 간편하고 자유롭게 사용할 수 있도록 기여해 주신 트랜스포머 패키지 컨트리뷰터 여러분과 이준범 님께 깊은 감사 인사 전합니다.

---

## 토크나이저 준비

자연어 처리 모델의 입력은 대개 **토큰(token)**입니다. 여기서 토큰이란 문장(sentence)보다 작은 단위입니다. 한 문장은 여러 개의 토큰으로 구성됩니다. 토큰 분리 기준은 그때그때 다를 수 있습니다. 문장을 띄어쓰기만으로 나눌 수도 있고, 의미의 최소 단위인 형태소(morpheme) 단위로 분리할 수도 있습니다. 

문장을 토큰 시퀀스(sequence)로 분석하는 과정을 **토큰화(tokenization)**, 토큰화를 수행하는 프로그램을 **토크나이저(tokenizer)**라고 합니다. 이 책에서는 **Byte Pair Encoding(BPE)** 계열의 알고리즘을 채택한 토크나이저를 튜토리얼에 활용할 계획입니다. BPE와 토큰화와 관련해서는 [3장 Vocab & Tokenization](http://localhost:4000/nlpbook/docs/preprocess)을 참고하시면 좋을 것 같습니다.

코드4는 `kcbert-base` 모델이 사용하는 BPE 계열 토크나이저를 준비하는 코드입니다. 이 역시 토크나이저가 로컬에 없다면 자동으로 내려받고 있다면 캐시에서 읽어옵니다.


## **코드4** kcbert-base 토크나이저 준비
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```

---

## 데이터 로더

파이토치(PyTorch)는 딥러닝 모델 학습을 지원하는 파이썬 라이브러리입니다. 파이토치에는 **데이터 로더(DataLoader)**라는 게 포함돼 있는데요. 파이토치로 딥러닝 모델을 만들고자 한다면 데이터 로더를 반드시 정의해야 합니다.

데이터 로더는 학습 때 데이터를 배치(batch) 단위로 모델에 밀어 넣어주는 역할을 합니다. 배치는 전체 데이터 가운데 일부 인스턴스를 뽑아 만든 미니 데이터셋으로 이해하면 쉽습니다. 그 개념도는 그림1과 같습니다.

## **그림1** dataloader
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bD07LbT.jpg" width="300px" title="source: imgur.com" />

**데이터셋(Dataset)**은 데이터 로더의 구성 요소 가운데 하나입니다. 말뭉치에 속한 각각의 문장(그림1에선 instance)을 들고 있는 객체입니다. 데이터 로더는 데이터셋에 있는 인스턴스 가운데 일부를 뽑아(sample) 배치를 만듭니다. 이를 뽑는 방식은 파이토치 사용자가 자유롭게 구성할 수 있습니다.

배치는 그 모양이 고정적어야 하는 경우도 많습니다. 예컨대 이번에 만들어야 하는 배치가 데이터셋의 0번, 7번, n번 인스턴스이고 각각의 토큰 갯수가 3, 5, 6개라고 가정해 보겠습니다. 모양이 고정적이어야 한다면 제일 긴 n번 인스턴스 길이(6개)에 맞춰 0번, 7번 인스턴스의 길이를 늘려주거나, 제일 짧은 0번 인스턴스(3개)에 맞춰 7번, 7번 인스턴스의 길이를 짧게 만들어 주어야 합니다. 

이같이 배치의 모양 등을 정비해 모델의 최종 입력으로 만들어주는 과정을 **collate**라고 합니다. collate 수행 방식 역시 파이토치 사용자가 자유롭게 구성할 수 있습니다.

---

## 태스크 정의

이 책에서는 모델 학습을 할 때 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이라는 패키지를 사용합니다. 파이토치로 딥러닝 모델을 학습할 때 반복적인 내용을 대신 수행해줘 사용자가 모델 구축에만 신경쓸 수 있도록 돕는 라이브러리입니다.

이 책에서는 파이토치 라이트닝이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 이 태스크에는 앞서 준비한 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다. 태스크와 관련한 구체적인 내용들은 4장 이후의 튜토리얼 파트를 참고하시면 좋을 것 같습니다.

## **그림2** task
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="300px" title="source: imgur.com" />

---

## 모델 학습

트레이너(Trainer)는 파이토치 라이트닝에서 제공하는 객체로 실제 학습을 수행합니다. 이 트레이너는 GPU 설정, 로그 및 체크포인트 등 귀찮은 설정들을 알아서 해줍니다. 

## **그림3** trainer
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tBfMqq4.jpg" width="400px" title="source: imgur.com" />

코드5처럼 앞서 준비한 데이터 로더와 태스크를 넣고 트레이너를 선언한 뒤 `fit` 함수를 호출하면 학습이 시작됩니다.

## **코드5** 학습 코드
{: .no_toc .text-delta }
```python
task = ClassificationTask(model, args)
_, trainer = nlpbook.get_trainer(args)
trainer.fit(
    task,
    train_dataloader=train_dataloader,
)
```

---
