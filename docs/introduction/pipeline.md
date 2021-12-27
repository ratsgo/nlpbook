---
layout: default
title: Training Pipeline
parent: Introduction
nav_order: 3
---

# 학습 파이프라인 소개
{: .no_toc }

이번 절에서는 모델 학습의 전체 파이프라인을 소개합니다. 이 파이프라인은 본 사이트에서 소개하는 태스크(문서 분류, 문장 쌍 분류, 개체명 인식, 질의/응답, 문장 생성)에 상관 없이 공통적으로 적용됩니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## 설명에 앞서

이 책에서 진행하는 모든 실습은 [ratsnlp](https://github.com/ratsgo/ratsnlp)라는 오픈소스 파이썬 패키지를 사용합니다. 이 패키지는 구글 코랩(Colab) 환경에서 책의 모든 실습을 진행할 수 있도록 필자가 직접 개발했습니다.

이번 절은 이 책의 각 과제를 어떻게 학습하는지 훑어보기 위한 것으로, 여기서는 모델의 학습 과정만 이해하면 됩니다. 본격적인 코드 실습은 [문서 분류](https://ratsgo.github.io/nlpbook/docs/doc_cls)부터 진행합니다.

---

## 각종 설정값 정하기

모델을 만들려면 가장 먼저 각종 설정값을 정해야 합니다. 어떤 프리트레인 모델을 사용할지, 학습에 사용할 데이터는 무엇인지, 학습 결과는 어디에 저장할지 등이 바로 그것입니다. 이 설정값들은 본격적인 학습에 앞서 미리 선언해 둡니다. 다음 코드는 4장에서 살펴볼 문서 분류를 위한 각종 설정값을 선언한 예입니다.

**하이퍼파라미터(hyperparameter)** 역시 미리 정해둬야 하는 중요한 정보입니다. 하이퍼파라미터란 모델 구조와 학습 등에 직접 관계된 설정값을 가리킵니다. 예를 들어 러닝 레이트(learning rate), 배치 크기(batch size) 등이 있습니다.


## **코드1** 설정값 예시
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
args = nlpbook.TrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="nsmc",
    downstream_corpus_root_dir="/root/Korpora",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-cls",
    learning_rate=5e-5,
    batch_size=32,
)
```

---

## 데이터 내려받기

이 책에서는 프리트레인을 마친 모델을 다운스트림 데이터로 파인튜닝하는 실습을 진행합니다. 파인튜닝을 위해서는 다운스트림 데이터를 미리 내려받아 두어야 합니다. 이 책에서는 상업적으로도 사용 가능한 다운스트림 데이터를 실습에 포함했습니다. 박은정 님이 공개하신 네이버 영화평 말뭉치인 [NAVER Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)가 대표적입니다.

코드2는 downstream_corpus_name(`nsmc`)에 해당하는 말뭉치를 downstream_corpus_root_dir(`/root/Korpora`) 아래에 저장합니다. 

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

코드2에서 확인할 수 있는 것처럼 이 책에서는 다운로드 툴킷으로 ratsnlp뿐 아니라 [코포라(Korpora)](https://github.com/ko-nlp/korpora)라는 오픈소스 파이썬 패키지를 사용합니다. 이 패키지는 다양한 한국어 말뭉치를 쉽게 내려받고 전처리할 수 있도록 도와줍니다.


---

## 프리트레인을 마친 모델 준비하기

대규모 말뭉치를 활용한 프리트레인에는 많은 리소스가 필요합니다. 다행히 최근 많은 기업과 개인이 프리트레인을 마친 모델을 자유롭게 사용할 수 있도록 공개하고 있어서 그 혜택을 볼 수 있습니다.

특히 미국 자연어처리 기업 '허깅페이스(huggingface)'에서 만든 [트랜스포머(transformers)](https://github.com/huggingface/transformers)라는 오픈소스 파이썬 패키지에 주목해야 합니다. 이 책에서는 BERT, GPT 같은 트랜스포머(Transformer) 계열 모델로 튜토리얼을 진행할 예정인데요. 이 패키지를 쓰면 단 몇 줄만으로 모델을 사용할 수 있습니다.

코드3은 이준범 님이 허깅페이스 트랜스포머에 등록해 주신 [kcbert-base 모델](https://github.com/Beomi/KcBERT)을 준비하는 코드입니다. 앞서 보인 코드1에서 args.pretrained_model_name에 `beomi/kcbert-base`라고 선언해 뒀으므로 코드1과 코드3을 순차적으로 실행하면 `kcbert-base` 모델을 쓸 수 있는 상태가 됩니다.

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

코드3을 실행하면 `kcbert-base`가 로컬 저장소에 없으면 자동으로 내려받고, 있으면 캐시 디렉터리에서 읽어옵니다.

---

## 토크나이저 준비하기

자연어 처리 모델의 입력은 대개 **토큰(token)**입니다. 여기서 토큰이란 **문장(sentence)**보다 작은 단위입니다. 한 문장은 여러 개의 토큰으로 구성됩니다. 토큰 분리 기준은 그때그때 다를 수 있습니다. 문장을 띄어쓰기만으로 나눌 수도 있고, 의미의 최소 단위인 **형태소(morpheme)** 단위로 분리할 수도 있습니다. 

문장을 **토큰 시퀀스(token sequence)**로 분석하는 과정을 **토큰화(tokenization)**, 토큰화를 수행하는 프로그램을 **토크나이저(tokenizer)**라고 합니다. 이 책에서는 **Byte Pair Encoding(BPE)**나 **워드피스(wordpiece)** 알고리즘을 채택한 토크나이저를 실습에 활용합니다. 토큰화, BPE, 워드피스는 관련해서는 [Preprocess](https://ratsgo.github.io/nlpbook/docs/preprocess)에서 자세히 다룹니다.

코드4는 `kcbert-base` 모델이 사용하는 토크나이저를 준비하는 코드입니다. 이 역시 토크나이저 관련 파일이 로컬 저장소에 없으면 자동으로 내려받고, 있으면 캐시에서 읽어옵니다.


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

## 데이터 로더 준비하기

파이토치(PyTorch)는 딥러닝 모델 학습을 지원하는 파이썬 라이브러리입니다. 파이토치에는 **데이터 로더(DataLoader)**라는 게 포함돼 있습니다. 파이토치로 딥러닝 모델을 만들고자 한다면 이 데이터 로더를 반드시 정의해야 합니다.

데이터 로더는 학습 때 데이터를 **배치(batch)** 단위로 모델에 밀어 넣어주는 역할을 합니다. 전체 데이터 가운데 일부 인스턴스를 뽑아(sample) 배치를 구성합니다. **데이터셋(dataset)**은 데이터 로더의 구성 요소 가운데 하나입니다. 데이터셋은 여러 인스턴스(문서+레이블)를 보유하고 있습니다. 그림1과 그림2에서는 편의를 위해 인스턴스가 10개인 데이터셋을 상정했지만 대개 인스턴스 개수는 이보다 훨씬 많습니다.

데이터 로더가 배치를 만들 때 인스턴스를 뽑는 방식은 파이토치 사용자가 자유롭게 정할 수 있습니다. 그림1과 그림2는 크기가 3인 배치를 구성하는 예시입니다. 배치1은 0번, 3번, 6번 인스턴스(왼쪽), 배치2는 1번, 4번, 7번 인스턴스(오른쪽)로 구성했음을 확인할 수 있습니다.

## **그림1** dataloader (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/W1YA2XW.jpg" width="300px" title="source: imgur.com" />

## **그림2** dataloader (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/M73kYYN.jpg" width="300px" title="source: imgur.com" />

배치는 그 모양이 고정적어야 할 때가 많습니다. 다시 말해 동일한 배치에 있는 문장들의 토큰(`input_ids`) 개수가 같아야 합니다. 예를 들어 이번에 만들 배치가 데이터셋의 0번, 3번, 6번 인스턴스이고 각각의 토큰 개수가 5, 3, 4개라고 가정해 보겠습니다. 제일 긴 길이로 맞춘다면 0번 인스턴스의 길이(5개)에 따라 3번과 6번 인스턴스의 길이를 늘여야 합니다. 이를 나타내면 다음 그림과 같습니다.

## **그림3** collate
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HR9VIJE.jpg" width="400px" title="source: imgur.com" />

이같이 배치의 모양 등을 정비해 모델의 최종 입력으로 만들어주는 과정을 **컬레이트(collate)**라고 합니다. 컬레이트 과정에는 파이썬 **리스트(list)**에서 파이토치 **텐서(tensor)**로의 변환 등 자료형 변환도 포함됩니다. 컬레이트 수행 방식 역시 파이토치 사용자가 자유롭게 구성할 수 있습니다. 다음은 문서 분류를 위한 데이터 로더를 준비하는 예시입니다.

## **코드5** 문서 분류 데이터 로더 선언
{: .no_toc .text-delta }
```python
from torch.utils.data import DataLoader, RandomSampler
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset corpus = NsmcCorpus()
train_dataset = ClassificationDataset(
    args=args, 
    corpus=corpus, 
    tokenizer=tokenizer, 
    mode="train",
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size, 
    sampler=RandomSampler(train_dataset, replacement=False), 
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
```

참고로 이 책에서 다루는 자연어 처리 모델의 입력은 토큰 시퀀스로 분석된 자연어입니다. 하지만 더 정확하게는 각 토큰이 그에 해당하는 정수(integer)로 변환된 형태입니다. 자연어 처리 모델은 계산 가능한 형태, 즉 숫자 입력을 받는다는 이야기입니다. 각 토큰을 그에 해당하는 정수로 변환하는 과정을 **인덱싱(indexing)**이라고 합니다. 인덱싱은 보통 토크나이저가 토큰화와 함 께 수행합니다. 좀 더 자세한 내용은 [Preprocess](https://ratsgo.github.io/nlpbook/docs/preprocess)를 참고하세요.


---

## 태스크 정의하기

이 책에서는 모델 학습을 할 때 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이라는 라이브러리를 사용합니다. 파이토치 라이트닝은 딥러닝 모델을 학습할 때 반복적인 내용을 대신 수행해줘 사용자가 모델 구축에만 신경쓸 수 있도록 돕는 라이브러리입니다.

이 책에서는 파이토치 라이트닝이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 이 태스크에는 앞서 준비한 모델과 최적화 방법, 학습 과정 등이 정의돼 있습니다. 그림4와 같습니다.

## **그림4** task
{: .no_toc .text-delta }
<img src="https://i.imgur.com/0ewAGnM.jpg" width="300px" title="source: imgur.com" />

**최적화(optimization)**란 특정 조건에서 어떤 값이 최대나 최소가 되도록 하는 과정을 가리킵니다. 앞에서 설명했던 것처럼 우리는 모델의 출력과 정답 사이의 차이를 작게 만드는 데 관심이 있습니다. 이를 위해 **옵티마이저(optimizer)**, **러닝 레이트 스케줄러(learning rate scheduler)** 등을 정의해 둡니다.

모델 학습은 배치 단위로 이뤄집니다. 배치를 모델에 입력한 뒤 모델 출력을 정답과 비교해 차이를 계산합니다. 이후 그 차이를 최소화하는 방향으로 모델을 업데이트합니다. 이 일련의 순환 과정을 **스텝(step)**이라고 합니다. task의 학습 과정에는 1회 스텝에서 벌어지는 일들을 정의해 둡니다. 옵티마이저 등 학습과 관련한 자세한 내용은 [Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics/)를 참고해주세요.



---

## 모델 학습하기

트레이너(Trainer)는 파이토치 라이트닝에서 제공하는 객체로 실제 학습을 수행합니다. 이 트레이너는 GPU\* 등 하드웨어 설정, 학습 기록 로깅, 모델 체크포인트 저장 등 복잡한 설정들을 알아서 해줍니다. 

\* GPU(Graphic Processing Unit)는 그래픽 연산을 빠르게 처리하는 장치입니다. 병렬 연산을 잘하는 덕분에 딥러닝 모델 학습에 널리 쓰이고 있습니다.
{: .fs-4 .ls-1 .code-example }

다음 코드는 문서 분류 모델을 학습하는 예시입니다. 태스크와 트레이너를 정의한 다음, 앞서 준비한 데이터 로더를 가지고 `fit()` 함수를 호출하면 학습을 시작합니다.

## **코드6** 문서 분류 모델 학습
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import ClassificationTask
task = ClassificationTask(model, args)
trainer = nlpbook.get_trainer(args)
trainer.fit(
    task,
    train_dataloader=train_dataloader,
)
```

---
