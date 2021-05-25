---
layout: default
title: ↗️ Customization
parent: Document Classification
nav_order: 4
---

# ↗️ 나만의 문서 분류 모델 만들기
{: .no_toc }

커스텀 데이터, 토크나이저, 모델, 트레이너(trainer)로 나만의 문서 분류 모델을 만드는 과정을 소개합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 내 데이터 사용하기

우리 책 문서 분류 튜토리얼은 박은정 님이 공개한 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)를 사용하고 있는데요. 나만의 문서 분류 모델 구축을 위한 첫걸음은 내가 가진 데이터를 활용하는 것일 겁니다. 이를 위해서는 말뭉치를 읽어들이는 코드에 대한 이해가 선행되어야 할텐데요. 우리 책 튜토리얼에서 NSMC 데이터를 어떻게 읽고 전처리하고 있는지 살펴보겠습니다. 코드1과 같습니다.

## **코드1** NSMC 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
# 데이터 로딩
from ratsnlp.nlpbook.classification import NsmcCorpus
corpus = NsmcCorpus()

# 데이터 전처리
from ratsnlp.nlpbook.classification import ClassificationDataset
train_dataset = ClassificationDataset(
	args=args,
	corpus=corpus,
	tokenizer=tokenizer,
	mode="train",
)
```

코드1에서 선언한 `NsmcCorpus` 클래스는 CSV 파일 형식의 NSMC 데이터를 파이썬 문자열(string) 자료형으로 읽어들이는 역할을 합니다. `NsmcCorpus` 클래스의 구체적 내용은 코드2와 같습니다. 이 클래스의 `get_examples` 메소드는 NSMC 데이터를 읽어들이고 `get_labels`는 NSMC 데이터의 모든 레이블 종류(긍정 1, 부정 0)를 반환하는 역할을 합니다. 

`ClassificationDataset`는 `NsmcCorpus` 클래스의 `get_examples` 메소드를 호출하는 방식으로 말뭉치를 읽어들이는데요. 따라서 `NsmcCorpus` 클래스의 `get_examples`를 자신이 가진 말뭉치에 맞게 고치면 우리가 원하는 목적을 달성할 수 있을 겁니다.

## **코드2** NsmcCorpus 클래스
{: .no_toc .text-delta }

```python
import os, csv
from ratsnlp.nlpbook.classification.corpus import ClassificationExample
class NsmcCorpus:

    def __init__(self):
        pass

    def get_examples(self, data_root_path, mode):
        data_fpath = os.path.join(data_root_path, f"ratings_{mode}.txt")
        lines = list(csv.reader(open(data_fpath, "r", encoding="utf-8"), delimiter="\t", quotechar='"'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        return len(self.get_labels())
```

자, 이제 커스텀 말뭉치 클래스를 만들어 봅시다. 예컨대 우리가 가진 학습데이터의 파일 이름이 `train.txt`이고 다음과 같이 기사 제목(문서)과 해당 기사의 카테고리(레이블) 쌍으로 구성되어 있다고 가정해 봅시다. 

```
군병원 입원한 트럼프 중증 치료제 렘데시비르 투약,국제
코로나19로 위축된 경매시장 경기도 아파트 나홀로 인기,경제
...
'4년차 추석민심' 문대통령 국정지지율 40% 후반대,정치
```

이 말뭉치를 읽어들일 수 있도록 클래스를 새로 정의한 것은 코드3입니다. `get_examples`에서 텍스트 파일을 라인(line) 단위로 읽어들인 뒤 쉼표(`,`)로 뉴스 제목과 뉴스 카테고리를 분리합니다. 이후 뉴스 제목은 `ClassificationExample`의 `text_a`에, 뉴스 카테고리는 `label`에 저장해 둡니다.

## **코드3** 커스텀 말뭉치 클래스
{: .no_toc .text-delta }

```python
import os
from ratsnlp.nlpbook.classification import ClassificationExample
class NewsCorpus:

    def __init__(self):
        pass

    def get_examples(self, data_root_path, mode):
        data_fpath = os.path.join(data_root_path, f"{mode}.txt")
        lines = open(data_fpath, "r", encoding="utf-8").readlines()
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        return ["국제", "경제", "정치"]

    @property
    def num_labels(self):
        return len(self.get_labels())
```

한편 `get_labels`은 분류 대상 레이블의 종류를 반환하는 역할을 하는 함수인데요. 코드3 예시에서는 이를 하드 코딩으로 ["국제", "경제", "정치"]라고 명시했습니다만, 말뭉치를 읽어들인 뒤 해당 말뭉치의 레이블을 전수 조사한 뒤 유니크(unique)한 레이블들만 리스트 형태로 리턴하는 방식으로 구현해도 상관 없습니다.

코드4는 코드3에서 정의한 커스텀 데이터에 전처리를 수행하는 코드입니다. 만일 평가용 데이터셋으로 `valid.txt`를 가지고 있다면 코드4에서 `mode="valid"` 인자를 주어서 `val_dataset`도 선언할 수 있습니다.

## **코드4** 커스텀 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.classification import ClassificationDataset

corpus = NewsCorpus()
train_dataset = ClassificationDataset(
	args=args,
	corpus=corpus,
	tokenizer=tokenier,
	mode="train",
)
```

---

## 피처 구축 방식 이해하기

`ClassificationDataset`은 파이토치의 데이터셋(`Dataset`) 클래스 역할을 하는 클래스입니다. 
모델이 학습할 데이터를 품고 있는 일종의 자료 창고라고 이해하면 좋을 것 같습니다. 
만약에 이번 학습에 $i$번째 문서-레이블이 필요하다고 하면 자료 창고에서 $i$번째 데이터를 꺼내 주는 기능이 핵심 역할입니다. 

코드5를 코드4와 연관지어 전체 데이터 전처리 과정이 어떻게 이뤄지는지 살펴보겠습니다.
코드4에서 `NewsCorpus`를 `ClassificationDataset` 클래스의 `corpus`로 넣었습니다.
따라서 `ClassificationDataset` 클래스는 `NewsCorpus`의 `get_examples` 메소드를 호출해 뉴스 제목과 카테고리를 `ClassificationExample` 형태로 읽어들입니다.


## **코드5** ClassificationDataset 클래스
{: .no_toc .text-delta }

```python
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from ratsnlp.nlpbook.classification.arguments import ClassificationTrainArguments
from ratsnlp.nlpbook.classification import _convert_examples_to_classification_features

class ClassificationDataset(Dataset):

    def __init__(
            self,
            args: ClassificationTrainArguments,
            tokenizer: PreTrainedTokenizer,
            corpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_classification_features,
    ):
        ...
            self.corpus = corpus
        ...
                examples = self.corpus.get_examples(corpus_path, mode)
                self.features = convert_examples_to_features_fn(
                    examples,
                    tokenizer,
                    args,
                    label_list=self.corpus.get_labels(),
                )
        ...

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()
```

`ClassificationDataset` 클래스는 이후 `_convert_examples_to_classification_features` 함수를 호출해 앞서 읽어들인 `example`을 `feature`로 변환합니다.
`convert_examples_to_classification_features`가 하는 역할은 문서-레이블을 모델이 학습할 수 있는 형태로 가공하는 것입니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 한편, 레이블 역시 정수(integer)로 바꿔주는 기능을 합니다.
이와 관련해 자세한 내용은 [4-2장 Training](https://ratsgo.github.io/nlpbook/docs/classification/train/#%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC)을 참고하면 좋을 것 같습니다.

한편 `ClassificationDataset` 클래스의 `convert_examples_to_features_fn` 인자로 기본값인 `_convert_examples_to_classification_features` 말고 다른 함수를 넣어줄 수도 있습니다.
이 경우 피처 구축은 해당 함수로 진행하게 됩니다. 단, 해당 함수의 결과물은 `List[ClassificationFeatures]` 형태여야 합니다. `ClassificationFeatures`의 구성 요소는 다음과 같습니다.

- input_ids: `List[int]`
- attention_mask: `List[int]`
- token_type_ids: `List[int]`
- label: `int`


---

## 다른 모델 사용하기

우리 책 문서 분류 튜토리얼에서는 이준범 님이 공개한 `kcbert`를 사용했습니다.
허깅페이스 라이브러리에 등록된 모델이라면 별다른 코드 수정 없이 다른 언어모델을 사용할 수 있습니다.
예컨대 `bert-base-uncased` 모델은 구글이 공개한 다국어 BERT 모델인데요.
`pretrained_model_name`에 해당 모델명을 입력하면 이 모델을 즉시 사용 가능합니다.

## **코드6** 다른 모델 사용하기
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
args = ClassificationTrainArguments(
    pretrained_model_name="bert-base-uncased",
    ...
)
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)
model = BertForSequenceClassification.from_pretrained(
    args.pretrained_model_name,
    config=pretrained_model_config,
)
```

허깅페이스에서 사용 가능한 모델 목록은 다음 링크를 참고하시면 됩니다.

- [https://huggingface.co/models](https://huggingface.co/models)

---

## 태스크 이해하기

우리 책 튜토리얼에서는 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)의 라이트닝모듈(`LightningModule`) 클래스를 상속 받아 태스크(task)를 정의합니다. 이 태스크에는 모델(model)과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다. 이와 관련된 튜토리얼 코드는 코드7과 같습니다.

## **코드7** 문서 분류 태스크 정의
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.classification import ClassificationTask
task = ClassificationTask(model, args)
```

`ClassificationTask`는 대부분의 문서 분류 태스크를 수행할 수 있도록 일반화되어 있어 말뭉치 등이 바뀌더라도 코드 수정을 별도로 할 필요가 없습니다. 다만 해당 클래스가 어떤 역할을 하고 있는지 추가 설명이 필요할 것 같습니다. 코드8은 코드7이 사용하는 `ClassificationTask` 클래스를 자세하게 나타낸 것입니다. 코드8 태스크 클래스의 주요 메소드에 관한 설명은 다음과 같습니다.

- **configure_optimizers** : 모델 학습에 필요한 옵티마이저(optimizer)와 러닝레이트 스케줄러(learning rate scheduler)를 정의합니다. 다른 옵티마이저와 스케줄러를 사용하려면 이 메소드의 내용을 고치면 됩니다.
- **training_step** : 학습(train) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_step** : 평가(validation) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.


## **코드8** 문서 분류 태스크 클래스
{: .no_toc .text-delta }

```python
from transformers import PreTrainedModel
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR
from ratsnlp.nlpbook.classification.arguments import ClassificationTrainArguments


class ClassificationTask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: ClassificationTrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: SequenceClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: SequenceClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
```

코드8의 `training_step`, `validation_step` 메소드에선  미니 배치(input)를 모델에 넣어 손실(loss), 로짓(logit) 등을 계산합니다. 모델의 최종 출력은 '입력 문장이 특정 범주일 확률'인데요. 로짓은 소프트맥스를 취하기 직전의 벡터입니다. 

로짓(`outputs.logits`)에 argmax를 취해 모델이 예측한 문서 범주를 가려내고 이로부터 정확도(accuracy)를 계산합니다. 로짓으로 예측 범주(`preds`)를 만드는 이유는 소프트맥스를 취한다고 대소 관계가 바뀌는 것은 아니니, 로짓으로 argmax를 하더라도 예측 범주가 달라지진 않기 때문입니다. 이후 손실, 정확도 등의 정보를 로그에 남긴 뒤 메소드를 종료합니다.

코드8의 `training_step`, `validation_step` 메소드는 `self.model`을 호출(call)해 손실과 로짓을 계산하는데요. `self.model`은 코드9의 `BertForSequenceClassification` 클래스를 가리킵니다. 본서에서는 허깅페이스의 [트랜스포머(transformers) 라이브러리](https://github.com/huggingface/transformers)에서 제공하는 클래스를 사용합니다. 그 핵심만 발췌한 코드는 코드9와 같습니다.

## **코드9** BertForSequenceClassification
{: .no_toc .text-delta }

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        ...

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        ...

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

코드9의 `self.bert`는 [4-1장](https://ratsgo.github.io/nlpbook/docs/classification/overview)의 BERT 모델을 가리킵니다. 빈칸 맞추기, 즉 마스크 언어모델(Masked Language Model)로 프리트레인을 이미 완료한 모델입니다. `self.dropout`와 `self.classifier`는 4-1장에서 소개한 [문서 분류 태스크 모듈](https://ratsgo.github.io/nlpbook/docs/classification/overview/#%ED%83%9C%EC%8A%A4%ED%81%AC-%EB%AA%A8%EB%93%88)이 되겠습니다. NSMC 데이터에 대해 리뷰의 감성을 최대한 잘 맞추는 방향으로 `self.bert`, `self.classifier`가 학습됩니다.

한편  코드8의 `training_step`, `validation_step` 메소드에서 `self.model`을 호출하면 `BertForSequenceClassification`의 `forward` 메소드가 실행됩니다. 다시 말해 코드8의 `training_step`, `validation_step` 메소드는 `self.model` 메소드와 짝을 지어 구현해야 한다는 이야기입니다.  


---


## 맺음말

지금까지 우리는 문서 분류 모델을 만드는 과정을 실습했습니다. 실습 대상 말뭉치가 NSMC여서 해당 말뭉치로 학습한 모델은 영화 리뷰의 극성을 분류할 수 있게 되는데요. 뉴스 본문과 카테고리(정치, 경제, 사회 등)로 짝지어진 말뭉치를 활용할 경우 뉴스 카테고리를 분류하는 모델을 만들 수 있게 됩니다. 뿐만 아니라 허깅페이스에 등록만 되어 있기만 하면 BERT보다 좋은 성능을 가진 언어 모델을 활용할 수 있게 돼 분류 성능을 더욱 끌어올릴 수 있습니다. 문서 분류 과제는 개체명 인식, 문서 생성 등 이후 소개할 과제의 기본이 되기 때문에 잘 숙지해 두시면 좋을 것 같습니다.

---