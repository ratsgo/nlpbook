---
layout: default
title: ↗️ Customization
parent: Document Classification
nav_order: 4
---

# ↗️ Customization
{: .no_toc }

커스텀 데이터, 토크나이저, 모델, trainer로 나만의 문서 분류 모델을 만드는 과정을 소개합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 내 데이터 사용하기

우리 책 문서 분류 튜토리얼은 박은정 님이 공개한 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)를 사용하고 있는데요. 커스텀 문서 분류 모델 구축을 위한 첫걸음은 내가 가진 데이터를 활용하는 것일 겁니다. 이를 위해서는 말뭉치를 읽어들이는 코드에 대한 이해가 선행되어야 할텐데요. 우리 책 튜토리얼에서 NSMC 데이터를 어떻게 읽고 있는지 살펴보겠습니다. 코드1과 같습니다.

## **코드1** NSMC 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.classification import NsmcCorpus
corpus = NsmcCorpus()
train_dataset = ClassificationDataset(
	args=args,
	corpus=corpus,
	tokenizer=tokenizer,
	mode="train",
)
```

코드1에서 선언한 `NsmcCorpus` 클래스는 코드2의 정의와 같습니다. 전처리를 담당하는 `ClassificationDataset`가 이 클래스의 `get_examples` 메소드를 호출하는 방식으로 말뭉치를 읽어들입니다. `get_examples`는 `_read_corpus`와 `_create_examples` 메소드를 호출하고 있음을 알 수 있네요. 따라서 이 3가지 메소드를 자신이 가진 말뭉치에 맞게 커스터마이즈하면 우리가 원하는 목적을 달성할 수 있을 겁니다.

## **코드2** NsmcCorpus 클래스
{: .no_toc .text-delta }

```python
import csv
from ratsnlp.nlpbook.classification.corpus import ClassificationExample
class NsmcCorpus:

    def __init__(self):
        pass

    def _read_corpus(cls, input_file, quotechar='"'):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, data_path, mode):
        logger.info(f"loading {mode} data... LOOKING AT {data_path}")
        return self._create_examples(self._read_corpus(data_path), mode)

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        return len(self.get_labels())
```

예컨대 우리가 가진 데이터가 다음과 같이 구성되어 있다고 가정해 봅시다. 파일명은 `ratings_{mode}.txt`여야 합니다(추후 파일명 일반화 예정). 예컨대 학습데이터(`mode=train`)를 읽어들이려면 아래 같은 말뭉치가 `ratings_{mode}.txt` 이름으로 저장되어 있어야 합니다.

```
군병원 입원한 트럼프 중증 치료제 렘데시비르 투약,국제
코로나19로 위축된 경매시장 경기도 아파트 나홀로 인기,경제
...
'4년차 추석민심' 문대통령 국정지지율 40% 후반대,정치
```

이 말뭉치를 읽어들일 수 있도록 클래스를 새로 정의한 것은 코드3입니다. `_read_corpus`에서 텍스트 파일을 라인(line) 단위로 읽어들인 뒤 쉼표(`,`)로 뉴스 제목과 뉴스 카테고리를 분리합니다. 이후 뉴스 제목은 `ClassificationExample`의 `text_a`에, 뉴스 카테고리는 `label`에 저장해 둡니다.

## **코드3** 커스텀 말뭉치 클래스
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.classification import ClassificationExample
class NewsCorpus:

    def __init__(self):
        pass

    def _read_corpus(cls, input_file):
        raw_lines = open(input_file, "r", encoding="utf-8").readlines()
        return [line.strip().split(",") for line in raw_lines]

    def _create_examples(self, lines):
        examples = []
        for line in lines:
            text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, data_path, mode):
        return self._create_examples(self._read_corpus(data_path), mode)

    def get_labels(self):
        return ["국제", "경제", "정치"]

    @property
    def num_labels(self):
        return len(self.get_labels())
```

한편 `get_labels`은 분류 대상 레이블의 종류를 리턴하는 역할을 하는 함수인데요. 코드3 예시에서는 이를 하드 코딩으로 ["국제", "경제", "정치"]라고 명시했습니다만, 말뭉치를 읽어들인 뒤 해당 말뭉치의 레이블을 전수 조사한 뒤 유니크한 레이블들만 리스트 형태로 리턴하는 방식으로 구현해도 상관 없습니다.

코드4는 코드3에서 정의한 커스텀 데이터를 실제로 사용할 수 있도록 구현하는 코드입니다. 코드4와 같은 방식으로 `val_dataset`도 선언해서 사용하면 됩니다.

## **코드4** 커스텀 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.classification import ClassificationDataset
corpus = NewsCorpus()
train_dataset = ClassificationDataset(
	args=args,
	corpus=corpus,
	tokenizer=tokenizer,
	mode="train",
)
```

---

## 다른 모델 사용하기

우리 책 문서 분류 튜토리얼에서는 이준범 님이 공개한 `kcbert`를 사용했습니다. 허깅페이스 라이브러리에 등록된 모델이라면 코드 수정 없이 `args` 변경만으로도 다른 모델을 사용할 수 있습니다. 예컨대 `bert-base-uncased` 모델은 구글이 공개한 다국어 BERT 모델인데요. `pretrained_model_name`에 해당 모델명을 입력하면 이 모델을 즉시 사용 가능합니다.

## **코드5** 다른 모델 사용하기
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
args = nlpbook.TrainArguments(
    pretrained_model_name="bert-base-uncased",
    ...
)
```

허깅페이스에서 사용 가능한 모델 목록은 다음 링크를 참고하시면 됩니다.

- [https://huggingface.co/models](https://huggingface.co/models)

---

## 태스크 이해하기

우리 책 튜토리얼에서는 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning) 모듈을 상속 받아 태스크(task)를 정의합니다. 이 태스크에는 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다. 이와 관련된 튜토리얼 코드는 코드6과 같습니다.

## **코드6** 문서 분류 태스크 정의
{: .no_toc .text-delta }

```python
from transformers import BertConfig, BertForSequenceClassification
from ratsnlp.nlpbook.classification import ClassificationTask
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)
model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
task = ClassificationTask(model, args)
```

`ClassificationTask`는 대부분의 문서 분류 태스크를 수행할 수 있도록 일반화되어 있어 말뭉치 등이 바뀌더라도 커스터마이즈를 별도로 할 필요가 없습니다. 다만 해당 클래스가 어떤 역할을 하고 있는지 추가 설명이 필요할 것 같습니다. 코드7은 코드6이 사용하는 `ClassificationTask` 클래스입니다. `ratnlp` 패키지에 포함돼 있습니다.

코드7 태스크 클래스의 주요 메소드에 관한 설명은 다음과 같습니다.

- **configure_optimizers** : 모델 학습에 필요한 옵티마이저(optimizer)와 학습률(learning rate) 스케줄러(scheduler)를 정의합니다. 본서에서 제공하는 옵티마이저(`AdamW`)와 스케줄러(`CosineAnnealingWarmRestarts`)와 다른걸 사용하려면 이 메소드의 내용을 고치면 됩니다.
- **training_step** : 학습(train) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_step** : 평가(validation) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **test_step** : 테스트(test) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_epoch_end** : 평가(validation) 데이터 전체를 한번 다 계산했을 때 마무리 과정을 정의합니다.
- **test_epoch_end** : 테스트(test) 데이터 전체를 한번 다 계산했을 때 마무리 과정을 정의합니다.
- **get_progress_bar_dict** : 학습, 평가, 테스트 전반에 걸쳐 진행률 바(progress bar)에 표시할 내용을 정의합니다.

## **코드7** 문서 분류 태스크 클래스
{: .no_toc .text-delta }

```python
from transformers import PreTrainedModel
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from ratsnlp.nlpbook.arguments import TrainArguments
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts


class ClassificationTask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.running_accuracy = TensorRunningAccum(window_length=args.stat_window_length)

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        else:
            raise NotImplementedError('Only AdamW is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, inputs, mode="train"):
        loss, logits = self.model(**inputs)
        preds = logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.running_accuracy.append(acc)
        logs = {f"{mode}_loss": loss, f"{mode}_acc": acc}
        return {"loss": loss, "log": logs}

    def training_step(self, inputs, batch_idx):
        return self.step(inputs, mode="train")

    def validation_step(self, inputs, batch_idx):
        return self.step(inputs, mode="val")

    def test_step(self, inputs, batch_idx):
        return self.step(inputs, mode="test")

    def epoch_end(self, outputs, mode="train"):
        loss_mean, acc_mean = 0, 0
        for output in outputs:
            loss_mean += output['loss']
            acc_mean += output['log'][f'{mode}_acc']
        acc_mean /= len(outputs)
        results = {
            'log': {
                f'{mode}_loss': loss_mean,
                f'{mode}_acc': acc_mean,
            },
            'progress_bar': {f'{mode}_acc': acc_mean},
        }
        return results

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, mode="val")

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs, mode="test")

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        running_train_accuracy = self.running_accuracy.mean()
        tqdm_dict = {
            'tr_loss': '{:.3f}'.format(running_train_loss.cpu().item()),
            'tr_acc': '{:.3f}'.format(running_train_accuracy.cpu().item()),
        }
        return tqdm_dict
```

코드7에서 핵심적인 역할을 하는 메소드는 `step`입니다. 미니 배치(input)를 모델에 태운 뒤 손실(loss)과 로짓(logit)을 계산합니다. 모델의 최종 출력은 '입력 문장이 특정 범주일 확률'인데요. 로짓은 소프트맥스를 취하기 직전의 벡터입니다. 

로짓에 argmax를 취해 모델이 예측한 문서 범주를 가려내고 이로부터 정확도(accuracy)를 계산합니다. 로짓으로 예측 범주(`preds`)를 만드는 이유는 소프트맥스를 취한다고 대소 관계가 바뀌는 것은 아니니, 로짓으로 argmax를 하더라도 예측 범주가 달라지진 않기 때문입니다. 이후 손실, 정확도 등의 정보를 로그에 남긴 뒤 `step` 메소드를 종료합니다.

코드7의 `step` 메소드는 `self.model`을 호출(call)해 손실과 로짓을 계산하는데요. `self.model`은 코드6의 `BertForSequenceClassification` 클래스를 가리킵니다. 본서에서는 허깅페이스의 [트랜스포머(transformers) 라이브러리](https://github.com/huggingface/transformers)에서 제공하는 클래스를 사용합니다. 코드8과 같습니다.

## **코드8** 문서 분류 모델
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

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased")
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
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
```

코드8의 `self.bert`는 [4-1장](https://ratsgo.github.io/nlpbook/docs/classification/overview)의 BERT 모델을 가리킵니다. 빈칸 맞추기, 즉 마스크 언어모델(Masked Language Model)로 프리트레인을 이미 완료한 모델입니다. `self.dropout`와 `self.classifier`는 4-1장에서 소개한 [문서 분류 태스크 모듈](https://ratsgo.github.io/nlpbook/docs/classification/overview/#%ED%83%9C%EC%8A%A4%ED%81%AC-%EB%AA%A8%EB%93%88)이 되겠습니다. NSMC 데이터에 대해 리뷰의 감성을 최대한 잘 맞추는 방향으로 `self.bert`, `self.classifier`가 학습됩니다.

한편 코드7의 `step` 메소드에서 `self.model`을 호출하면 `BertForSequenceClassification`의 `forward` 메소드가 실행됩니다. 레이블(label)이 있을 경우 `BertForSequenceClassification.forward` 메소드의 출력은 `loss`, `logits`이고, `ClassificationTask.step` 메소드에서는 `loss, logits = self.model(**inputs)`로 호출함을 확인할 수 있습니다. 다시 말해 `step` 메소드는 `self.model`과 짝을 지어 구현해야 한다는 이야기입니다. 


---
