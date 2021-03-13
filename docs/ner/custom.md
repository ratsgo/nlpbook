---
layout: default
title: ↗️ Customization
parent: Named Entity Recognition
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

## 피처 구축 방식 이해하기

개체명 인식의 경우 레이블 데이터 포맷이 저마다 천차만별입니다. 레이블 종류와 데이터 포맷이 달라질 경우 우리 책에서 제공하는 개체명 인식 튜토리얼용 코드를 그대로 사용하기 어렵습니다. 이에 이 글에서는 우리 책에서 제공하는 코드의 피처 구축 방식을 이해하는 데 중점을 두도록 하겠습니다. 우리 책 개체명 인식 튜토리얼의 피처 구축 관련 코드는 코드1과 같습니다.

## **코드1** NERCorpus
{: .no_toc .text-delta }

```python
# 데이터 로딩
from ratsnlp.nlpbook.ner import NERCorpus
corpus = NERCorpus(args)

# 데이터 전처리
from ratsnlp.nlpbook.ner import NERDataset
train_dataset = NERDataset(
	args=args,
	corpus=corpus,
	tokenizer=tokenizer,
	mode="train",
)
```

우리 책 튜토리얼의 학습데이터는 [한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER)에서 공개한 데이터와 자체적으로 구축한 데이터를 합친 것입니다. 그 포맷은 `원본 문서␞레이블링된 문서`입니다. 첫번째 문서의 경우 `김일성`, `한 차례`, `두 번` 각각에 대해 `PER(인명)`, `NOH(기타 수량표현)`, `POH(기타)` 표기가 된 것을 확인할 수 있습니다. 

```
이어 옆으로 움직여 김일성의 오른쪽에서 한 차례씩 두 번 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.␞이어 옆으로 움직여 <김일성:PER>의 오른쪽에서 <한 차례:NOH>씩 <두 번:NOH> 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.
제철과일리코타치즈샐러드는 직접 만든 쫀쫀한 치즈도 맛있지만, 영귤청드레싱이 상큼함을 더한다.␞<제철과일리코타치즈샐러드:POH>는 직접 만든 쫀쫀한 치즈도 맛있지만, 영귤청드레싱이 상큼함을 더한다.
정씨는 “사고 예측을 위한 빅데이터나 전자 항해 등 그동안 알지 못했던 분야에 대해 배울 수 있는 기회였다”며 “새로운 교육이 재취업에 많은 도움이 됐다”고 말했다.␞<정:PER>씨는 “사고 예측을 위한 빅데이터나 전자 항해 등 그동안 알지 못했던 분야에 대해 배울 수 있는 기회였다”며 “새로운 교육이 재취업에 많은 도움이 됐다”고 말했다.
```

코드2의 `NERCorpus` 클래스는 위의 데이터 형식을 `NERExample`로 읽어들이는 역할을 합니다. 첫번째 문서의 경우 다음 각각을 `NERExample`의 `text`와 `label`에 집어넣습니다. 이같은 작업을 전체 데이터에 대해 수행합니다.

- **text** : 이어 옆으로 움직여 김일성의 오른쪽에서 한 차례씩 두 번 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.
- **label** : 이어 옆으로 움직여 <김일성:PER>의 오른쪽에서 <한 차례:NOH>씩 <두 번:NOH> 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.


## **코드2** NERCorpus
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.ner import NERTrainArguments

class NERCorpus:

    def __init__(
            self,
            args: NERTrainArguments
    ):
        self.args = args

    def get_examples(self, data_root_path, mode):
        data_fpath = os.path.join(data_root_path, f"{mode}.txt")
        examples = []
        for line in open(data_fpath, "r", encoding="utf-8").readlines():
            text, label = line.split("\u241E")
            examples.append(NERExample(text=text, label=label))
        return examples

    def get_labels(self):
    	...
        return labels

    @property
    def num_labels(self):
        return len(self.get_labels())
```

한편 코드2의 `get_labels`는 NER 태그 종류를 리턴하는 역할을 하는 메소드인데요. 우리 책 튜토리얼 코드에서는 말뭉치를 읽어들인 뒤 해당 말뭉치의 레이블을 전수 조사한 뒤 유니크한 레이블들만 리스트 형태로 리턴하는 방식으로 구현되어 있습니다. 코드가 다소 복잡해 책에는 생략하였습니다.

코드3의 `NERDataset`은 파이토치의 데이터셋(`Dataset`) 클래스 역할을 하는 클래스입니다. 모델이 학습할 데이터를 품고 있는 일종의 자료 창고라고 이해하면 좋을 것 같습니다. 만약에 이번 학습에 $i$번째 문서-레이블이 필요하다고 하면 자료 창고에서 $i$번째 데이터를 꺼내 주는 기능이 핵심 역할입니다. 

학습을 본격적으로 시작하기 전 전처리 과정은 이렇습니다. `NERDataset` 클래스를 선언할 때 코드1에서처럼 `corpus` 인자에 `NERCorpus`를 넣었다면 `NERDataset` 클래스는 코드2 `NERCorpus`의 `get_examples` 메소드를 호출해 원본 문서와 레이블링된 문서 각각을 `NERExample` 형태로 읽어들입니다.


## **코드3** NERDataset
{: .no_toc .text-delta }

```python
from transformers import BertTokenizer
from torch.utils.data.dataset import Dataset
from ratsnlp.nlpbook.ner import NERTrainArguments, _convert_examples_to_ner_features
class NERDataset(Dataset):

    def __init__(
            self,
            args: NERTrainArguments,
            tokenizer: BertTokenizer,
            corpus: NERCorpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_ner_features,
    ):
    	...
            self.corpus = corpus
    	...
                examples = self.corpus.get_examples(corpus_fpath, mode)
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

`NERDataset` 클래스는 이후 `convert_examples_to_features_fn` 함수를 호출해 앞서 읽어들인 `example`을 `feature`로 변환합니다. `convert_examples_to_features_fn`가 하는 역할은 원본 문서와 레이블링된 문서를 모델이 학습할 수 있는 형태로 가공하는 것입니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 한편, 레이블 역시 정수(integer)로 바꿔주는 기능을 합니다.
이와 관련해 자세한 내용은 [6-2장 Training](https://ratsgo.github.io/nlpbook/docs/ner/train/)을 참고하면 좋을 것 같습니다.

한편 `NERDataset` 클래스의 `convert_examples_to_features_fn` 인자로 기본값인 `_convert_examples_to_ner_features` 말고 다른 함수를 넣어줄 수도 있습니다.
이 경우 피처 구축은 해당 함수로 진행하게 됩니다. 단, 해당 함수의 결과물은 `List[NERFeatures]` 형태여야 합니다. `NERFeatures`의 구성 요소는 다음과 같습니다.

- input_ids: `List[int]`
- attention_mask: `List[int]`
- token_type_ids: `List[int]`
- label: `int`


---

## 다른 모델 사용하기

우리 책 개체명 인식 튜토리얼에서는 이준범 님이 공개한 `kcbert`를 사용했습니다.
허깅페이스 라이브러리에 등록된 모델이라면 별다른 코드 수정 없이 다른 모델을 사용할 수 있습니다.
예컨대 `bert-base-uncased` 모델은 구글이 공개한 다국어 BERT 모델인데요.
`pretrained_model_name`에 해당 모델명을 입력하면 이 모델을 즉시 사용 가능합니다. 코드4와 같습니다.

## **코드4** 다른 모델 사용하기
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.ner import NERTrainArguments
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
args = NERTrainArguments(
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
model = BertForTokenClassification.from_pretrained(
    args.pretrained_model_name,
    config=pretrained_model_config,
)
```

허깅페이스에서 사용 가능한 모델 목록은 다음 링크를 참고하시면 됩니다.

- [https://huggingface.co/models](https://huggingface.co/models)

---

## 태스크 이해하기

우리 책 튜토리얼에서는 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning) 모듈을 상속 받아 태스크(task)를 정의합니다.
이 태스크에는 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다. 코드5와 같습니다.

## **코드5** 개체명 인식 태스크 정의
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.ner import NERTask
task = NERTask(model, args)
```

`NERTask`는 대부분의 개체명 인식 태스크를 수행할 수 있도록 일반화되어 있어 말뭉치 등이 바뀌더라도 커스터마이즈를 별도로 할 필요가 없습니다. 
다만 해당 클래스가 어떤 역할을 하고 있는지 추가 설명이 필요할 것 같습니다. 
코드6은 코드5가 사용하는 `NERTask` 클래스를 자세하게 나타낸 것입니다. 

코드6 태스크 클래스의 주요 메소드에 관한 설명은 다음과 같습니다.

- **configure_optimizers** : 모델 학습에 필요한 옵티마이저(optimizer)와 학습률(learning rate) 스케줄러(scheduler)를 정의합니다. 본서에서 제공하는 옵티마이저(`AdamW`)와 스케줄러(`CosineAnnealingWarmRestarts`)와 다른걸 사용하려면 이 메소드의 내용을 고치면 됩니다.
- **training_step** : 학습(train) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_step** : 평가(validation) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **test_step** : 테스트(test) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_epoch_end** : 평가(validation) 데이터 전체를 한번 다 계산했을 때 마무리 과정을 정의합니다.
- **test_epoch_end** : 테스트(test) 데이터 전체를 한번 다 계산했을 때 마무리 과정을 정의합니다.
- **get_progress_bar_dict** : 학습, 평가, 테스트 전반에 걸쳐 진행률 바(progress bar)에 표시할 내용을 정의합니다.

## **코드6** 개체명 인식 태스크 클래스
{: .no_toc .text-delta }

```python
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from transformers import BertPreTrainedModel
from ratsnlp.nlpbook.ner import NERTrainArguments, NER_PAD_ID
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

class NERTask(LightningModule):

    def __init__(self,
                 model: BertPreTrainedModel,
                 args: NERTrainArguments,
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
        acc = accuracy(preds, labels, ignore_index=NER_PAD_ID)
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

코드6에서 핵심적인 역할을 하는 메소드는 `step`입니다. 
미니 배치(input)를 모델에 태운 뒤 손실(loss)과 로짓(logit)을 계산합니다. 
모델의 최종 출력은 '입력 문장이 특정 범주일 확률'인데요. 
로짓은 소프트맥스를 취하기 직전의 벡터입니다. 

로짓에 argmax를 취해 모델이 예측한 문서 범주를 가려내고 이로부터 정확도(accuracy)를 계산합니다. 
로짓으로 예측 범주(`preds`)를 만드는 이유는 소프트맥스를 취한다고 대소 관계가 바뀌는 것은 아니니, 로짓으로 argmax를 하더라도 예측 범주가 달라지진 않기 때문입니다. 
이후 손실, 정확도 등의 정보를 로그에 남긴 뒤 `step` 메소드를 종료합니다.

코드6의 `step` 메소드는 `self.model`을 호출(call)해 손실과 로짓을 계산하는데요. 
`self.model`은 코드7의 `BertForTokenClassification` 클래스를 가리킵니다. 
본서에서는 허깅페이스의 [트랜스포머(transformers) 라이브러리](https://github.com/huggingface/transformers)에서 제공하는 클래스를 사용합니다. 
코드7과 같습니다.

## **코드7** BertForTokenClassification
{: .no_toc .text-delta }

```python
class BertForTokenClassification(BertPreTrainedModel):
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
```

코드7의 `self.bert`는 [4-1장](https://ratsgo.github.io/nlpbook/docs/classification/overview)의 BERT 모델을 가리킵니다. 
빈칸 맞추기, 즉 마스크 언어모델(Masked Language Model)로 프리트레인을 이미 완료한 모델입니다. 
`self.dropout`와 `self.classifier`는 4-1장에서 소개한 [문서 분류 태스크 모듈](https://ratsgo.github.io/nlpbook/docs/classification/overview/#%ED%83%9C%EC%8A%A4%ED%81%AC-%EB%AA%A8%EB%93%88)이 되겠습니다. 
개체명 인식 데이터에 대해 개체명 태그를 최대한 잘 맞추는 방향으로 `self.bert`, `self.classifier`가 학습됩니다.

한편 코드6의 `step` 메소드에서 `self.model`을 호출하면 코드7 `BertForTokenClassification`의 `forward` 메소드가 실행됩니다. 
레이블(label)이 있을 경우 `BertForTokenClassification.forward` 메소드의 출력은 `loss`, `logits`이고, `ClassificationTask.step` 메소드에서는 `loss, logits = self.model(**inputs)`로 호출함을 확인할 수 있습니다. 
다시 말해 `step` 메소드는 `self.model`과 짝을 지어 구현해야 한다는 이야기입니다. 


---