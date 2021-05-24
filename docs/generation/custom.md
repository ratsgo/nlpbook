---
layout: default
title: ↗️ Customization
parent: Sentence Generation
nav_order: 4
---


# ↗️ 나만의 문장 생성 모델 만들기
{: .no_toc }

커스텀 데이터, 토크나이저, 모델, 트레이너(trainer)로 나만의 문장 생성 모델을 만드는 과정을 소개합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 내 데이터 사용하기

우리 책 문장 생성 파인튜닝 튜토리얼은 박은정 님이 공개한 [Naver Sentiment Movie Corpus(NSMC)](https://github.com/e9t/nsmc)를 사용하고 있는데요. 나만의 문장 생성 모델 구축을 위한 첫걸음은 내가 가진 데이터를 활용하는 것일 겁니다. 이를 위해서는 말뭉치를 읽어들이는 코드에 대한 이해가 선행되어야 할텐데요. 우리 책 튜토리얼에서 NSMC 데이터를 어떻게 읽고 전처리하고 있는지 살펴보겠습니다. 코드1과 같습니다.

## **코드1** NSMC 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
# 데이터 로딩
from ratsnlp.nlpbook.generation import NsmcCorpus
corpus = NsmcCorpus()

# 데이터 전처리
from ratsnlp.nlpbook.generation import GenerationDataset
train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
```

코드1에서 선언한 `NsmcCorpus` 클래스는 CSV 파일 형식의 NSMC 데이터를 파이썬 문자열(string) 자료형으로 읽어들이는 역할을 합니다. `NsmcCorpus` 클래스의 구체적 내용은 코드2와 같습니다. 이 클래스의 `get_examples` 메소드는 `_create_examples` 메소드를 호출해 NSMC 데이터를 읽어들이는 역할을 합니다. `_create_examples`는 NSMC 데이터를 `레이블(긍정 혹은 부정)`과 `리뷰 문장`을 공백으로 연결한 텍스트로 처리하는걸 확인할 수 있습니다.

`GenerationDataset`는 `NsmcCorpus` 클래스의 `get_examples` 메소드를 호출하는 방식으로 말뭉치를 읽어들이는데요. 따라서 `NsmcCorpus` 클래스의 `get_examples`를 자신이 가진 말뭉치에 맞게 고치면 우리가 원하는 목적을 달성할 수 있을 겁니다.

## **코드2** NsmcCorpus 클래스
{: .no_toc .text-delta }

```python
import os, csv
from ratsnlp.nlpbook.generation.corpus import GenerationExample
class NsmcCorpus:

    def __init__(self):
        pass

    def _read_corpus(cls, input_file, quotechar='"'):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, review_sentence, sentiment = line
            sentiment = "긍정" if sentiment == "1" else "부정"
            text = sentiment + " " + review_sentence
            examples.append(GenerationExample(text=text))
        return examples

    def get_examples(self, data_root_path, mode):
        data_fpath = os.path.join(data_root_path, f"ratings_{mode}.txt")
        logger.info(f"loading {mode} data... LOOKING AT {data_fpath}")
        return self._create_examples(self._read_corpus(data_fpath))
```

자, 이제 커스텀 말뭉치 클래스를 만들어 봅시다. 예컨대 우리가 가진 학습데이터의 파일 이름이 `train.txt`이고 다음과 같이 기사 제목(문서)과 해당 기사의 카테고리(레이블) 쌍으로 구성되어 있다고 가정해 봅시다. 

```
군병원 입원한 트럼프 중증 치료제 렘데시비르 투약,국제
코로나19로 위축된 경매시장 경기도 아파트 나홀로 인기,경제
...
'4년차 추석민심' 문대통령 국정지지율 40% 후반대,정치
```

이 말뭉치를 읽어들일 수 있도록 클래스를 새로 정의한 것은 코드3입니다. `get_examples`에서 텍스트 파일을 라인(line) 단위로 읽어들인 뒤 쉼표(`,`)로 뉴스 제목과 뉴스 카테고리를 분리합니다. 이후 뉴스 카테고리와 뉴스 제목을 공백으로 연결한 텍스트를 `GenerationExample`의 `text`에 저장해 둡니다.


## **코드3** 커스텀 말뭉치 클래스
{: .no_toc .text-delta }

```python
import os
from ratsnlp.nlpbook.generation.corpus import GenerationExample
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
            text, label = line
            sentence = label + " " + text 
            examples.append(GenerationExample(text=sentence))
        return examples
```

코드4는 코드3에서 정의한 커스텀 데이터에 전처리를 수행하는 코드입니다. 만일 평가용 데이터셋으로 `valid.txt`를 가지고 있다면 코드4에서 `mode="valid"` 인자를 주어서 `val_dataset`도 선언할 수 있습니다.

## **코드4** 커스텀 데이터 로딩 및 전처리
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.generation import GenerationDataset

corpus = NewsCorpus()
train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenier,
    mode="train",
)
```

---

## 피처 구축 방식 이해하기

`GenerationDataset`은 파이토치의 데이터셋(`Dataset`) 클래스 역할을 하는 클래스입니다. 모델이 학습할 데이터를 품고 있는 일종의 자료 창고라고 이해하면 좋을 것 같습니다. 만약에 이번 학습에 $i$번째 문서가 필요하다고 하면 자료 창고에서 $i$번째 데이터를 꺼내 주는 기능이 핵심 역할입니다. 

코드5를 코드4와 연관지어 전체 데이터 전처리 과정이 어떻게 이뤄지는지 살펴보겠습니다. 코드4에서 `NewsCorpus`를 `GenerationDataset` 클래스의 `corpus`로 넣었습니다. 따라서 `GenerationDataset` 클래스는 `NewsCorpus`의 `get_examples` 메소드를 호출해 뉴스 제목과 카테고리를 `GenerationExample` 형태로 읽어들입니다.


## **코드5** GenerationDataset 클래스
{: .no_toc .text-delta }

```python
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from ratsnlp.nlpbook.generation.arguments import GenerationTrainArguments
from ratsnlp.nlpbook.generation.corpus import _convert_examples_to_generation_features

class GenerationDataset(Dataset):

    def __init__(
            self,
            args: GenerationTrainArguments,
            tokenizer: PreTrainedTokenizerFast,
            corpus,
            mode: Optional[str] = "train",
            convert_examples_to_features_fn=_convert_examples_to_generation_features,
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
```


`GenerationDataset` 클래스는 이후 `_convert_examples_to_generation_features` 함수를 호출해 앞서 읽어들인 `example`을 `feature`로 변환합니다.
`_convert_examples_to_generation_features`가 하는 역할은 입력 텍스트를 모델이 학습할 수 있는 형태로 가공하는 것입니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 기능을 합니다.
이와 관련해 자세한 내용은 [4-2장 Training](https://ratsgo.github.io/nlpbook/docs/classification/train/#%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC)을 참고하면 좋을 것 같습니다.

한편 `GenerationDataset` 클래스의 `convert_examples_to_features_fn` 인자로 기본값인 `_convert_examples_to_generation_features` 말고 다른 함수를 넣어줄 수도 있습니다.

이 경우 피처 구축은 해당 함수로 진행하게 됩니다. 단, 해당 함수의 결과물은 `List[GenerationFeatures]` 형태여야 합니다. `GenerationFeatures`의 구성 요소는 다음과 같습니다.

- input_ids: `List[int]`
- attention_mask: `List[int]`
- token_type_ids: `List[int]`
- labels: `List[int]`

단 labels는 input_ids와 동일하게 넣어주어야 합니다. 모델 파인튜닝시 모델 클래스(`GPT2LMHeadModel`)가 알아서 labels를 오른쪽으로 한 칸씩 움직여 input_ids의 다음 토큰을 맞추는 방식으로 가공해 학습하기 때문입니다.


---

## 다른 모델 사용하기

우리 책 문장 생성 파인튜닝 튜토리얼에서는 SK텔레콤이 공개한 `skt/kogpt2-base-v2`를 사용했습니다. 허깅페이스 라이브러리에 등록된 모델이라면 별다른 코드 수정 없이 다른 모델을 사용할 수 있습니다. 예컨대 `gpt2` 모델은 OpenAI가 공개한 영어 GPT2 모델인데요.
`pretrained_model_name`에 해당 모델명을 입력하면 이 모델을 즉시 사용 가능합니다. 코드4와 같습니다.

## **코드4** 다른 모델 사용하기
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.generation import GenerationTrainArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
args = GenerationTrainArguments(
    pretrained_model_name="gpt2",
    ...
)
tokenizer = GPT2Tokenizer.from_pretrained(
    args.pretrained_model_name,
)
model = GPT2LMHeadModel.from_pretrained(
    args.pretrained_model_name,
)
```

허깅페이스에서 사용 가능한 모델 목록은 다음 링크를 참고하시면 됩니다.

- [https://huggingface.co/models](https://huggingface.co/models)

---

## 태스크 이해하기

우리 책 튜토리얼에서는 [파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning) 모듈을 상속 받아 태스크(task)를 정의합니다. 이 태스크에는 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다. 코드5와 같습니다.

## **코드5** 질의 응답 태스크 정의
{: .no_toc .text-delta }

```python
from ratsnlp.nlpbook.generation import GenerationTask
task = GenerationTask(model, args)
```

`GenerationTask`는 대부분의 문장 생성 파인튜닝 태스크를 수행할 수 있도록 일반화되어 있어 말뭉치 등이 바뀌더라도 커스터마이즈를 별도로 할 필요가 없습니다. 다만 해당 클래스가 어떤 역할을 하고 있는지 추가 설명이 필요할 것 같습니다. 코드6은 코드5가 사용하는 `GenerationTask` 클래스를 자세하게 나타낸 것입니다. 

코드6 태스크 클래스의 주요 메소드에 관한 설명은 다음과 같습니다.

- **configure_optimizers** : 모델 학습에 필요한 옵티마이저(optimizer)와 학습률(learning rate) 스케줄러(scheduler)를 정의합니다. 본서에서 제공하는 옵티마이저(`AdamW`)와 스케줄러(`CosineAnnealingWarmRestarts`)와 다른걸 사용하려면 이 메소드의 내용을 고치면 됩니다.
- **training_step** : 학습(train) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.
- **validation_step** : 평가(validation) 과정에서 한 개의 미니배치(inputs)가 입력됐을 때 손실(loss)을 계산하는 과정을 정의합니다.

## **코드6** 문장 생성 태스크 클래스
{: .no_toc .text-delta }

```python
from transformers import PreTrainedModel
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule
from ratsnlp.nlpbook.generation.arguments import GenerationTrainArguments
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

class GenerationTask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: GenerationTrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args

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

    def training_step(self, inputs, batch_idx):
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
```

코드6의 `training_step`은 파인튜닝 학습 과정에서 미니 배치(input)를 모델에 태운 뒤 손실(loss)을 계산합니다. 이후 손실 정보를 로그에 남긴 뒤 메소드를 종료합니다. 마찬가지로 `validation_step`은 파인튜닝 평가 과정에서 미니 배치의 손실을 계산한 뒤 로그를 남깁니다.

코드6의 `training_step`, `validation_step` 메소드 둘 모두 `self.model`을 호출(call)해 손실을 계산하는데요. `self.model`은 코드7의 `GPT2LMHeadModel` 클래스를 가리킵니다. 본서에서는 허깅페이스의 [트랜스포머(transformers) 라이브러리](https://github.com/huggingface/transformers)에서 제공하는 클래스를 사용합니다. GPT2LMHeadModel 클래스의 핵심만 발췌한 코드는 코드7과 같습니다.

## **코드7** GPT2LMHeadModel
{: .no_toc .text-delta }

```python
class GPT2LMHeadModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.model_parallel = False

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
```

코드7의 `self.transformer`와 `self.lm_head`는 [8-1장](https://ratsgo.github.io/nlpbook/docs/generation/overview/)의 GPT2 모델을 가리킵니다. 다음 단어 맞추기, 즉 언어모델(Language Model)로 프리트레인을 이미 완료한 모델입니다. 파인튜닝 데이터에 대해 컨텍스트 다음 단어를 최대한 잘 맞추는 방향으로 `self.transformer`, `self.lm_head`가 학습됩니다.

한편 코드6의 `training_step`, `validation_step` 메소드에서 `self.model`을 호출하면 코드7 `GPT2LMHeadModel`의 `forward` 메소드가 실행됩니다. 다시 말해 `training_step`, `validation_step` 메소드는 `self.model` 메소드와 짝을 지어 구현해야 한다는 이야기입니다. 


---
