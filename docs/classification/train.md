---
layout: default
title: Training
parent: Document Classification
nav_order: 3
---

# Training
{: .no_toc }

모델 및 학습 과정을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 모델 로딩

## **코드1** 의존성 패키지 읽기
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertForSequenceClassification
```

## **코드2** BERT 설정 읽기
{: .no_toc .text-delta }
```python
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)
```

```
09/25/2020 06:23:00 - INFO - transformers.configuration_utils -   Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 300,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 30000
}
```

## **코드3** BERT 모델 읽기
{: .no_toc .text-delta }
```python
model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
```


---

## 학습 준비

## **그림1** Task의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="500px" title="source: imgur.com" />


## **코드4** Task 정의
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import ClassificationTask
task = ClassificationTask(model, args)
```

## **그림1** Trainer의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tBfMqq4.jpg" width="500px" title="source: imgur.com" />


## **코드5** Trainer 정의
{: .no_toc .text-delta }
```python
_, trainer = nlpbook.get_trainer(args)
```


```
09/25/2020 06:23:17 - INFO - lightning -   GPU available: True, used: True
09/25/2020 06:23:17 - INFO - lightning -   TPU available: False, using: 0 TPU cores
09/25/2020 06:23:17 - INFO - lightning -   CUDA_VISIBLE_DEVICES: [0]
```

---

## 학습

## **코드6** 학습 개시
{: .no_toc .text-delta }
```python
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

## **그림1** 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Igrcjjx.png" width="500px" title="source: imgur.com" />

----