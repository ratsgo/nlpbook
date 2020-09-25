---
layout: default
title: Inference
parent: Document Classification
nav_order: 4
---

# Inference
{: .no_toc }

학습을 마친 모델을 인퍼런스하고 플라스크 등 웹서비스하는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 개요

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />

---

## 환경 설정

## **코드1** 구글드라이브 연동
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

## **코드2** 모델 설정
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
args = nlpbook.DeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_checkpoint_path="/gdrive/My Drive/nlpbook/checkpoint-cls/_ckpt_epoch_0.ckpt",
    downstream_task_name="document-classification",
    max_seq_length=128,
)
```


---


## 토크나이저 불러들이기


## **코드3** 토크나이저 로드
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```

---

## 모델 로딩


## **코드4** 체크포인트 로드
{: .no_toc .text-delta }
```python
import torch
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_path,
    map_location=torch.device("cpu")
)
```

## **코드5** BERT 설정 로드
{: .no_toc .text-delta }
```python
from transformers import BertConfig
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
```

## **코드6** BERT 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification(pretrained_model_config)
```

## **코드7** 체크포인트 읽기
{: .no_toc .text-delta }
```python
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

```

## **코드7** eval mode
{: .no_toc .text-delta }
```python
model.eval()
```

---

## inference 과정 정의

## **코드8** inference
{: .no_toc .text-delta }
```python
def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        logits, = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
    return {
        'sentence': sentence,
        'prediction': pred,
        'positive_data': f"긍정 {positive_prob}",
        'negative_data': f"부정 {negative_prob}",
        'positive_width': f"{positive_prob * 100}%",
        'negative_width': f"{negative_prob * 100}%",
    }
```

---


## web service 띄우기

## **코드9** web service
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DnRSS7E.png" width="500px" title="source: imgur.com" />

## **영상1** colab에서 띄운 예시
{: .no_toc .text-delta }
<video autoplay controls loop muted preload="auto" width="500">
  <source src=" https://drive.google.com/uc?export=download&id=1k2wBsFRt2hHmzALMcBpT94R356NFAS57" type="video/mp4">
</video>


---
