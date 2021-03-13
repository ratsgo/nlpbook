---
layout: default
title: Inference
parent: Document Classification
nav_order: 3
---

# Inference
{: .no_toc }

학습을 마친 문서 분류 모델을 인퍼런스하는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 코랩 노트북

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/document_classification/deploy_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---


## 이번 실습의 목표

이번 실습에서는 학습을 마친 문서 분류 모델을 가지고 웹 서비스(web service)를 만들어보려고 합니다. 대강의 개념도는 그림1과 같습니다. 쿼리를 받아 답변하는 웹 서비스인데요. 쿼리를 토큰화한 뒤 모델 입력값으로 만들고 이를 모델에 태워 확률값을 계산하게 만듭니다. 이후 약간의 후처리 과정을 거쳐 답변(answer)을 하게 만드는 방식입니다.

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />

우리는 이전 장에서 문서 분류 모델을 학습했으므로 그림1에서 모델의 출력은 쿼리가 특정 범주일 확률이 됩니다.

---

## 환경 설정

이전 장에서 학습한 모델의 체크포인트는 구글 드라이브에 저장해 두었으므로 코드1을 실행해 코랩 노트북과 자신 구글 드라이브를 연동합니다.

## **코드1** 구글드라이브 연동
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

코드2를 실행하면 각종 설정을 할 수 있습니다.

## **코드2** 인퍼런스 설정
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
args = nlpbook.DeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_checkpoint_path="/gdrive/My Drive/nlpbook/checkpoint-cls/_ckpt_epoch_0.ckpt",
    downstream_task_name="document-doc_cls",
    max_seq_length=128,
)
```

각 인자(argument)의 역할과 내용은 다음과 같습니다.

- **pretrained_model_name** : 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_model_checkpoint_path** : 이전 장에서 파인튜닝한 모델의 체크포인트 저장 위치.
- **downstream_task_name** : 다운스트림 태스크의 이름.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이. 아무 것도 입력하지 않으면 128입니다.


---


## 토크나이저 및 모델 로딩

코드3을 실행하면 토크나이저를 초기화할 수 있습니다.

## **코드3** 토크나이저 로드
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```

코드4를 실행하면 이전 장에서 파인튜닝한 모델의 체크포인트를 읽어들입니다.

## **코드4** 체크포인트 로드
{: .no_toc .text-delta }
```python
import torch
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_path,
    map_location=torch.device("cpu")
)
```

코드5를 수행하면 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델의 설정 값들을 읽어들일 수 있습니다. 이어 코드6을 실행하면 해당 설정값대로 BERT 모델을 초기화합니다.

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

코드7을 수행하면 코드6에서 초기화한 BERT 모델에 코드4의 체크포인트를 읽어들이게 됩니다. 이어 코드8을 실행하면 모델이 평가 모드로 전환됩니다.

## **코드7** 체크포인트 읽기
{: .no_toc .text-delta }
```python
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
```

## **코드8** eval mode
{: .no_toc .text-delta }
```python
model.eval()
```

---

## 인퍼런스 과정 정의

코드9는 인퍼런스 과정을 정의한 함수입니다. 문장(sentence)을 입력받아 토큰화를 수행한 뒤 `input_ids` 따위의 입력값으로 만듭니다. 이들 입력값을 파이토치 텐서(tensor) 자료형으로 변환한 뒤 모델에 입력합니다. 모델 출력값는 소프트맥스 함수 적용 이전의 로짓(logit) 형태인데요. 여기에 소프트맥스 함수를 써서 모델 출력을 [`부정`일 확률, `긍정`일 확률] 형태의 확률 형태로 바꿉니다.

마지막으로 모델 출력을 약간 후처리하여 예측 확률의 최댓값이 `부정` 위치일 경우 해당 문장이 부정(positive), 반대의 경우 긍정(positive)이 되도록 `pred` 값을 만듭니다.

## **코드9** inference
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

한편 `positive_width`, `negative_width`는 아래 영상1의 긍/부정 막대 길이 조정을 위한 것으로 크게 신경쓰지 않아도 됩니다.

---


## 웹 서비스 런칭

코드9에서 정의한 인퍼런스 함수(`inference_fn`)을 가지고 코드10을 실행하면 웹 서비스를 띄울 수 있습니다. 파이썬 플라스크(flask)를 활용한 앱입니다.

## **코드10** 웹 서비스
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

코드10을 실행하면 그림2처럼 뜨는데요. 웹 브라우저로 `http://daf9739bca4e.ngrok.io`에 접속하면 영상1 같은 화면을 만날 수 있습니다. 단 실행할 때마다 이 주소가 변동하니 실제 접속할 때는 직접 코드10을 실행해 당시 출력된 주소로 접근해야 합니다.

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DnRSS7E.png" width="500px" title="source: imgur.com" />

## **영상1** colab에서 띄운 예시
{: .no_toc .text-delta }
<video autoplay controls loop muted preload="auto" width="500">
  <source src=" https://drive.google.com/uc?export=download&id=1k2wBsFRt2hHmzALMcBpT94R356NFAS57" type="video/mp4">
</video>


---