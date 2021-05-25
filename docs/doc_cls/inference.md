---
layout: default
title: Inference
parent: Document Classification
nav_order: 3
---

# 학습 마친 모델을 실전 투입하기
{: .no_toc }

학습을 마친 문서 분류 모델을 인퍼런스(inference)하는 과정을 실습합니다. 인퍼런스란 학습을 마친 모델로 실제 과제를 수행하는 행위 혹은 그 과정을 가리킵니다. 다시 말해 모델을 문서 분류라는 실전에 투입하는 것입니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## 이번 실습의 목표

이번 실습에서는 학습을 마친 문서 분류 모델을 가지고 웹 서비스(web service)를 만들어보려고 합니다. 대강의 개념도는 그림1과 같습니다. 문장을 받아 해당 문장이 긍정인지 부정인지 답변하는 웹 서비스인데요. 문장을 토큰화한 뒤 모델 입력값으로 만들고 이를 모델에 태워 [해당 문장이 긍정일 확률, 해당 문장이 부정일 확률]을 계산하게 만듭니다. 이후 약간의 후처리 과정을 거쳐 응답하게 만드는 방식입니다.

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />

## **팁**: 웹 서비스(web service)
{: .no_toc .text-delta }
웹 서비스란 네트워크 상에서 컴퓨터들 간에 상호 작용을 하기 위해 만들어진 소프트웨어 시스템입니다. 이 장에서는 원격 사용자가 보낸 문장을 수신해 해당 문장이 긍정인지 부정인지 응답을 만들고 이 응답을 원격 사용자에게 송신하는 웹 서비스를 코랩 환경에 만듭니다.
{: .fs-3 .ls-1 .code-example }

우리는 이전 장에서 문서 분류 모델을 학습했으므로 그림1에서 모델의 출력은 쿼리가 특정 범주일 확률이 됩니다.


---

## 1단계 코랩 노트북 초기화하기

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/document_classification/deploy_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 이 튜토리얼에서는 하드웨어 가속기가 따로 필요 없습니다. 그림2와 같이 코랩 화면의 메뉴 탭에서 런타임 > 런타임 유형 변경을 클릭합니다. 이후 그림3의 화면에서 `None`을 선택합니다.

## **그림2** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림3** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />


---

## 2단계 환경 설정하기

코드1을 실행해 의존성 있는 패키지를 우선 설치합니다. 코랩 환경에서는 명령어 맨 앞에 느낌표(!)를 붙이면 파이썬이 아닌, 배쉬 명령을 수행할 수 있습니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

이전 장에서 학습한 모델의 체크포인트는 구글 드라이브에 저장해 두었으므로 코드2를 실행해 코랩 노트북과 자신 구글 드라이브를 연동합니다.

## **코드2** 구글드라이브 연동
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

코드3을 실행하면 각종 설정을 할 수 있습니다.

## **코드3** 인퍼런스 설정
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import ClassificationDeployArguments
args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-doccls",
    max_seq_length=128,
)
```

각 인자(argument)의 역할과 내용은 다음과 같습니다.

- **pretrained_model_name** : 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_model_dir** : 이전 장에서 파인튜닝한 모델의 체크포인트 저장 위치.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이. 아무 것도 입력하지 않으면 128입니다.


---


## 3단계 토크나이저 및 모델 불러오기

코드4를 실행하면 토크나이저를 초기화할 수 있습니다.

## **코드4** 토크나이저 로드
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```

코드5를 실행하면 이전 장에서 파인튜닝한 모델의 체크포인트를 읽어들입니다.

## **코드5** 체크포인트 로드
{: .no_toc .text-delta }
```python
import torch
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)
```

코드6을 수행하면 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델의 설정 값들을 읽어들일 수 있습니다. 이어 코드7을 실행하면 해당 설정값대로 BERT 모델을 초기화합니다.

## **코드6** BERT 설정 로드
{: .no_toc .text-delta }
```python
from transformers import BertConfig
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
```

## **코드7** BERT 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification(pretrained_model_config)
```

코드8을 수행하면 코드6에서 초기화한 BERT 모델에 코드5의 체크포인트(`fine_tuned_model_ckpt`)를 읽어들이게 됩니다. 이어 코드9를 실행하면 모델이 평가 모드로 전환됩니다. 드롭아웃 등 학습 때만 사용하는 기법들을 무효화하는 역할을 합니다.

## **코드8** 체크포인트 읽기
{: .no_toc .text-delta }
```python
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
```

## **코드9** 평가 모드 전환
{: .no_toc .text-delta }
```python
model.eval()
```

---

## 4단계 모델 출력값 만들고 후처리하기

코드10은 인퍼런스 과정을 정의한 함수입니다. 문장(sentence)에 토큰화를 수행한 뒤 `input_ids`, `attention_mask`, `token_type_ids`를 만듭니다. 이들 입력값을 파이토치 텐서(tensor) 자료형으로 변환한 뒤 모델에 입력합니다. 모델 출력 값(`outputs.logits`)은 소프트맥스 함수 적용 이전의 로짓(logit) 형태인데요. 여기에 소프트맥스 함수를 써서 모델 출력을 [`부정`일 확률, `긍정`일 확률] 형태의 확률 형태로 바꿉니다.

마지막으로 모델 출력을 약간 후처리하여 예측 확률의 최댓값이 `부정` 위치일 경우 해당 문장이 부정(positive), 반대의 경우 긍정(positive)이 되도록 `pred` 값을 만듭니다.

## **코드10** inference
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
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
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


## 5단계 웹 서비스 시작하기

코드10에서 정의한 인퍼런스 함수(`inference_fn`)을 가지고 코드11을 실행하면 플라스크(flask)라는 파이썬 라이브러리의 도움을 받아 웹 서비스를 띄울 수 있습니다.

## **코드11** 웹 서비스
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.classification import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

코드11을 실행하면 그림2처럼 뜨는데요. 웹 브라우저로 `http://37e862e6897f.ngrok.io`에 접속하면 그림3 같은 화면을 만날 수 있습니다. 단 실행할 때마다 이 주소가 변동하니 실제 접속할 때는 직접 코드10을 실행해 당시 출력된 주소로 접근해야 합니다.

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/KZMLYwt.png" width="500px" title="source: imgur.com" />

## **그림3** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5BWV3CS.png" width="500px" title="source: imgur.com" />


---
