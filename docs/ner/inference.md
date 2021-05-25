---
layout: default
title: Inference
parent: Named Entity Recognition
nav_order: 3
---

# 학습 마친 모델을 실전 투입하기
{: .no_toc }

학습을 마친 개체명 인식 모델을 인퍼런스(inference)하는 과정을 실습합니다. 인퍼런스란 학습을 마친 모델로 실제 과제를 수행하는 행위 혹은 그 과정을 가리킵니다. 다시 말해 모델을 개체명 인식이라는 실전에 투입하는 것입니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 이번 실습의 목표

이번 실습에서는 학습을 마친 개체명 인식 모델을 가지고 웹 서비스(web service)를 만들어보려고 합니다. 대강의 개념도는 그림1과 같습니다. 문장을 받아 답변하는 웹 서비스인데요. 문장을 토큰화한 뒤 모델 입력값으로 만들고 이를 모델에 태워 문장 내 각 토큰이 특정 개체명 태그일 확률값을 계산하게 만듭니다. 이후 약간의 후처리 과정을 거쳐 응답하게 만드는 방식입니다.

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />


---

## 1단게 코랩 노트북 초기화하기

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/named_entity_recognition/deploy_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


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
from ratsnlp.nlpbook.ner import NERDeployArguments
args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-ner",
    max_seq_length=64,
)
```

각 인자(argument)의 역할과 내용은 다음과 같습니다.

- **pretrained_model_name** : 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_model_dir** : 이전 장에서 파인튜닝한 모델의 체크포인트 저장 위치.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이.


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
from transformers import BertForTokenClassification
model = BertForTokenClassification(pretrained_model_config)
```

코드8을 수행하면 코드6에서 초기화한 BERT 모델에 코드5의 체크포인트를 읽어들이게 됩니다. 이어 코드9를 실행하면 모델이 평가 모드로 전환됩니다.

## **코드8** 체크포인트 읽기
{: .no_toc .text-delta }
```python
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
```

## **코드9** eval mode
{: .no_toc .text-delta }
```python
model.eval()
```

---

## 4단계 모델 출력값 만들고 후처리하기

개체명 인식 모델의 출력은 각 토큰이 어떤 개체명 태그에 속하는지 확률입니다. 인퍼런스를 하려면 확률값의 각 요소값이 어떤 태그에 대응하는지 정보를 알고 있어야 합니다. 이와 관련해 코드10을 실행하면 정수(integer) 인덱스를 레이블에 매핑하는 사전(dictionary)을 만듭니다.

## **코드10** 레이블 맵 작성
{: .no_toc .text-delta }
```python
labels = [label.strip() for label in open(args.downstream_model_labelmap_fpath, "r").readlines()]
id_to_label = {}
for idx, label in enumerate(labels):
  if "PER" in label:
    label = "인명"
  elif "LOC" in label:
    label = "지명"
  elif "ORG" in label:
    label = "기관명"
  elif "DAT" in label:
    label = "날짜"
  elif "TIM" in label:
    label = "시간"
  elif "DUR" in label:
    label = "기간"
  elif "MNY" in label:
    label = "통화"
  elif "PNT" in label:
    label = "비율"
  elif "NOH" in label:
    label = "기타 수량표현"
  elif "POH" in label:
    label = "기타"
  else:
    label = label
  id_to_label[idx] = label
```

코드10 수행 결과(`id_to_label`)는 다음과 같습니다.

- {0: '[CLS]', 1: '[SEP]', 2: '[PAD]', 3: '[MASK]', 4: 'O', 5: '인명', 6: '기타 수량표현', 7: '기타', 8: '기관명', 9: '날짜', 10: '지명', 11: '통화', 12: '비율', 13: '시간', 14: '기간', 15: '인명', 16: '기타 수량표현', 17: '기타', 18: '기관명', 19: '날짜', 20: '지명', 21: '통화', 22: '비율', 23: '시간', 24: '기간'}

코드11은 인퍼런스 과정을 정의한 함수입니다. 문장(sentence)을 입력받아 토큰화를 수행한 뒤 `input_ids`, `attention_mask`, `token_type_ids`를 만듭니다. 이들 입력값을 파이토치 텐서(tensor) 자료형으로 변환한 뒤 모델에 입력합니다. 모델 출력값(`outputs.logits`)은 토큰 각각에 대해 반환되며 소프트맥스 함수 적용 이전의 로짓(logit)입니다. 여기에 소프트맥스 함수를 써서 해당 토큰이 특정 개체명 태그일 확률 형태로 바꿉니다. 마지막으로 모델 출력을 약간 후처리하여 예측 확률의 최댓값이 특정 태그의 위치일 경우 해당 태그가 출력되도록 `pred` 값을 만듭니다.


## **코드11** inference
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
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
    return {
        "sentence": sentence,
        "result": result,
    }
```

---


## 5단계 웹 서비스 시작하기

코드11에서 정의한 인퍼런스 함수(`inference_fn`)을 가지고 코드12를 실행하면 웹 서비스를 띄울 수 있습니다. 파이썬 플라스크(flask)를 활용한 앱입니다.

## **코드12** 웹 서비스
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.ner import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

코드12를 실행하면 그림2처럼 뜨는데요. 웹 브라우저로 `http://f58617dfc683.ngrok.io`에 접속하면 그림3 같은 화면을 만날 수 있습니다. 단 실행할 때마다 이 주소가 변동하니 실제 접속할 때는 직접 코드12를 실행해 당시 출력된 주소로 접근해야 합니다.

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/OOmMxyA.png" width="500px" title="source: imgur.com" />

## **그림3** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/iVw3IL6.png" width="500px" title="source: imgur.com" />


---