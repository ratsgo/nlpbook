---
layout: default
title: Inference
parent: Question Answering
nav_order: 3
---


# 학습 마친 모델을 실전 투입하기
{: .no_toc }

학습을 마친 질의 응답 모델을 인퍼런스하는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 코랩 노트북

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/question_answering/deploy_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---


## 이번 실습의 목표

이번 실습에서는 학습을 마친 질의 응답 모델을 가지고 웹 서비스(web service)를 만들어보려고 합니다. 대강의 개념도는 그림1과 같습니다. 지문과 질문을 받아 답변하는 웹 서비스인데요. 지문과 질문을 각각 토큰화한 뒤 모델 입력값으로 만들고 이를 모델에 태워 지문에서 정답이 어떤 위치에 나타나는지 확률값을 계산하게 만듭니다. 이후 약간의 후처리 과정을 거쳐 응답하게 만드는 방식입니다.

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />

---

## 1단계 환경 설정하기

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
from ratsnlp.nlpbook.qa import QADeployArguments
args = QADeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_checkpoint_path="/gdrive/My Drive/nlpbook/checkpoint-qa/epoch=0.ckpt",
    max_seq_length=128,
    max_query_length=32,
    max_answer_length=30,
)
```

각 인자(argument)의 역할과 내용은 다음과 같습니다.

- **pretrained_model_name** : 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_model_checkpoint_path** : 이전 장에서 파인튜닝한 모델의 체크포인트 저장 위치.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이(지문, 질문 모두 포함).
- **max_query_length** : 토큰 기준 질문 최대 길이.
- **doc_stride** : 지문(context)에서 몇 개 토큰을 슬라이딩해가면서 데이터를 불릴지 결정. [7장 Training](http://ratsgo.github.io/nlpbook/docs/qa/train/#4%EB%8B%A8%EA%B3%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0) 참고.


---


## 2단계 토크나이저 및 모델 불러오기

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
    args.downstream_model_checkpoint_path,
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
)
```

## **코드7** BERT 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering(pretrained_model_config)
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

## 3단계 모델 출력값 만들고 후처리하기

코드10은 인퍼런스 과정을 정의한 함수입니다. 질문(question)과 지문(context)을 입력받아 토큰화를 수행한 뒤 `input_ids`, `attention_mask`, `token_type_ids`를 만듭니다. 이들 입력값을 파이토치 텐서(tensor) 자료형으로 변환한 뒤 모델에 입력합니다. 모델 출력값은 소프트맥스 함수 적용 이전의 로짓(logit) 형태입니다.

마지막으로 모델 출력을 약간 후처리하여 정답 시작과 관련한 로짓(`start_logits`)의 최댓값에 해당하는 인덱스부터, 정답 끝과 관련한 로짓(`end_logits`)의 최댓값이 위치하는 인덱스까지에 해당하는 토큰을 이어붙여 `pred_text`으로 만듭니다. 로짓에 소프트맥스(softmax)를 취하더라도 최댓값은 바뀌지 않기 때문에 소프트맥스 적용은 생략했습니다.

## **코드10** inference
{: .no_toc .text-delta }
```python
def inference_fn(question, context):
    if question and context:
        truncated_query = tokenizer.encode(
            question,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_query_length
       )
        inputs = tokenizer.encode_plus(
            text=truncated_query,
            text_pair=context,
            truncation="only_second",
            padding="max_length",
            max_length=args.max_seq_length,
            return_token_type_ids=True,
        )
        with torch.no_grad():
            start_logits, end_logits, = model(**{k: torch.tensor([v]) for k, v in inputs.items()})
            start_pred = start_logits.argmax(dim=-1).item()
            end_pred = end_logits.argmax(dim=-1).item()
            pred_text = tokenizer.decode(inputs['input_ids'][start_pred:end_pred+1])
    else:
        pred_text = ""
    return {
        'question': question,
        'context': context,
        'answer': pred_text,
    }
```

---


## 4단계 웹 서비스 시작하기

코드10에서 정의한 인퍼런스 함수(`inference_fn`)을 가지고 코드11을 실행하면 웹 서비스를 띄울 수 있습니다. 파이썬 플라스크(flask)를 활용한 앱입니다.

## **코드11** 웹 서비스
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

코드11을 실행하면 그림2처럼 뜨는데요. 웹 브라우저로 `http://17b289803156.ngrok.io`에 접속한 뒤 각각의 질문, 지문을 입력하면 그림3, 그림4, 그림5와 같은 화면을 만날 수 있습니다. 단 실행할 때마다 이 주소가 변동하니 실제 접속할 때는 직접 코드10을 실행해 당시 출력된 주소로 접근해야 합니다.

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/WuFRNOI.png" width="500px" title="source: imgur.com" />

## **그림3** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/wFl0r2t.png" width="500px" title="source: imgur.com" />

## **그림4** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/VuA85p3.png" width="500px" title="source: imgur.com" />

## **그림5** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/uu4moTQ.png" width="500px" title="source: imgur.com" />


---