---
layout: default
title: Inference (2)
parent: Sentence Generation
nav_order: 3
---


# 파인튜닝 마친 모델로 문장 생성하기
{: .no_toc }

파인튜닝을 마친 문장 생성 모델을 인퍼런스하는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 이번 실습의 목표

이번 실습에서는 SK텔레콤이 공개한 [KoGPT2 모델](https://github.com/SKT-AI/KoGPT2)을 [NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)로 파인튜닝한 모델을 인퍼런스합니다. 대강의 개념도는 그림1과 같습니다. 지문과 질문을 받아 답변하는 웹 서비스인데요. 지문과 질문을 각각 토큰화한 뒤 모델 입력값으로 만들고 이를 모델에 태워 지문에서 정답이 어떤 위치에 나타나는지 확률값을 계산하게 만듭니다. 이후 약간의 후처리 과정을 거쳐 응답하게 만드는 방식입니다.

## **그림1** web service의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I4lGm3J.jpg" width="500px" title="source: imgur.com" />


---

## 1단계 코랩 노트북 초기화하기

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/sentence_generation/deploy_colab2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 이 튜토리얼에서는 하드웨어 가속기가 따로 필요 없습니다. 그림1과 같이 코랩 화면의 메뉴 탭에서 런타임 > 런타임 유형 변경을 클릭합니다. 이후 그림2의 화면에서 `None`을 선택합니다.

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
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
from ratsnlp.nlpbook.generation import GenerationDeployArguments
args = GenerationDeployArguments(
    pretrained_model_name="skt/kogpt2-base-v2",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-generation",
)
```

각 인자(argument)의 역할과 내용은 다음과 같습니다.

- **pretrained_model_name** : 이전 장에서 파인튜닝한 모델이 사용한 프리트레인 마친 언어모델 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_model_dir** : 이전 장에서 파인튜닝한 모델의 체크포인트 저장 위치. 이 인자에 `None`으로 입력하면 파인튜닝한 모델 대신 SK텔레콤이 공개한 KoGPT2(`skt/kogpt2-base-v2`)를 인퍼런스합니다.

---


## 3단계 토크나이저 및 모델 불러오기

코드4를 실행하면 토크나이저를 초기화할 수 있습니다.

## **코드4** 토크나이저 로드
{: .no_toc .text-delta }
```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)
```


코드5를 실행하면 모델을 불러올 수 있습니다. 코드3처럼 `downstream_model_checkpoint_path` 인자에 파인튜닝한 모델 체크포인트 위치를 입력했다면 파인튜닝을 마친 모델을 불러옵니다. 만일 `downstream_model_checkpoint_path` 인자에 `None`을 입력했다면 SK텔레콤이 공개한 KoGPT2(`skt/kogpt2-base-v2`)를 인퍼런스합니다.


## **코드5** 모델 로드
{: .no_toc .text-delta }
```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
if args.downstream_model_checkpoint_path is None:
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name,
    )
else:
    from google.colab import drive
    drive.mount('/gdrive', force_remount=True)    
    pretrained_model_config = GPT2Config.from_pretrained(
        args.pretrained_model_name,
    )
    model = GPT2LMHeadModel(pretrained_model_config)
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_path,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()
```

---

## 4단계 모델 출력값 만들고 후처리하기

코드6은 인퍼런스 과정을 정의한 함수입니다. 우선 프롬프트(입력 문장, `prompt`)을 받아 토큰화하고 인덱싱한 뒤 파이토치 텐서(tensor)로 만듭니다(`input_ids`). 이를 모델에 넣어 이후 입력 문장에 이어지는 토큰ID 시퀀스(`generated_ids`)를 생성합니다. 마지막으로 토큰ID 시퀀스를 사람이 보기 좋은 형태의 문장(string)으로 변환해 반환합니다.


## **코드6** 인퍼런스
{: .no_toc .text-delta }
```python
def inference_fn(
        prompt,
        min_length=10,
        max_length=20,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=1.0,
):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                top_p=float(top_p),
                top_k=int(top_k),
                min_length=int(min_length),
                max_length=int(max_length),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature),
           )
        generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])
    except:
        generated_sentence = """처리 중 오류가 발생했습니다. <br>
            변수의 입력 범위를 확인하세요. <br><br> 
            min_length: 1 이상의 정수 <br>
            max_length: 1 이상의 정수 <br>
            top-p: 0 이상 1 이하의 실수 <br>
            top-k: 1 이상의 정수 <br>
            repetition_penalty: 1 이상의 실수 <br>
            no_repeat_ngram_size: 1 이상의 정수 <br>
            temperature: 0 이상의 실수
            """
    return {
        'result': generated_sentence,
    }
```

한편 `top_p`, `top_k` 등은 문장 생성에 관련한 인자(argument)들인데요. 자세한 내용은 [8-3장](http://localhost:4000/nlpbook/docs/generation/inference1/)을 참고하시기 바랍니다.


---


## 5단계 웹 서비스 시작하기

코드6에서 정의한 인퍼런스 함수(`inference_fn`)을 가지고 코드7을 실행하면 웹 서비스를 띄울 수 있습니다. 파이썬 플라스크(flask)를 활용한 앱입니다.

## **코드7** 웹 서비스
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()
```

코드7을 실행하면 그림2처럼 뜨는데요. 웹 브라우저로 `http://a2894f7a6aee.ngrok.io`에 접속한 뒤 최소 길이 등 설정값과 프롬프트를 입력하면 그림3, 그림4와 같은 화면을 만날 수 있습니다. 단 실행할 때마다 이 주소가 변동하니 실제 접속할 때는 직접 코드7을 실행해 당시 출력된 주소로 접근해야 합니다.

## **그림2** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qpNBWEx.png" width="500px" title="source: imgur.com" />


이제 인퍼런스 준비가 모두 끝났습니다. 웹 브라우저로 `http://a2894f7a6aee.ngrok.io`에 접속한 뒤 다음과 같이 최소 길이 등 각종 설정값을 지정하고 프롬프트로 `긍정 아 정말`을 입력하면 나머지 문장을 모델이 완성한 결과를 볼 수 있습니다. 우리는 NSMC 데이터를 `레이블(긍정 혹은 부정) + 리뷰 문장` 형태로 가공해 파인튜닝했기 때문에 `긍정 아 정말` 뒤에 생성된 문장의 극성(polarity)은 긍정임을 확인할 수 있습니다(`이거 진짜재밌게봄`).


## **그림3** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/KpAWR5z.png" width="500px" title="source: imgur.com" />


그림4는 설정값은 동일하나 프롬프트만 다르게 준 결과입니다. `부정 아 정말` 뒤에 생성된 문장의 극성은 부정임을 알 수 있습니다(`이런 영화는 머야 이따구로 만드나요?ㅠㅠ`)


## **그림4** colab에서 띄운 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/FW1nAgj.png" width="500px" title="source: imgur.com" />


---