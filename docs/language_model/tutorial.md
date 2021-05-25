---
layout: default
title: Embedding Tutorial
parent: Language Model
nav_order: 4
---

# 단어/문장을 벡터로 변환하기
{: .no_toc }

프리트레인이 완료된 언어 모델에서 단어, 문장 수준 임베딩을 추출하는 실습을 해봅니다. 실습엔 미국 자연어 처리 기업 '허깅페이스'가 만든 [트랜스포머(transformer) 라이브러리](https://github.com/huggingface/transformers)를 사용합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 파인튜닝

이 책에서 소개하는 튜토리얼은 모두 트랜스포머(transformer) 계열 언어 모델(Language Model)을 사용합니다. 프리트레인(pretrain)을 마친 언어 모델 위에 작은 모듈을 조금 더 쌓아 태스크를 수행하는 구조입니다. 문서 분류, 개체명 인식 등 다운스트림 데이터로 프리트레인 마친 BERT와 그 위의 작은 모듈을 포함한 전체 모델을 업데이트하는 과정을 파인튜닝(fine-tuning)이라고 합니다. BERT의 출력 결과 가운데 어떤 걸 사용하느냐에 따라 두 가지 방식으로 나눠볼 수 있습니다. 


### 문장 벡터 활용 : 문서 분류 등

문서 분류를 수행하는 모델을 만든다고 하면 그림1과 같은 모양이 됩니다.

## **그림1** 문서 분류
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5lpkDEB.png" width="350px" title="source: imgur.com" />

그림1에서 노란색 박스가 바로 BERT 모델입니다. '빈칸 맞추기'로 프리트레인을 이미 마쳤습니다. [BERT](https://ratsgo.github.io/nlpbook/docs/language_model/bert_gpt/#bert)는 트랜스포머의 인코더 블록(레이어)을 여러 개 쌓은 구조입니다. 그림1에서 확인할 수 있다시피 이 블록(레이어)의 입력과 출력은 단어 시퀀스(정확히는 입력 단어에 해당하는 벡터들의 시퀀스)이며, 블록(레이어) 내에서는 입력 단어(벡터)를 두 개씩 쌍을 지어 서로의 관계를 모두 고려하는 방식으로 계산됩니다.

문장을 [워드피스(wordpiece)](https://ratsgo.github.io/nlpbook/docs/tokenization/bpe/#%EC%9B%8C%EB%93%9C%ED%94%BC%EC%8A%A4)로 토큰화한 뒤 앞뒤에 문장 시작과 끝을 알리는 스페셜 토큰 `CLS`와 `SEP`를 각각 추가한 뒤 BERT에 입력합니다. 이후 BERT 모델의 마지막 블록(레이어)의 출력 가운데 `CLS`에 해당하는 벡터를 추출합니다. 트랜스포머 인코더 블록에서는 모든 단어가 서로 영향을 끼치기 때문에 마지막 블록 `CLS` 벡터는 문장 전체(`이 영화 재미없네요`)의 의미가 벡터 하나로 응집된 것이라고 할 수 있겠습니다.

이렇게 뽑은 `CLS` 벡터에 작은 모듈을 하나 추가해, 그 출력이 미리 정해 놓은 범주(예컨대 `긍정`, `중립`, `부정`)가 될 확률이 되도록 합니다. 학습 과정에서는 BERT와 그 위에 쌓은 작은 모듈을 포함한 전체 모델의 출력이 정답 레이블과 최대한 같아지도록 모델 전체를 업데이트합니다. 이것이 **파인튜닝(fine-tuning)**입니다.

### 단어 벡터 활용 : 개체명 인식 등

문서 분류는 마지막 블록의 `CLS` 벡터만을 사용하는 반면, 개체명 인식 같은 과제에서는 마지막 블록의 모든 단어 벡터를 활용합니다. 그림2와 같습니다.

## **그림2** 개체명 인식
{: .no_toc .text-delta }
<img src="https://i.imgur.com/I0Fdtfe.png" width="350px" title="source: imgur.com" />

그림2에서도 노란색 박스가 바로 BERT 모델인데요. 이 역시 '빈칸 맞추기'로 프리트레인을 이미 마쳤습니다. 문서 분류 때와 동일한 방식으로 입력값을 만들고 BERT의 마지막 레이어까지 계산을 수행합니다. BERT 모델의 마지막 블록(레이어)의 출력은 문장 내 모든 단어에 해당하는 벡터들의 시퀀스가 됩니다.

이렇게 뽑은 단어 벡터들 위에 작은 모듈을 각각 추가해, 그 출력이 각 개체명 범주(`기관명`, `인명`, `지명` 등)가 될 확률이 되도록 합니다. 학습 과정에서는 BERT와 그 위에 쌓은 각각의 작은 모듈을 포함한 전체 모델의 출력이 정답 레이블과 최대한 같아지도록 모델 전체를 업데이트합니다.

---

## 튜토리얼

이 챕터에서는 프리트레인을 마친 BERT 모델에 문장을 입력해서 이를 벡터로 변환하는 실습을 해보도록 하겠습니다. 이러한 절차는 BERT 이외의 다른 모델들도 거의 비슷합니다.


### 실습 환경 만들기


이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭하면 코랩 환경에서 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/basic/embedding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 이 튜토리얼에서는 하드웨어 가속기가 따로 필요 없습니다. 그림1과 같이 코랩 화면의 메뉴 탭에서 런타임 > 런타임 유형 변경을 클릭합니다. 이후 그림2의 화면에서 `None`을 선택합니다.

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />


### 토크나이저 초기화

BERT 모델의 입력값을 만들려면 토크나이저부터 선언해두어야 합니다. 코드1을 실행하면 이준범 님이 허깅페이스에 등록한 `kcbert-base` 모델이 쓰는 토크나이저를 선언할 수 있습니다. 

## **코드1** 토크나이저 선언
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False,
)
```

### 모델 초기화

코드2를 수행하면 모델을 초기화할 수 있습니다. 여기서 중요한 것은 사용 대상 BERT 모델이 프리트레인할 때 썼던 토크나이저를 그대로 사용해야 벡터 변환에 문제가 없다는 점입니다. 모델과 토크나이저의 토큰화 방식이 다를 경우 모델이 엉뚱한 결과를 출력하기 때문이죠. 따라서 코드2를 실행해 모델을 선언할 때 코드1과 동일한 모델 이름을 적용합니다. 

## **코드2** 모델 선언
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertModel
pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base"
)
model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)
```

코드2의 `pretrained_model_config`에는 BERT 모델을 프리트레인할 때 설정했던 내용이 담겨 있습니다. 코랩에서 `pretrained_model_config`를 입력하면 그림1을 확인할 수 있습니다. 블록(레이어) 수는 12개, 헤드의 수는 12개, 어휘 집합의 크기는 3만개 등 정보를 확인할 수 있습니다. 

## **그림1** pretrained_model_config
{: .no_toc .text-delta }
```json
BertConfig {
  "_name_or_path": "beomi/kcbert-base",
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
  "position_embedding_type": "absolute",
  "transformers_version": "4.2.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30000
}
```

코드2의 맨 마지막 줄에서는 이같은 설정에 따라 모델 전체를 초기화한 뒤 미리 학습된 `kcbert-base` 체크포인트를 읽어들이는 역할을 합니다. 체크포인트가 로컬에 저장되어 있지 않을 경우 웹에서 다운로드하는 것까지 한번에 수행합니다.


### 입력값 만들기

코드3을 수행하면 BERT 모델의 입력값을 만들 수 있습니다. 코드3 수행 결과는 그림2와 같습니다. 두 개의 입력 문장 각각에 대해 워드피스 토큰화를 수행한 뒤 이를 토큰 인덱스로 변환한 결과가 `input_ids`입니다. BERT 모델은 문장 시작에 `CLS`, 끝에 `SEP`라는 스페셜 토큰을 추가하기 때문에 문장 두 개 모두 앞뒤에 이들 토큰에 대응하는 인덱스 `2`, `3`이 덧붙여져 있음을 볼 수 있습니다.

토큰 최대 길이(`max_length`)를 10으로 설정하고, 토큰 길이가 이보다 짧으면 최대 길이에 맞게 패딩(`0`)을 주고(`padding="max_length"`), 길면 자르는(`truncation=True`) 것으로 설정해 두었기 때문에 `input_ids`의 길이는 두 문장 모두 10인걸 확인할 수 있습니다.

## **코드3** 입력값 만들기
{: .no_toc .text-delta }
```python
sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)
```

## **그림2** Features
{: .no_toc .text-delta }
```python
{
    'input_ids': [
        [2, 19017, 8482, 3, 0, 0, 0, 0, 0, 0], 
        [2, 15830, 5, 3, 0, 0, 0, 0, 0, 0]
    ], 
    'token_type_ids': [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], 
    'attention_mask': [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ]
}
```

한편 `attention_mask`는 패딩이 아닌 토큰이 `1`, 패딩인 토큰이 `0`으로 실제 토큰이 자리하는지 아닌지를 정보를 나타냅니다. `token_type_ids`는 세그먼트(segment) 정보로 파인튜닝을 실시할 때는 모두 0을 줍니다.


### BERT에 태우기

이 책에서는 딥러닝 프레임워크로 파이토치(PyTorch)를 쓰고 있는데요. 파이토치 모델의 입력값 자료형은 파이토치에서 제공하는 텐서(tensor)여야 합니다. 따라서 코드3에서 만든 파이썬 리스트(list) 형태의 `features`를 텐서로 변환해 줍니다. 코드4와 같습니다.

## **코드4** 피처를 토치 텐서로 변환
{: .no_toc .text-delta }
```python
features = {k: torch.tensor(v) for k, v in features.items()}
```

드디어 BERT 입력값을 다 만들었습니다. 코드5를 실행해 BERT 모델을 실행합니다.

## **코드5** BERT에 태우기
{: .no_toc .text-delta }
```python
outputs = model(**features)
```

코드5 실행 결과인 `outputs`은 BERT 모델의 여러 출력 결과를 한데 묶은 것입니다. 코랩에서 `outputs.last_hidden_state`을 확인해 보면 그림3과 같은 결과를 볼 수 있습니다. 

그림3의 shape은 [2, 10, 768]입니다. 문장 두 개에 속한 각각의 토큰(최대 길이 10)을 768차원짜리의 벡터로 변환했다는 의미입니다. 이들은 입력 단어 각각에 해당하는 BERT의 마지막 레이어 출력 벡터들입니다. 이는 그림2의 노란색 실선로 표기한 단어들에 대응합니다. 그림3과 같은 결과는 개체명 인식 과제 같이 단어별로 수행해야 하는 태스크에 활용됩니다.

## **그림3** 단어 수준 임베딩
{: .no_toc .text-delta }
```
tensor([[[-0.6969, -0.8248,  1.7512,  ..., -0.3732,  0.7399,  1.1907],
         [-1.4803, -0.4398,  0.9444,  ..., -0.7405, -0.0211,  1.3064],
         [-1.4299, -0.5033, -0.2069,  ...,  0.1285, -0.2611,  1.6057],
         ...,
         [-1.4406,  0.3431,  1.4043,  ..., -0.0565,  0.8450, -0.2170],
         [-1.3625, -0.2404,  1.1757,  ...,  0.8876, -0.1054,  0.0734],
         [-1.4244,  0.1518,  1.2920,  ...,  0.0245,  0.7572,  0.0080]],
        [[ 0.9371, -1.4749,  1.7351,  ..., -0.3426,  0.8050,  0.4031],
         [ 1.6095, -1.7269,  2.7936,  ...,  0.3100, -0.4787, -1.2491],
         [ 0.4861, -0.4569,  0.5712,  ..., -0.1769,  1.1253, -0.2756],
         ...,
         [ 1.2362, -0.6181,  2.0906,  ...,  1.3677,  0.8132, -0.2742],
         [ 0.5409, -0.9652,  1.6237,  ...,  1.2395,  0.9185,  0.1782],
         [ 1.9001, -0.5859,  3.0156,  ...,  1.4967,  0.1924, -0.4448]]],
       grad_fn=<NativeLayerNormBackward>)
```

코랩에서 `outputs.last_hidden_state`을 입력해 output의 두번째 요소를 확인해 보면 그림4와 같은 결과를 확인할 수 있습니다. 

그림4의 shape은 [2, 768]입니다. 문장 두 개가 각각 768차원짜리의 벡터로 변환됐다는 의미입니다. 이들은 BERT의 마지막 레이어 `CLS` 벡터들입니다. 이는 그림1의 노란색 실선으로 표기한 `CLS`에 대응합니다. 그림4와 같은 결과는 문서 분류 과제 같이 문장 전체를 벡터 하나로 변환한 뒤 이 벡터에 어떤 계산을 수행하는 태스크에 활용됩니다.


## **그림4** 문장 수준 임베딩
{: .no_toc .text-delta }
```
tensor([[-0.1594,  0.0547,  0.1101,  ...,  0.2684,  0.1596, -0.9828],
        [-0.9221,  0.2969, -0.0110,  ...,  0.4291,  0.0311, -0.9955]],
       grad_fn=<TanhBackward>)
```

자연어를 벡터로 바꾼 결과를 임베딩(embedding) 또는 리프레젠테이션(representation)이라고 합니다. `안녕하세요`, `하이!`라는 문장은 그림3에선 단어 수준의 벡터 시퀀스로, 그림4에선 문장 수준의 벡터로 변환되었습니다. 전자를 단어 수준 임베딩(리프레젠테이션), 후자를 문장 수준 임베딩(리프레젠테이션)이라고 부릅니다. 

### 태스크 모듈 만들기

파인튜닝을 수행하기 위해서는 단어 혹은 문장 수준 임베딩 위에 태스크를 수행하기 위한 작은 모듈을 추가해야 합니다. 어떤 모듈을 사용할지는 다운스트림 태스크별로 조금씩 달라지는데요. 이와 관련해서는 4장 이후의 각 튜토리얼 파트를 참고하시면 좋을 것 같습니다. 

---
