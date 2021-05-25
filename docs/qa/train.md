---
layout: default
title: Training
parent: Question Answering
nav_order: 2
---


# 질의 응답 모델 학습하기
{: .no_toc }

질의 응답 모델의 데이터 전처리 및 학습 과정을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 1단계 코랩 노트북 초기화하기

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/question_answering/train_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


위 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

위 링크를 클릭한 후 구글 아이디로 로그인한 뒤 메뉴 탭 하단의 `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 이 다음부터는 해당 노트북을 자유롭게 수정, 실행할 수 있게 됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

한편 모델을 파인튜닝하려면 하드웨어 가속기를 사용해야 계산 속도를 높일 수 있습니다. 코랩에서는 GPU와 TPU 두 종류의 가속기를 지원합니다. 그림1과 같이 코랩 화면의 메뉴 탭에서 `런타임 > 런타임 유형 변경`을 클릭합니다. 이후 그림2와 같이 GPU 혹은 TPU 둘 중 하나를 선택합니다. 

## **그림1** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림2** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />

`None을 선택할 경우 하드웨어 가속 기능을 사용할 수 없게 돼 파인튜닝 속도가 급격히 느려집니다. 반드시 GPU 혹은 TPU 둘 중 하나를 사용하세요!
{: .fs-3 .ls-1 .code-example }


---


## 2단계 각종 설정하기

코드1을 실행해 의존성 있는 패키지를 우선 설치합니다. 코랩 환경에서는 명령어 맨 앞에 느낌표(!)를 붙이면 파이썬이 아닌, 배쉬 명령을 수행할 수 있습니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

1단계 코랩 노트북 초기화 과정에서 하드웨어 가속기로 TPU를 선택했다면 코드1에 이어 코드2를 실행하세요. TPU 관련 라이브러리들을 설치하게 됩니다. GPU를 선택했다면 코드1만 수행하고 코드2는 실행하면 안됩니다.

## **코드2** TPU 관련 패키지 설치
{: .no_toc .text-delta }
```python
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8.1-cp37-cp37m-linux_x86_64.whl
```


코랩 노트북은 일정 시간 사용하지 않으면 당시까지의 모든 결과물들이 날아갈 수 있습니다. 모델 체크포인트 등을 저장해 두기 위해 자신의 구글 드라이브를 코랩 노트북과 연결합니다. 코드3을 실행하면 됩니다.

## **코드3** 구글드라이브와 연결
{: .no_toc .text-delta }
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

이번 튜토리얼에서는 이준범 님이 공개하신 `kcbert-base` 모델을 [KorQuAD 1.0 데이터](https://korquad.github.io/KorQuad%201.0/)로 파인튜닝해볼 예정입니다. 코드4를 실행하면 관련 설정을 할 수 있습니다.

## **코드4** 모델 환경 설정
{: .no_toc .text-delta }
```python
import torch
from ratsnlp.nlpbook.qa import QATrainArguments
args = QATrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="korquad-v1",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-qa",
    max_seq_length=128,
    max_query_length=32,
    doc_stride=64,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)
```

참고로 `QATrainArguments`의 각 인자(argument)가 하는 역할과 의미는 다음과 같습니다.

- **pretrained_model_name** : 프리트레인 마친 언어모델의 이름(단 해당 모델은 허깅페이스 라이브러리에 등록되어 있어야 합니다)
- **downstream_corpus_name** : 다운스트림 데이터의 이름.
- **downstream_model_dir** : 파인튜닝된 모델의 체크포인트가 저장될 위치. `/gdrive/My Drive/nlpbook/checkpoint-qa`라고 함은 자신의 구글 드라이브의 `내 폴더` 하위의 `nlpbook/checkpoint-qa` 디렉토리에 모델 체크포인트가 저장됩니다.
- **max_seq_length** : 토큰 기준 입력 문장 최대 길이(지문, 질문 모두 포함).
- **max_query_length** : 토큰 기준 질문 최대 길이.
- **doc_stride** : 지문(context)에서 몇 개 토큰을 슬라이딩해가면서 데이터를 불릴지 결정. [4단계 데이터 전처리하기](http://ratsgo.github.io/nlpbook/docs/qa/train/#4%EB%8B%A8%EA%B3%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0) 참고.
- **batch_size** : 배치 크기. 하드웨어 가속기로 GPU를 선택(`torch.cuda.is_available() == True`)했다면 32, TPU라면(`torch.cuda.is_available() == False`) 4. 코랩 환경에서 TPU는 보통 8개 코어가 할당되는데 `batch_size`는 코어별로 적용되는 배치 크기이기 때문에 이렇게 설정해 둡니다.
- **learning_rate** : 러닝레이트. 1회 스텝에서 한 번에 얼마나 업데이트할지에 관한 크기를 가리킵니다. 이와 관련한 자세한 내용은 [3-2-2장 Technics](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics)를 참고하세요.
- **epochs** : 학습 에폭 수. 3이라면 학습 데이터를 3회 반복 학습합니다.
- **tpu_cores** : TPU 코어 수. 하드웨어 가속기로 GPU를 선택(`torch.cuda.is_available() == True`)했다면 0, TPU라면(`torch.cuda.is_available() == False`) 8.
- **seed** : 랜덤 시드 값. 아무 것도 입력하지 않으면 7입니다. 

코드5를 실행해 랜덤 시드를 설정합니다. `args`에 지정된 시드로 고정하는 역할을 합니다.

## **코드5** 랜덤 시드 고정
{: .no_toc .text-delta }
```python
from transformers import set_seed
set_seed(args.seed)
```

코드6을 실행해 각종 로그들을 출력하는 로거를 설정합니다.

## **코드6** 로거 설정
{: .no_toc .text-delta }
```python
from ratsnlp import nlpbook
nlpbook.set_logger(args)
```



---

## 3단계 말뭉치 내려받기

코드7을 실행하면 [KorQuAD 1.0 데이터](https://korquad.github.io/KorQuad%201.0/) 다운로드를 수행합니다. 데이터를 내려받는 도구로 `nlpbook`에 포함된 패키지를 사용해, corpus_name(`korquad-v1`)에 해당하는 말뭉치를 내려 받습니다.

## **코드7** 말뭉치 다운로드
{: .no_toc .text-delta }
```python
nlpbook.download_downstream_dataset(args)
```


---

## 4단계 토크나이저 준비하기

코드8을 실행해 이준범 님이 공개하신 `kcbert-base` 모델이 사용하는 토크나이저를 선언합니다.

## **코드8** 토크나이저 준비
{: .no_toc .text-delta }
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
```


---

## 5단계 데이터 전처리하기

딥러닝 모델을 학습하려면 학습데이터를 배치(batch) 단위로 지속적으로 모델에 공급해 주어야 합니다. 파이토치(PyTorch)에서는 이 역할을 데이터 로더(DataLoader)가 수행하는데요. 그 개념을 도식적으로 나타내면 그림1과 같습니다.

## **그림3** DataLoader
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bD07LbT.jpg" width="350px" title="source: imgur.com" />

코드9를 수행하면 그림3의 Dataset을 만들 수 있습니다. 여기에서 `KorQuADV1Corpus`는 KorQuAD 1.0 데이터를 읽어들이는 역할을 하고요. `QADataset`는 그림1의 DataSet 역할을 수행합니다.

## **코드9** 학습 데이터셋 구축
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import KorQuADV1Corpus, QADataset
corpus = KorQuADV1Corpus()
train_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
```

`KorQuADV1Corpus` 클래스는 `json` 포맷의 [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/) 데이터를 아래와 같은 `QAExample`로 읽어들이는 역할을 합니다. `QAExample`의 필드명과 예시는 다음과 같습니다.

- **question_text** : 바그너는 괴테의 파우스트를 읽고 무엇을 쓰고자 했는가?
- **context_text** : 1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다.
- **answer_text** : 교향곡
- **start_position_character** : 54

`QADataset` 클래스는 `KorQuADV1Corpus`와 코드8에서 선언해 둔 토크나이저를 품고 있습니다. `KorQuADV1Corpus`가 넘겨준 데이터를 모델이 학습할 수 있는 형태로 가공합니다. 다시 말해 문장을 토큰화하고 이를 인덱스로 변환하는 한편, 레이블을 만들어 주는 역할을 합니다. 

예컨대 `KorQuADV1Corpus`가 넘겨준 데이터가 위와 같은 `QAExample`이라고 할 때 `QADataset` 클래스는 이를 다음과 같은 정보로 변환합니다.

- **tokens** : [CLS] 바 ##그 ##너 ##는 괴 ##테 ##의 파 ##우스 ##트를 읽고 무엇을 쓰고 ##자 했 ##는가 ? [SEP] 18 ##3 ##9년 바 ##그 ##너 ##는 괴 ##테 ##의 파 ##우스 ##트 ##을 처음 읽고 그 내용 ##에 마음이 끌려 이를 소재 ##로 해서 하나의 교 ##향 ##곡 ##을 쓰 ##려는 뜻을 갖 ##는다 . 이 시기 바 ##그 ##너 ##는 18 ##3 ##8년 ##에 빛 독 ##촉 ##으로 산 ##전 ##수 ##전을 다 걲 ##은 상황이 ##라 좌 ##절 ##과 실망 ##에 가득 ##했 ##으며 메 ##피 ##스 ##토 ##펠 ##레스 ##를 만나는 파 ##우스 ##트 ##의 심 ##경 ##에 공감 ##했다고 한다 . 또한 파리 ##에서 아 ##브 ##네 ##크 ##의 지휘 ##로 파리 음악 ##원 관 ##현 ##악 ##단이 연 ##주 ##하는 베 ##토 [SEP]
- **start_positions** : 45
- **end_positions** : 47

`tokens`는 `[CLS] + 질문 + [SEP] + 지문 + [SEP]`의 형태입니다. 코드3에서 `max_seq_length`와 `max_query_length`를 각각 128, 64로 설정해 두었기 때문에 `tokens`의 전체 토큰 갯수, 질문 토큰 갯수가 이보다 많아지지 않도록 지문, 질문을 자릅니다. 

이후 `tokens`를 인덱싱한 결과는 아래의 `input_ids`입니다. `attention_mask`는 해당 위치의 토큰이 패딩 토큰인지(0) 아닌지(1)를 나타냅니다. 이번 예시에선 패딩 토큰이 전혀 없기 때문에 `attention_mask`가 모두 1인 걸 알 수 있습니다.

`start_positions`와 `end_positions`는 `tokens` 기준 정답의 시작/끝 위치를 나타냅니다. 이에 해당하는 토큰은 각각 `교`, `##곡`이 됩니다. 이를 `QAExample`의 answer_text(교향곡)와 비교하면 제대로 처리된 걸 확인할 수 있습니다.

한편 `token_type_ids`는 세그먼트(segment) 정보를 나타냅니다. `[CLS] + 질문 + [SEP]`에 해당하는 첫번째 세그먼트는 0, `지문 + [SEP]`에 해당하는 두번째 세그먼트는 1, 나머지 패딩에 속하는 세번째 세그먼트는 0을 줍니다. 질문과 지문의 토큰 수는 각각 17, 108개이므로 0으로 채우는 첫번째 세그먼트의 길이는 `[CLS]`와 `[SEP]`를 합쳐 19, 1로 채우는 두번째 세그먼트는 `[SEP]`를 포함해 109가 됩니다. 마지막 세그먼트(0으로 채움)의 길이는 128(`max_seq_length`) - 19(첫번째 세그먼트 길이) - 109(두번째 세그먼트 길이), 즉 0이 됩니다.

- **input_ids** : [2, 1480, 4313, 4538, 4008, 336, 4065, 4042, 3231, 23243, 19143, 13985, 12449, 9194, 4105, 3385, 9411, 32, 3, 8601, 4633, 29697, 1480, 4313, 4538, 4008, 336, 4065, 4042, 3231, 23243, 4104, 4027, 8793, 13985, 391, 9132, 4113, 10966, 11728, 12023, 14657, 4091, 8598, 16639, 341, 4573, 4771, 4027, 2139, 8478, 14416, 214, 8202, 17, 2451, 13007, 1480, 4313, 4538, 4008, 8601, 4633, 22903, 4113, 1676, 868, 4913, 7965, 1789, 4203, 4110, 15031, 786, 250, 4057, 10878, 4007, 2593, 4094, 4128, 10289, 4113, 10958, 4062, 9511, 1355, 4600, 4103, 4775, 5602, 10770, 4180, 26732, 3231, 23243, 4104, 4042, 2015, 4012, 4113, 9198, 8763, 8129, 17, 10384, 23008, 7971, 2170, 4408, 4011, 4147, 4042, 17015, 4091, 23008, 21056, 4165, 323, 4175, 4158, 11413, 2273, 4043, 7966, 1543, 4775, 3]
- **attention_mask** : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- **token_type_ids** : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- **start_positions** : 45
- **end_positions** : 47

한편 코드3에서 설정해 두었던 `doc_stride`라는 인자에 주목할 필요가 있습니다. 질의 응답 태스크를 수행할 때 지문의 길이는, BERT 모델이 처리할 수 있는 `max_seq_length`보다 긴 경우가 많습니다. 이 경우 어쩔 수 없이 지문 뒷부분을 자르는 방식으로 학습데이터를 만들 수밖에 없게 됩니다.

그런데 우리는 레이블 데이터로 `start_positions`, `end_positions`를 활용하고 있는데요. 각각 `tokens`에서 정답의 시작/끝 위치를 나타냅니다. `max_seq_length` 때문에 지문을 인위적으로 자를 경우 정답이 포함된 단락도 함께 날아갈 수 있습니다. 이 때문에 질문은 그대로 두고, 지문만 `doc_stride` 갯수만큼 슬라이딩해가면서 학습데이터를 불려서 구축하게 됩니다. 

`tokens1`을 만들고 나서 `doc_stride`를 30으로 설정해 둔다면 `tokens2`는 다음과 같이 구축하게 됩니다. 원래 지문 앞부분 30개 토큰(`18 ##3 ##9년 ... 교 ##향 ##곡 ##을`)을 없애고 뒷부분에 30개 토큰을 이어 붙입니다. 이렇게 슬라이딩함으로써 정답(`교향곡`)이 사라지게 되었는데요. 이에 지문에 정답이 없다는 취지로 `start_positions2`, `end_positions2`에 모두 0을 줍니다.

- **tokens1** : [CLS] 바 ##그 ##너 ##는 괴 ##테 ##의 파 ##우스 ##트를 읽고 무엇을 쓰고 ##자 했 ##는가 ? [SEP] 18 ##3 ##9년 바 ##그 ##너 ##는 괴 ##테 ##의 파 ##우스 ##트 ##을 처음 읽고 그 내용 ##에 마음이 끌려 이를 소재 ##로 해서 하나의 교 ##향 ##곡 ##을 쓰 ##려는 뜻을 갖 ##는다 . 이 시기 바 ##그 ##너 ##는 18 ##3 ##8년 ##에 빛 독 ##촉 ##으로 산 ##전 ##수 ##전을 다 걲 ##은 상황이 ##라 좌 ##절 ##과 실망 ##에 가득 ##했 ##으며 메 ##피 ##스 ##토 ##펠 ##레스 ##를 만나는 파 ##우스 ##트 ##의 심 ##경 ##에 공감 ##했다고 한다 . 또한 파리 ##에서 아 ##브 ##네 ##크 ##의 지휘 ##로 파리 음악 ##원 관 ##현 ##악 ##단이 연 ##주 ##하는 베 ##토 [SEP]
- **start_positions1** : 45
- **end_positions1** : 47

- **tokens2** : [CLS] 바 ##그 ##너 ##는 괴 ##테 ##의 파 ##우스 ##트를 읽고 무엇을 쓰고 ##자 했 ##는가 ? [SEP] 쓰 ##려는 뜻을 갖 ##는다 . 이 시기 바 ##그 ##너 ##는 18 ##3 ##8년 ##에 빛 독 ##촉 ##으로 산 ##전 ##수 ##전을 다 걲 ##은 상황이 ##라 좌 ##절 ##과 실망 ##에 가득 ##했 ##으며 메 ##피 ##스 ##토 ##펠 ##레스 ##를 만나는 파 ##우스 ##트 ##의 심 ##경 ##에 공감 ##했다고 한다 . 또한 파리 ##에서 아 ##브 ##네 ##크 ##의 지휘 ##로 파리 음악 ##원 관 ##현 ##악 ##단이 연 ##주 ##하는 베 ##토 ##벤 ##의 교 ##향 ##곡 9 ##번 ##을 듣고 깊은 감 ##명을 받았는데 , 이것이 이 ##듬 ##해 1월 ##에 파 ##우스 ##트 ##의 서 ##곡 ##으로 쓰여 ##진 이 [SEP]
- **start_positions2** : 0
- **end_positions2** : 0

한편 코드10을 실행하면 학습할 때 쓰이는 데이터 로더를 만들 수 있습니다. 그림1에서 Dataset 역할을 하는 `ClassificationDataset`은 학습데이터에 속한 각각의 문장을 `input_ids`, `attention_mask`, `token_type_ids`, `label` 등 네 가지로 변환한 형태로 가지고 있습니다. 그림1에서 인스턴스(instance)에 해당합니다. 데이터 로더는 Dataset이 들고 있는 전체 인스턴스 가운데 배치 크기(코드3에서 정의한 args의 batch_size)만큼을 뽑아 배치 형태로 가공하는 역할을 수행합니다. 

## **코드10** 학습 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import DataLoader, RandomSampler
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
```

코드10을 자세히 보면 `sampler`와 `collate_fn`이 눈에 띕니다. 전자는 샘플링 방식을 정의합니다. 코드10 실행으로 만들어진 데이터 로더는 배치를 만들 때 `QADataset`이 들고 있는 전체 인스턴스 가운데 batch_size 갯수만큼을 비복원(`replacement=False`) 랜덤 추출합니다. 

후자는 이렇게 뽑힌 인스턴스를 배치로 만드는 역할을 하는 함수입니다. `QADataset`는 파이썬 리스트(list) 형태의 자료형인데요. 이를 파이토치가 요구하는 자료형인 텐서(tensor) 형태로 바꾸는 등의 역할을 수행합니다.

코드11을 실행하면 평가용 데이터 로더를 구축할 수 있습니다. 학습용 데이터 로더와 달리 평가용 데이터 로더는 `SequentialSampler`를 사용하고 있음을 알 수 있습니다. 학습 때 배치 구성은 랜덤으로 하는 것이 좋은데요. 평가할 때는 평가용 데이터 전체를 사용하기 때문에 굳이 랜덤으로 구성할 이유가 없기 때문입니다.

## **코드11** 평가용 데이터 로더 구축
{: .no_toc .text-delta }
```python
from torch.utils.data import SequentialSampler
val_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)
```



---


## 6단계 모델 불러오기

코드12를 수행해 모델을 초기화합니다. `BertForQuestionAnswering`은 프리트레인을 마친 BERT 모델 위에 [7-1장](https://ratsgo.github.io/nlpbook/docs/qa/overview/)에서 설명한 질의 응답용 태스크 모듈이 덧붙여진 형태의 모델 클래스입니다.

## **코드12** 모델 초기화
{: .no_toc .text-delta }
```python
from transformers import BertConfig, BertForQuestionAnswering
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)
model = BertForQuestionAnswering.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)
```


---

## 7단계 모델 학습시키기

[파이토치 라이트닝(pytorch lightning)](https://github.com/PyTorchLightning/pytorch-lightning)이 제공하는 라이트닝(lightning) 모듈을 상속받아 태스크(task)를 정의합니다. 태스크에는 그림4와 같이 모델과 옵티마이저(optimizer), 학습 과정 등이 정의돼 있습니다.

## **그림4** Task의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ViOHWFw.jpg" width="350px" title="source: imgur.com" />

코드13을 실행하면 질의 응답용 Task를 정의할 수 있습니다. 모델은 코드11에서 준비한 모델 클래스를 사용하고요, 옵티마이저로는 아담(Adam), 러닝레이트 스케줄러로는 `ExponentialLR`을 사용합니다.

## **코드13** Task 정의
{: .no_toc .text-delta }
```python
from ratsnlp.nlpbook.qa import QATask
task = QATask(model, args)
```

코드14를 실행하면 트레이너(Trainer)를 정의할 수 있습니다. 이 트레이너는 GPU/TPU 설정, 로그 및 체크포인트 등 귀찮은 설정들을 알아서 해줍니다.

## **코드13** Trainer 정의
{: .no_toc .text-delta }
```python
trainer = nlpbook.get_trainer(args)
```

코드15처럼 트레이너의 fit 함수를 호출하면 학습이 시작됩니다. 그림5는 코랩 환경에서 학습되는 화면입니다.

## **코드15** 학습 개시
{: .no_toc .text-delta }
```python
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

## **그림5** 코랩 환경에서의 학습
{: .no_toc .text-delta }
<img src="https://i.imgur.com/eUaeOvB.png" width="500px" title="source: imgur.com" />

----
