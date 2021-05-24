---
layout: default
title: Inference (1)
parent: Sentence Generation
nav_order: 3
---


# 프리트레인 마친 모델로 문장 생성하기
{: .no_toc }

프리트레인을 마친 문장 생성 모델을 인퍼런스하는 과정을 실습합니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## 이번 실습의 목표

이번 실습에서는 프리트레인을 마친 GPT 모델을 가지고 문장을 생성해 보도록 하겠습니다. [8-1장](http://ratsgo.github.io/nlpbook/docs/generation/overview/)에서 이미 살펴봤던 것처럼 GPT 모델의 프리트레인 태스크는 '다음 단어 맞추기'이기 때문에 파인튜닝을 수행하지 않고도 프리트레인을 마친 GPT 모델만으로 문장을 생성해볼 수가 있습니다. 실습 대상 모델은 SK텔레콤이 공개한 [KoGPT2](https://github.com/SKT-AI/KoGPT2)입니다.

이 튜토리얼에서 사용하는 코드를 모두 정리해 구글 코랩(colab) 노트북으로 만들어 두었습니다. 아래 링크를 클릭해 코랩 환경에서도 수행할 수 있습니다. 코랩 노트북 사용과 관한 자세한 내용은 [1-4장 개발환경 설정](https://ratsgo.github.io/nlpbook/docs/introduction/environment) 챕터를 참고하세요.

- <a href="https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/sentence_generation/deploy_colab1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


## 모델 초기화하기


우선 코드1을 실행해 의존성 있는 패키지를 우선 설치합니다. 코랩 환경에서는 명령어 맨 앞에 느낌표(!)를 붙이면 파이썬이 아닌, 배쉬 명령을 수행할 수 있습니다.

## **코드1** 의존성 패키지 설치
{: .no_toc .text-delta }
```python
!pip install ratsnlp
```

코드2를 수행해 프리트레인을 마친 KoGPT2 모델을 읽어들입니다. `model.eval()`을 수행하면 드롭아웃(dropout) 등 학습 때만 필요한 기능들을 꺼서 평가 모드로 동작하도록 해줍니다.

## **코드5** 체크포인트 로드
{: .no_toc .text-delta }
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    "skt/kogpt2-base-v2",
)
model.eval()
```

코드3을 실행하면 KoGPT2의 토크나이저를 선언할 수 있습니다.

## **코드3** 토크나이저 로드
{: .no_toc .text-delta }
```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    eos_token="</s>",
)
```

코드4를 실행하면 KoGPT2 모델에 넣을 입력값, 즉 컨텍스트(context)를 만들 수 있습니다. 토크나이저의 `encode` 메소드는 입력 문장(srting)을 토큰화한 뒤 정수(integer)로 인덱싱하는 역할을 수행합니다. `return_tensors` 인자를 "pt"로 주면 인덱싱 결과를 파이토치의 텐서(tensor) 자료형으로 반환합니다.

## **코드4** 모델 입력값 만들기
{: .no_toc .text-delta }
```python
input_ids = tokenizer.encode("안녕하세요", return_tensors="pt")
```

코드4를 수행한 뒤 `input_ids`를 확인해 보면 그 결과는 다음과 같습니다. `안녕하세요`라는 문자열이 네 개의 정수로 구성된 파이토치 텐서로 변환되었습니다.

- tensor([[25906,  8702,  7801,  8084]])

이번 튜토리얼에서는 위와 같이 입력값을 `안녕하세요`로 통일해보겠습니다. 다시 말해 `안녕하세요`를 모델에 입력해 이후 문장을 생성해보는 것입니다.


---

## 그리디 서치

[2장](http://ratsgo.gihub.io/nlpbook/docs/language_model/semantics/)에서 이미 살펴봤듯이 언어모델(Language Model)은 컨텍스트(단어 혹은 단어 시퀀스)를 입력 받아 다음 단어가 나타날 확률을 출력으로 반환합니다. 모델의 출력 확률 분포로부터 다음 토큰을 반복적으로 선택하는 과정이 바로 문장 생성 태스크가 됩니다.

하지만 문제는 특정 컨텍스트 다음에 올 단어로 무수히 많은 경우의 수가 존재한다는 것입니다. 다음 단어가 어떤 것이 되느냐에 따라 생성되는 문장의 의미가 180도 달라질 수 있습니다. 예컨대 그림3은 `그`라는 컨텍스트로부터 출발해 다음 단어로 어떤 것이 오는게 적절한지 언어모델이 예측한 결과입니다.

## **그림3** 예제
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JhcrUp8.jpg" width="400px" title="source: imgur.com" />

그림1로부터 생성 문장 후보를 추려보면 다음과 같습니다. 띄어쓰기는 이해하기 쉽도록 제가 임의로 넣었습니다. 모두 9가지입니다.

- 그 집 사
- 그 집은
- 그 집에
- 그 책이
- 그 책을
- 그 책 읽
- 그 사람에게
- 그 사람처럼
- 그 사람 누구

언어모델의 입력값은 컨텍스트, 출력값은 컨텍스트 다음에 오는 단어의 확률 분포라는 점을 감안하면 정석대로 문장을 생성하려면 위의 9가지 모든 케이스를 모델에 입력해서 다음 단어 확률분포를 계산하고 이로부터 다음 단어를 선택하는 과정을 거쳐야 하고, 또 이걸 반복해야 합니다. 

이해가 쉽도록 모델의 예측 결과를 아주 단순화해서 적었습니다만 실제로는 위의 9가지보다 훨씬 많은 경우의 수가 존재할 것입니다. 모든 경우의 수를 계산해보는 건 사실상 불가능에 가깝다는 이야기입니다.

그리디 서치(greedy search)는 이 같은 문제의 대안으로 제시되었습니다. 매순간 최선(best)를 선택해 탐색 범위를 줄여보자는 것이 핵심 아이디어입니다. 그림4와 같습니다. 

`그` 다음에 올 단어로 모델은 `책`이 0.5로 가장 높다고 예측했습니다. 그러면 다음 단어로 `책`을 선택하고 `그 책`을 모델에 입력해 다음 단어 확률분포를 계산합니다. 셋 가운데 확률값(0.4)이 가장 높은 `이`를 그 다음 단어로 선택합니다.


## **그림4** 그리디 서치
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YwqI1bg.jpg" width="400px" title="source: imgur.com" />


자 이제 실습을 해 봅시다! 코드5를 실행하면 그리디 서치를 수행합니다. 핵심 인자(argument)는 `do_sample=False`입니다. `max_length`는 생성 최대 길이이며 이보다 길거나, 짧더라도 EOS(end of sentence) 등 스페셜 토큰이 나타나면 생성을 중단합니다. `min_length`는 생성 최소 길이이며 이보다 짧은 구간에서 스페셜 토큰이 등장해 생성이 중단될 경우 해당 토큰이 나올 확률을 0으로 수정하여 문장 생성이 종료되지 않도록 강제합니다.

## **코드5** 그리디 서치
{: .no_toc .text-delta }
```python
import torch
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
    )
```

코드5의 `generated_ids`는 토큰ID 시퀀스여서 사람이 알아보기 어렵습니다. 코드6을 수행하면 토크나이저가 `generated_ids`를 문장(string)으로 변환해 줍니다.

## **코드6** 토큰ID를 문장으로 복원하기
{: .no_toc .text-delta }
```python
print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

코드6을 수행한 결과는 다음과 같습니다. 그리디 서치는 최대 확률을 내는 단어 시퀀스를 찾는 방법입니다. 이 때문에 코드5, 코드6 수행을 반복해도 결과가 바뀌진 않습니다.

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
```

---

## 빔 서치


하지만 그리디 서치도 완벽한 대안은 아닙니다. 순간의 최선이 전체의 최선이 되지 않을 수 있기 때문입니다. 빔 서치(beam search)는 빔(beam) 크기만큼의 선택지를 계산 범위에 넣습니다. 그림5는 빔 크기가 2인 빔 서치의 예시입니다.

`그` 다음에 올 단어로 모델은 `책`(0.5), `집`(0.4), `사람`(0.1) 순으로 예측했습니다. 우리는 빔 크기를 2로 설정해 두었으므로 `사람`은 제거하고 `책`과 `집`만 탐색 대상으로 남겨 둡니다. 

모델에 `그 책`을 입력해 단어 시퀀스 확률을 계산합니다. 그 계산 결과는 다음과 같습니다.

- `그 책이` : $0.5 \times 0.4 = 0.2$
- `그 책을` : $0.5 \times 0.3 = 0.15$
- `그 책 읽` : $0.5 \times 0.3 = 0.15$

모델에 `그 집`을 입력해 단어 시퀀스 확률을 계산합니다. 그 계산 결과는 다음과 같습니다.

- `그 집에` : $0.4 \times 0.7 = 0.28$
- `그 집은` : $0.4 \times 0.2 = 0.08$
- `그 집 사` : $0.4 \times 0.1 = 0.04$

우리는 빔 크기를 2로 설정해 두었으므로 위의 6가지 경우의 수에서 가장 확률이 높은 시퀀스 두 개만을 남겨둡니다. `그 집에`(0.28), `그 책이`(0.2)가 바로 그것입니다. 만일 빔 서치를 여기에서 그만둔다면 이 둘 가운데 확률값이 조금이라도 높은 `그 집에`가 최종 생성 결과가 됩니다.


## **그림5** 빔 서치
{: .no_toc .text-delta }
<img src="https://i.imgur.com/UCkkvzH.jpg" width="400px" title="source: imgur.com" />


빔 서치는 그리디 서치 대비 계산량은 많은 편입니다. 그리디 서치가 매순간 최고 확률을 내는 한 가지 경우의 수만 선택한다면 빔 서치는 빔 크기만큼의 경우의 수를 선택하기 때문입니다. 하지만 빔 서치는 그리디 서치보다 조금이라도 더 높은 확률을 내는 문장을 생성할 수 있는 가능성을 높이게 됩니다. 일례로 그림4와 그림5를 비교해서 보면 그리디 서치로 찾은 단어 시퀀스 확률은  0.2(`그 책이`)인 반면 빔 서치는 이보다 약간 높은 0.28(`그 집에`)인 것을 확인할 수 있습니다.

이제 실습을 해봅시다. 코드7을 수행하면 빔 서치를 수행합니다. 핵심 인자는 `do_sample=False`, `num_beams=3`입니다. `num_beams`는 빔 크기를 가리킵니다. `num_beams=1`로 설정한다면 매순간 최대 확률을 내는 단어만 선택한다는 뜻이 되므로 정확히 그리디 서치로 동작하게 됩니다.

## **코드7** 빔 서치
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        num_beams=3,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

코드7을 수행한 결과는 다음과 같습니다. 빔 서치는 그리디 서치와 마찬가지로 최대 확률을 내는 단어 시퀀스를 찾는 방법입니다. 이 때문에 코드7 수행을 반복해도 결과가 바뀌진 않습니다.

```
안녕하세요?"
"그렇지 않습니다."
"그렇지 않습니다."
"그렇지 않습니다."
"그렇지 않습니다."
"그렇지 않습니다."
"그렇지 않습니다."
"그
```

---

## 반복 줄이기


그리디 서치("그럼, 그건 뭐예요?")나 빔 서치("그렇지 않습니다.") 모두 특정 표현이 반복되고 있음을 확인할 수 있습니다. 코드8처럼 수행하면 토큰이 n-gram 단위로 반복될 경우 모델이 계산한 결과를 무시하고, 해당 n-gram의 등장 확률을 0으로 만들어 생성에서 배제하게 됩니다. 핵심 인자는 `no_repeat_ngram_size=3`입니다. 3개 이상의 토큰이 반복될 경우 해당 3-gram 등장 확률을 인위적으로 0으로 만듭니다.

## **코드8** 반복 줄이기 (1)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        no_repeat_ngram_size=3,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

코드8 수행 결과는 다음과 같습니다. 반복되는 n-gram 등장 확률을 인위적으로 조정할 뿐 최대 확률을 내는 단어 시퀀스를 찾는다는 본질이 바뀐게 아니므로 코드8 수행을 반복해도 생성 결과가 바뀌진 않습니다.

```
안녕하세요?"
"그럼, 그건 뭐예요?" 하고 나는 물었다.
"그건 뭐죠?" 나는 물었다.
나는 대답하지 않았다.
"그런데 왜 그걸 물어요? 그건 무슨 뜻이에요?
```

리피티션 패널티(repetition penalty)라는 방식으로 반복을 통제할 수도 있습니다. `repetition_penalty`라는 인자를 주면 됩니다. 그 값은 1.0 이상이어야 하며 클 수록 페널티가 세게 적용됩니다. 코드9는 아무 패널티를 적용하지 않는 것이 되어 그리디 서치와 동일한 효과를 냅니다. 그 결과는 다음과 같습니다.

## **코드8** 반복 줄이기 (2)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        repetition_penalty=1.0,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
```

코드9\~코드11로 갈수록 페널티가 세게 적용된 결과입니다. 점점 반복이 줄어드는 경향이 있습니다. 이 역시 여러 번 수행해도 생성 결과가 바뀌지 않습니다.

## **코드9** 반복 줄이기 (3)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        repetition_penalty=1.1,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"아니요, 저는요."
"그럼, 그건 무슨 말씀이신지요?"
"그럼, 그건 뭐예요?"
```

## **코드10** 반복 줄이기 (4)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        repetition_penalty=1.2,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요, 아저씨. 저는 지금 이 순간에도 괜찮아요."
"그래서 오늘은 제가 할 수 있는 일이 무엇인지 말해 보겠습니다."
"이제
```

## **코드11** 반복 줄이기 (5)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=False,
        min_length=10,
        max_length=50,
        repetition_penalty=1.5,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요, 아저씨. 저는 지금 이 순간에도 괜찮아요. 그리고 제가 할 수 있는 일은 아무것도 없어요.
이제 그만 돌아가고 싶어요.
제가 하는 일이 무엇
```


---

## 탑k 샘플링


지금까지 살펴본 문장 생성 방식은 모델이 출력한 다음 토큰 확률분포를 점수로 활용한 것입니다. 전체 어휘 가운데 점수가 가장 높은 토큰을 다음 토큰으로 결정하는 방식입니다. 이렇게 하면 동일한 모델에 동일한 컨텍스트를 입력하는 경우 문장 생성을 여러 번 반복해도 그 결과는 같습니다. 

샘플링(sampling)이라는 방식도 있습니다. 그림6을 보면 `그`라는 컨텍스트를 입력했을 때 모델은 다음 토큰으로 `집`(0.5), `책`(0.4), `사람`(0.1)이 그럴듯하다고 예측했습니다. 여기에서 다음 토큰을 확률적으로 선택합니다. `집`이 선택될 가능성이 50%로 제일 크지만 `사람`이 선택될 가능성도 10%로 작지만 없지 않습니다. 복권 당첨 확률이 아주 낮지만 당첨되는 사람이 나오는 것과 비슷한 이치입니다. 

그림6에서 실제 선택된 다음 토큰은 `사람`입니다. 샘플링 방식으로 다음 토큰을 선택하게 된다면 동일한 모델, 동일한 컨텍스트라 하더라도 시행 때마다 문장 생성 결과가 다를 수 있습니다.


## **그림6** 샘플링
{: .no_toc .text-delta }
<img src="https://i.imgur.com/it5cvUT.jpg" width="500px" title="source: imgur.com" />


탑k 샘플링(top-k sampling)은 모델이 예측한 다음 토큰 확률 분포 에서 확률값이 가장 높은 $k$개 토큰 가운데 하나를 다음 토큰으로 선택하는 기법입니다. 그림7은 컨텍스트를 `그`, $k$를 6으로 뒀을 때 샘플링 대상 토큰들을 나타낸 것입니다. `책`처럼 확률값이 큰 단어가 다음 단어로 뽑힐 가능성이 높지만 $k$개 안에 있는 단어라면 `의자` 같이 확률값이 낮은 케이스도 다음 토큰으로 추출될 수 있습니다. 따라서 탑k 샘플링은 매 시행 때마다 생성 결과가 달라집니다.


## **그림7** 탑k 샘플링 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9UMEQsl.jpg" width="500px" title="source: imgur.com" />


그림7에서 선택된 다음 단어가 `사람`이라고 가정해 봅시다. 그림8은 기존 컨텍스트(`그`)에 이번에 선택된 `사람`을 이어붙인 새로운 컨텍스트 `그 사람`을 모델에 입력해 다음 토큰 확률분포를 계산하고 이를 내림차순 정렬한 것입니다. 우리는 $k$를 6으로 뒀으므로 `처럼`부터 `만큼`까지의 6개가 다음 토큰 후보가 됩니다.  


## **그림8** 탑k 샘플링 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nxeDQ1n.jpg" width="500px" title="source: imgur.com" />


코드12는 탑k 샘플링을 수행하는 코드입니다. 핵심 인자는 `do_sample=True`, `top_k=50`입니다. 샘플링 방식으로 다음 단어를 선택해되 $k$를 50으로 설정한다는 뜻입니다. `top_k`는 1 이상의 정수를 입력해야 합니다.


## **코드12** 탑k 샘플링 (1)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=50,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

코드12는 수행할 때마다 다른 문장이 생성됩니다. 아래는 코드12를 두 번 수행한 결과인데요. 여러분이 실행한 문장은 아래 두 개와 다른 문장일 가능성이 큽니다.

```
안녕하세요"라고 인사한 뒤 함께 내려왔다.
이들은 경찰서 방범 CCTV에 포착된 시민을 쫓아내던 중 한 남성이 '안녕하세요'라고 남긴 쪽지를 들고 달려들자 함께 올라탔다.
당시
```

```
안녕하세요?"
"뭐죠? 아니, 그게 아니라 우리한테 물어보시면 될 것 같습니다."
"그게 아니라. 그리고 그게 아니라요."
"그냥 내가 봤을 때. 우리 둘 다 내가
```

`top_k`를 1로 입력한다면 `do_sample` 인자를 `True`로 두더라도 그리디 서치와 동일한 효과를 냅니다. 코드13을 수행한 결과와  코드5, 코드6의 그리디 서치 결과가 같음을 확인할 수 있습니다.


## **코드13** 탑k 샘플링 (2)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=1,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
```


---

## 템퍼러쳐 스케일링


템퍼러처 스케일링(temperature scaling)이란 모델의 다음 토큰 확률분포에 변형을 가해 문장을 다양하게 생성하는 기법입니다. 확률분포를 변형한다는 의미는, 대소 관계의 역전 없이 분포의 모양만을 바꾼다는 걸 가리킵니다. 그림9는 그림3에서 템퍼러처 스케일링을 적용한 예시입니다.

원래(그림3)대로라면 `그` 다음 토큰 확률은 각각 `책`(0.5), `집`(0.4), `사람`(0.1)이었습니다. 템퍼러처 스케일링을 적용한 결과 그 확률이 `책`(0.75), `집`(0.23), `사람`(0.02)으로 바뀌었습니다. 순위의 변동은 없지만 원래 컸던 확률은 더 커지고, 작았던 확률은 더 작아져 확률분포의 모양이 뾰족(sharp)해졌음을 알 수 있습니다.

마찬가지로 원래(그림3)대로라면 `그 책` 다음 토큰 확률은 각각 `이`(0.4), `을`(0.3), `읽`(0.3)이었습니다. 템퍼러처 스케일링을 적용한 결과 그 확률이 `이`(0.6), `을`(0.2), `읽`(0.2)으로 바뀌었습니다. 순위의 변동은 없지만 원래 컸던 확률은 더 커지고, 작았던 확률은 더 작아져 확률분포의 모양이 뾰족(sharp)해졌음을 알 수 있습니다.


## **그림9** 템퍼러처 스케일링
{: .no_toc .text-delta }
<img src="https://i.imgur.com/2Lb8CGj.jpg" width="500px" title="source: imgur.com" />


코드14는 탑k 샘플링에 템퍼러처 스케일링을 적용한 코드입니다. 다음 단어를 선택할 때 확률값이 높은 50개 가운데에서 고르되, 그 확률값은 템퍼러처 스케일링(`temperature=0.01`)으로 변형한다는 뜻입니다. `temperature`는 0 이상의 값을 가져야 합니다. 

코드14는 `temperature`를 0.01로 설정해두었는데요. 이 값이 0에 가까울 수록 그림9처럼 확률분포 모양이 원래 대비 뾰족해 집니다. 확률분포 모양이 뾰족하다는 말은 원래 컸던 확률은 더 커지고 작았던 확률은 더 작아진다는 의미인데요. 그만큼 확률값 기준 1등 토큰이 다음 토큰으로 뽑힐 가능성이 높아진다는 이야기입니다. 코드14를 실행해보면 탑k 샘플링을 수행함에도 생성된 문장이 그리디 서치와 동일함을 확인할 수 있습니다.


## **코드14** 템퍼러처 스케일링 (1)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=50,
        temperature=0.01,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
```

한편 `temperature`를 1로 설정한다면 모델이 출력한 확률분포를 어떤 변형도 없이 사용한다는 의미가 됩니다. 반대로 코드15처럼 `temperature`를 1보다 큰 값을 둔다면 확률분포가 평평해집니다(uniform). 원래 컸던 확률과 작았던 확률 사이의 차이가 줄어든다는 이야기입니다. 바꿔 말하면 확률값이 작아서 기존 탑k 샘플링에선 선택되기 어려웠던 토큰들이 다음 토큰으로 선택될 수 있습니다. 그만큼 다양한 문장이 생성될 가능성이 높아지지만 생성 문장의 품질이 나빠질 수 있습니다.

## **코드15** 템퍼러처 스케일링 (2)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_k=50,
        temperature=100000000.0,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요' 같은 말이 나왔다는 것이다.
'당신도 내 말을 들으시오' 등 문구를 그대로 쓴다는 건 매우 적절하고 적절하다.
'우리'라는 단어를 쓰느냐 아니되라는 건 '우리'는 우리, 아니 우리, 아니
```

요컨대 `temperature`를 1보다 작게 하면 상대적으로 정확한 문장을, 1보다 크게 하면 상대적으로 다양한 문장을 생성할 수 있습니다. 템퍼러처 스케일링은 탑k 샘플링, 탑p 샘플링과 같이 적용해야 의미가 있습니다. 탑p 샘플링은 이어서 바로 설명합니다.


---

## 탑p 샘플링

탑p 샘플링(top-p sampling)은 확률값이 높은 순서대로 내림차순 정렬을 한 뒤 누적 확률값이 $p$ 이하인 단어들 가운데 하나를 다음 단어로 선택하는 기법입니다. 뉴클리어스 샘플링(necleus sampling)이라고도 불립니다. 확률값을 기준으로 단어들을 내림차순 정렬해 그 값이 높은 단어들을 후보로 삼는다는 점에서는 탑k 샘플링과 같지만 상위 $k$개를 후보로 삼느냐(탑k 샘플링), 누적 확률값이 $p$ 이하인 단어들을 후보로 삼느냐(탑p 샘플링)에 따라 차이가 있습니다.

그림10은 `그`라는 컨텍스트를 입력했을 때 모델이 출력한 다음 확률 분포입니다. $p$를 0.92로 설정했을 때 다음 단어 후보는 `책`부터 `회사`까지 9개, 그리고 이들의 누적 확률합은 0.94가 되는 걸 확인할 수 있습니다. $k$가 6인 탑k 샘플링(그림7)에서는 다음 단어 후보가 `책`부터 `의자`까지 6개, 그리고 이들의 누적확률합은 0.68인 걸 알 수 있습니다.

## **그림10** 탑p 샘플링 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZEQoJU2.jpg" width="500px" title="source: imgur.com" />


그림11은 `그 사람`이라는 컨텍스트를 입력했을 때 모델이 출력한 다음 확률 분포입니다. $p$를 0.92로 설정했을 때 다음 단어 후보는 `처럼`부터 `누구`까지 3개, 그리고 이들의 누적 확률합은 0.97이 되는 걸 확인할 수 있습니다. $k$가 6인 탑k 샘플링(그림8)에서는 다음 단어 후보가 `처럼`부터 `만큼`까지 6개, 그리고 이들의 누적확률합은 0.99입니다. 

## **그림11** 탑p 샘플링 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/WjpElDc.jpg" width="500px" title="source: imgur.com" />


탑p 샘플링은 누적 확률합으로 후보 단어를 취하기 때문에 누적 확률합이 일정한 반면 후보 단어 갯수는 해당 분포에 따라 달라지게 됩니다. 반대로 탑k 샘플링은 단어 갯수로 후보 단어를 취하기 때문에 후보 단어 갯수는 일정한 반면 누적 확률합은 해당 분포에 따라 달라집니다. 다만 둘 모두 확률값이 낮은 단어는 다음 단어 후보에서 제거하기 때문에 품질 높은 문장을 생성할 가능성을 높이게 됩니다.

코드16은 탑p 샘플링을 수행합니다. 핵심 인자는 `do_sample=True`, `top_p=0.92`입니다. 샘플링 방식으로 다음 단어를 선택해되 $p$를 0.92로 설정한다는 뜻입니다. `top_p`는 0\~1 사이의 실수를 입력해야 합니다.

## **코드16** 탑p 샘플링 (1)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_p=0.92,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요!
저번주에 만나볼 때 제가 꼭 한 번쯤은 뵙고 가실 수 있게 도와주신다니 정말 감사드립니다
제가 워낙 좋은 멘토가 되서 더 많이 도와주셔서 제가 항상 감사합니다.
오늘도
```

한편 `top_p`를 1.0으로 설정한다면 확률값이 낮은 단어를 전혀 배제하지 않고 다음 단어 후보로 전체 어휘를 고려한다는 의미가 됩니다. `top_p`가 0에 가까울 수록 후보 단어 수가 줄어들어 그리디 서치와 비슷해집니다. 코드17을 수행하면 탑p 샘플링을 수행함에도 코드5, 코드6의 그리디 서치 결과가 같음을 확인할 수 있습니다. 

## **코드17** 탑p 샘플링 (1)
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        top_p=0.01,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

```
안녕하세요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
"그럼, 그건 뭐예요?"
```


---

## 종합 적용하기

지금까지 설명드렸던 문장 생성 방식을 모두 종합해 적용해 봅시다. 코드17과 같습니다. 인자별 설명은 다음과 같습니다. 그리디 서치나 빔 서치는 가장 높은 확률값을 지니는 문장을 생성해주기는 하나 컨텍스트가 동일한 경우 매번 같은 문장이 나오기 때문에, 샘플링 방식을 적용하겠습니다(`do_sample=True`). 생성된 문장이 너무 짧거나 길지 않도록 하겠습니다(`min_length=10`, `max_length=50`). 반복되는 토큰을 가급적 배제하겠습니다(`repetition_penalty=1.5`, `no_repeat_ngram_size=3`). 원래 확률분포를 조금 뾰족하게 해 확률값이 높은 토큰이 살짝 더 잘 나오도록 하겠습니다(`temperature=0.9`). 탑k 샘플링과 탑p 샘플링을 동시에 적용해 확률값이 낮은 토큰들은 후보 단어에서 배제하도록 하겠습니다(`top_k=50`, `top_p=0.92`).


## **코드17** 종합 적용
{: .no_toc .text-delta }
```python
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
    )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
```

코드17을 수행한 결과 예시는 다음과 같습니다. 코드17은 수행할 때마다 다른 문장이 생성됩니다.

```
안녕하세요~
오늘도 맛있게 먹어요
아무리 좋은 음식을 먹어도 맛이 없더라구요.
그래서 더 열심히 저에게 맛있는 음식 메뉴를 알려주고 싶었어요.^^ 
오징어, 붕장어는 정말
```


---

## 참고문헌

- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)


---