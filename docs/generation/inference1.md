---
layout: default
title: Inference (1)
parent: Sentence Generation
nav_order: 3
---


# 프리트레인 마친 문장 생성 모델을 실전 투입하기
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


## 1단계 모델 초기화하기


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

## 2단계 그리디 서치

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

언어모델의 입력값은 컨텍스트, 출력값은 컨텍스트 다음에 오는 단어의 확률 분포라는 점을 감안하면 정색대로 문장을 생성하려면 위의 9가지 모든 케이스를 모델에 입력해서 다음 단어 확률분포를 계산하고 이로부터 다음 단어를 선택하는 과정을 거쳐야 하고, 또 이걸 반복해야 합니다. 

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

## 3단계 빔 서치


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

## 4단계 반복 줄이기


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

코드9\~코드11로 갈수록 페널티가 세게 적용된 결과입니다. 점점 반복이 줄어드는 경향이 있습니다. 이 역시 반복적으로 수행해도 생성 결과가 바뀌지 않습니다.

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

## 5단계 확률적인 방식으로 문장 생성하기


## **그림1** Sampling
{: .no_toc .text-delta }
<img src="https://i.imgur.com/it5cvUT.jpg" width="500px" title="source: imgur.com" />


## **그림1** Top-k sampling (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9UMEQsl.jpg" width="500px" title="source: imgur.com" />


## **그림1** Top-k sampling (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nxeDQ1n.jpg" width="500px" title="source: imgur.com" />


## **그림1** Sampling with Temperature Scailing
{: .no_toc .text-delta }
<img src="https://i.imgur.com/2Lb8CGj.jpg" width="500px" title="source: imgur.com" />


## **그림1** Top-p sampling (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZEQoJU2.jpg" width="500px" title="source: imgur.com" />


## **그림1** Top-p sampling (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/WjpElDc.jpg" width="500px" title="source: imgur.com" />


---

## References

- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)


---