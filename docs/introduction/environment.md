---
layout: default
title: Environment
parent: Introduction
nav_order: 4
---

# 개발환경 설정
{: .no_toc }

구글 코랩(colab) 등 개발환경 설정 방법을 살펴보겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## 코랩(colab)이란?

코랩(colab)은 colaboratory의 준말로 구글에서 서비스하고 있는 가상 컴퓨팅 환경입니다. Colab을 사용하면 누구나 브라우저를 통해 파이썬 코드를 작성하고 실행할 수 있습니다. GPU는 물론 TPU 학습도 할 수 있습니다. 

코랩은 구글 회원 가입만 하면 누구나 사용할 수 있는데요. 코랩을 유료 계정으로 사용한다면 좀 더 나은 GPU를 할당 받는 등의 혜택이 있지만 무료 계정이어도 딥러닝 모델 학습에 큰 장애가 없습니다. 

게다가 운영 체제 설치, 의존성 있는 소프트웨어 설치 등 환경 구축을 별도로 할 필요가 없어서, 컴퓨팅 자원이 부족하거나 개발 환경 설정에 애를 먹고 계신 분들께는 상당히 유용한 환경이 될 수 있습니다.

구글 회원가입을 마치고 로그인을 한 뒤 아래 사이트에 접속하면 코랩을 사용할 수 있습니다. 접속 화면은 그림1과 같습니다.

- https://colab.research.google.com

## **그림1** 코랩 첫 화면
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kcwS2Qb.png" width="500px" title="source: imgur.com" />

코랩은 주피터 노트북처럼 대화형으로 파이썬 명령어를 실행할 수 있습니다. 파이썬 명령어를 입력한 뒤 `ctrl+enter`를 누르면 됩니다. 그림2와 같습니다.


## **그림2** 파이썬 명령어 실행하기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/as2WsqO.png" width="300px" title="source: imgur.com" />

명령어 맨 앞에 느낌표(!)를 입력한 뒤 실행하면 해당 명령어가 배쉬 쉘(bash shell)에서 실행됩니다. 이같은 방식으로 의존성 있는 파이썬 패키지 역시 설치 가능합니다. 그림3과 같습니다.

## **그림3** 배쉬 명령 실행하기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/4hjxuu2.png" width="300px" title="source: imgur.com" />


---

## 구글드라이브와 연결하기

코랩은 일정 시간 접속하거나 아무것도 실행하지 않으면 해당 노트북이 초기화됩니다. 당시까지의 모든 결과물들이 날아갈 수 있다는 이야기인데요. 따라서 중간 결과물들을 어딘가에 저장해두는 것이 좋습니다. 

다른 저장 매체 대비 구글 드라이브와의 연동이 상대적으로 쉬운 편입니다. 일단 코랩 노트북에 그림4와 같이 실행해봅시다. 그림4의 실행 결과를 보면 `Go to this URL in a browser`에 링크가 표시된 걸 알 수 있습니다. 

## **그림4** 구글드라이브와 연결 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9rGhTtl.png" width="500px" title="source: imgur.com" />

그림4의 링크를 클릭한 후 구글 아이디로 로그인을 하면 그림5와 같은 화면을 볼 수 있습니다. `허용` 버튼을 누르면 인증 코드가 나타납니다.

## **그림5** 구글드라이브와 연결 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/SxLIFZ5.png" width="300px" title="source: imgur.com" />

인증코드를 다시 그림4의 입력란에 붙여넣기를 하면 현재 사용 중인 코랩 노트북이 자신의 구글 드라이브에 액세스할 수 있게 됩니다. 이 모든 과정을 거쳐 성공적으로 마운트가 됐다면 그림6의 [1]처럼 `Mounted at /gdrive`라는 메세지가 출력됩니다.

## **그림6** 구글드라이브와 연결 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pUgIRV8.png" width="500px" title="source: imgur.com" />

그림6의 [2]는 실제로 파일을 써보는 연습을 하는 코드입니다. 이를 실행한 결과가 그림7인데요. 구글 드라이브의 '내 드라이브'에 test.txt가 존재함을 확인할 수 있습니다.

## **그림7** 구글드라이브와 연결 (4)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ftCRrCP.png" width="500px" title="source: imgur.com" />

구글드라이브 연결에 성공했다면 구글 드라이브에 업로드한 자신의 파일들도 얼마든지 코랩 노트북에서 읽어 들여 사용할 수 있습니다.

---


## 코랩 노트북 복사하기

이 책 3장 이후 각 튜토리얼에서 코랩 노트북 링크를 제공합니다. 하지만 해당 노트북은 읽기 권한만 부여돼 있기 때문에 실행하거나 노트북 내용을 고칠 수가 없을 겁니다. 노트북을 복사해 내 것으로 만들면 이 문제를 해결할 수 있습니다. 

튜토리얼의 코랩 노트북 링크를 클릭한 후 구글 아이디로 로그인하면 그림8과 같은 화면을 볼 수 있습니다. `드라이브로 복사`를 클릭하면 코랩 노트북이 자신의 드라이브에 복사됩니다. 별도의 설정을 하지 않았다면 해당 노트북은 그림9처럼 `내 드라이브/Colab Notebooks` 폴더에 담깁니다.

## **그림8** 노트북 복사하기 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/xNmcoKE.png" width="300px" title="source: imgur.com" />

## **그림9** 노트북 복사하기 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/D4pzPe0.png" width="450px" title="source: imgur.com" />

---

## 하드웨어 가속기 선택하기

딥러닝 모델을 학습하려면 하드웨어 가속기를 사용해야 계산 속도를 높일 수 있습니다. 코랩에서는 GPU와 TPU 두 종류의 가속기를 지원합니다. 그림11과 같이 코랩 화면의 메뉴 탭에서 `런타임 > 런타임 유형 변경`을 클릭합니다. 이후 그림12와 같이 GPT 혹은 TPU 둘 중 하나를 선택합니다. 

## **그림10** 하드웨어 가속기 설정 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JFUva3P.png" width="500px" title="source: imgur.com" />

## **그림11** 하드웨어 가속기 설정 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/i4XvOhQ.png" width="300px" title="source: imgur.com" />


사용자 계정에 따라 할당받을 수 있는 하드웨어 가속기의 수와 종류가 다릅니다. 코랩이 가지고 있는 자원 대비 사용자가 많을 경우에도 역시 달라질 수 있습니다.


---


## 런타임 종료하기

개인 계정으로 동시에 사용할 수 있는 세션 수와 코랩 자원은 한정적이기 때문에 불필요한 노트북 실행은 피하는 게 좋습니다. 만일 노트북 실행을 마쳤다는 판단이 선다면 그림12럼 명시적으로 런타임을 초기화해 주세요. 이 경우 노트북 실행을 완전히 중단하고 점유하고 있던 CPU, RAM, 하드웨어 가속기 등 계산 자원을 반납하게 됩니다.


## **그림12** 런타임 종료
{: .no_toc .text-delta }
<img src="https://i.imgur.com/vYxJoQT.png" width="500px" title="source: imgur.com" />


---

## 기타 자세한 내용

코랩 관련 더 자세한 내용은 다음을 참고하세요.

- [Colaboratory 자주 묻는 질문](https://research.google.com/colaboratory/faq.html)


---
