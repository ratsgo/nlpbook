---
layout: default
title: Paper Book Notice
nav_order: 10
has_children: false
has_toc: false
permalink: /docs/notice
---

# 종이책 정오표
{: .no_toc }

종이책 오탈자 및 수정 사항을 안내합니다. 해당 내용은 [nlpbook issue](https://github.com/ratsgo/nlpbook/issues)에서 토론을 거쳐 확정된 것입니다. 여기에 나오지 않는 사항이라도 언제든지 편하게 리포트 해주세요! 정정 의견은 [github.com](https://github.com)에 회원 가입을 한 뒤 [nlpbook issue 등록 항목](https://github.com/ratsgo/nlpbook/issues/new)에서 제목과 본문을 입력하면 등록할 수 있습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## 237페이지

다음 문단을 교체.

**수정 전**

> 샘플링 방식 예를 든 다음 그림을 보면 그라는 컨텍스트를 입력했을 때 모델은 다음 토큰으로 집(0.5), 책(0.4), 사람(0.1)이 그럴듯하다고 예측했습니다. 여기에서 다음 토큰을 확률적으로 선택합니다. 집이 선택될 가능성이 50%로 제일 크고 사람이 선택될 가능성도 10%로 작지만 없지 않습니다.

**수정 후**

> 샘플링 방식 예를 든 다음 그림을 보면 그라는 컨텍스트를 입력했을 때 모델은 다음 토큰으로 책(0.5), 집(0.4), 사람(0.1)이 그럴듯하다고 예측했습니다. 여기에서 다음 토큰을 확률적으로 선택합니다. 책이 선택될 가능성이 50%로 제일 크고 사람이 선택될 가능성도 10%로 작지만 없지 않습니다.


그림 8-8을 다음으로 교체.

<img width="500" src="https://user-images.githubusercontent.com/26211652/147377581-e8abb8d9-6db6-4a0b-aa37-d7e6497db2a7.png">


## 240페이지


그림 8-11을 다음으로 교체.

<img width="500" src="https://user-images.githubusercontent.com/26211652/147377570-42915486-d087-49f3-8834-2d579654d5ee.png">


---