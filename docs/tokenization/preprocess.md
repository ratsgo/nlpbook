---
layout: default
title: Vocab & Tokenization
nav_order: 4
has_children: true
has_toc: true
permalink: /docs/preprocess
---

# 문장을 작은 단위로 쪼개기
{: .no_toc }

자연어 처리의 첫 단추는 자연어 문장을 작은 단위인 토큰(token)으로 분석하는 과정입니다. 이 장에서는 Byte Pair Encoding(BPE)을 중심으로 이론을 살펴봅니다. 구체적으로는 ∆ BPE 방식으로 어휘 집합(vocabulary) 구축 ∆ 앞서 구축한 어휘 집합으로 토큰화(tokenization) ∆ 토큰들을 정수(int)로 변환(encode) 등입니다. 마지막으로는 허깅페이스 패키지를 활용해 튜토리얼을 진행합니다. 
