---
layout: default
title: Preprocess
nav_order: 3
has_children: true
has_toc: false
permalink: /docs/preprocess
---

# 문장을 작은 단위로 쪼개기
{: .no_toc }

자연어 처리의 첫 단추는 자연어 문장을 작은 단위인 토큰(token)으로 분석하는 과정입니다. 이 장에서는 Byte Pair Encoding(BPE)을 중심으로 이론을 살펴봅니다. 이어 어휘 집합을 구축하고 이 어휘 집합으로 잘게 쪼갠 토큰들을 정수로 변환하는 실습을 진행합니다.
