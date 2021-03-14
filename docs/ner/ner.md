---
layout: default
title: Named Entity Recognition
nav_order: 7
has_children: true
has_toc: true
permalink: /docs/ner
---

# 6장 단어에 꼬리표 달기
{: .no_toc }

시퀀스 레이블링(sequence labeling)이란 음성, 단어 따위의 시퀀스 데이터에 레이블을 달아주는 과제(task)를 가리킵니다. 자연어 처리에서 대표적인 시퀀스 레이블링 과제로 개체명 인식(Named Entity Recognition)이 있습니다. 문장을 토큰화한 뒤 토큰 각각에 인명, 기관명, 장소 등 개체명 태그를 붙여주는 과제입니다. 이 장에서는 한국해양대학교에서 공개한 데이터셋에 자체로 제작한 데이터셋을 합친 데이터를 가지고 개체명 인식 모델을 구축하는 방법을 살펴봅니다.
