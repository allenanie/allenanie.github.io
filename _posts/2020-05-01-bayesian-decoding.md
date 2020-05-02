---
published: true
layout: post
draft: true
title: "Bayesian Neural Decoding: <br> &mdash; Building an Image Captioner to Answer Visual Questions"
---

## Preface

In the past 2 years, it has been clear that NLP will continue the path of training ultra-large general models like BERT, XLNet, OpenAI-GPT2, ELECTRA, with general-purpose unsupervised learning objectives (such as language modeling, or discourse objectives[^1][^2] etc.). In this sense, model re-purposing becomes important -- how do we leverage these ultra-large model to do what we want them to do? In the case of supervised learning, it was **fine-tuning** that enabled us to utilize these general models. What about the case of language generation?

Before we dive too deep into the thesis, let's 



[^1]:  Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." *arXiv preprint arXiv:1909.11942* (2019).
[^2]:  Nie, Allen, Erin D. Bennett, and Noah D. Goodman. "Dissent: Sentence representation learning from explicit discourse relations." *arXiv preprint arXiv:1710.04334* (2017).