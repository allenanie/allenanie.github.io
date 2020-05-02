---
published: true
layout: post
draft: true
title: "Bayesian Neural Decoding: <br> &mdash; Building an Image Captioner That Can Answer Visual Questions"
---

## Preface

In the past 2 years, it has been clear that NLP will continue the path of training ultra-large general models like BERT, XLNet, OpenAI-GPT2, ELECTRA, with general-purpose unsupervised learning objectives (such as language modeling, or discourse objectives[^1][^2] etc.). These models are massive, takes a long time to train, and once trained, we don't want to retrain from scratch again. In this sense, model re-purposing becomes important -- how do we leverage these ultra-large model to do what we want them to do? In the case of supervised learning, it was **fine-tuning** that enabled utilization of these general models. But what else? 

In this post, I will attempt to introduce a method to control neural language generation without retraining any model. I will attempt to **coerce the model to do something it was never trained to do** (similar to zero-shot learning), and demonstrate the power of Bayesian neural decoding. This post is to illustrate key points from my new paper with Reuben Cohn-Gordon and Chris Potts[^3]. 

But firist, I want to show you the "magic" we can achieve through Bayesian decoding:

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/semadv-algorithm1.png?raw=true" style="width:50%"> </p>

## A Tutorial of Rational Speech Act for Deep Learning People



[^1]:  Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." *arXiv preprint arXiv:1909.11942* (2019).
[^2]:  Nie, Allen, Erin D. Bennett, and Noah D. Goodman. "Dissent: Sentence representation learning from explicit discourse relations." *arXiv preprint arXiv:1710.04334* (2017).
[^3]: Nie, Allen, Reuben Cohn-Gordon, and Chris Potts. "Pragmatic Issue-Sensitive Image Captioning ." arXiv preprint arXiv:2004.14451 (2020).