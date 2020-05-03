---
published: true
layout: post
draft: true
title: "Bayesian Neural Decoding: <br> &mdash; Building an Image Captioner That Can Answer Visual Questions"
---

## Preface

In the past 2 years, it has been clear that NLP will continue the path of training ultra-large general models like BERT, XLNet, OpenAI-GPT2, ELECTRA, with general-purpose unsupervised learning objectives (such as language modeling, or discourse objectives[^1][^2] etc.). These models are massive, takes a long time to train, and once trained, we don't want to retrain from scratch again. In this sense, model re-purposing becomes important -- how do we leverage these ultra-large model to do what we want them to do? In the case of supervised learning, it was **fine-tuning** that enabled utilization of these general models. But what else? 

In this post, I will attempt to introduce a method to control neural language generation without retraining any model. I will attempt to **coerce the model to do something it was never trained to do** (similar to zero-shot learning), and demonstrate the power of Bayesian neural decoding. Through this post I will illustrate key points from my new paper with Reuben Cohn-Gordon and Chris Potts[^3]. 

But firist, I want to show you the "magic" we can achieve through Bayesian decoding:

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/fig5_large.png?raw=true" style="width:100%"> </p>

We make a 6-layer encoder-decoder Transformer with 55M parameters that is trained on MS COCO dataset that obtained SoTA CIDEr score, which has NEVER seen any VQA question/answer before, answer VQA questions in the image captions. The base caption shows what the Transformer originally would have outputted without our method (last column). The penultimate column (Issue-sensitive caption) shows what our method would produce. 

Since image captioner can be considered a very general model learning the association between text and image (grounding text into image scenes), it avoids making potential mistakes VQA model might have made. For example, the fourth image asks "What color is the sky?". A SoTA VQA model will answer "Gray." (If you want to try it -- here's a link to a [live demo](https://vqa.cloudcv.org/)). However, this is neither a very informative nor helpful answer -- in fact, **the sky is not gray** in this picture, it is only gray because it's **a black and white photo of the sky**. Luckily, because an image captioning model has more general-domain knowledge, it picks up on that and spells out the right answer.

This is perhaps the best reason for repurposing an image captioning model to answer question beyond "just because we can". An image captioning model has the chance to fully learn the world and the knowledge within (text and vision). It does not need to live under VQA's data bias where answer is always one or two words. With our method (and some tuning), it can answer any visual questions without training on QA data.

**Imagine if we can train an ultra-large joint text-image model on a generative objective (just like BERT), then we can repurpose such model to answer visual questions. Bayesian decoding to Ultra-large Image Captioner is similar to what fine-tuning is to BERT.** Both offer valuable ways to repurpose very powerful models to do our bidding.

After this proper motivation, all we need is how to **coerce the model to be question-aware**. In the paper, we refer to this as **"issue-sensitive"**.

## Asking a Question without Asking a Question

Let's formalize our setting a bit. It's an image captioner, so the input is  just an image: $\mathbf{i}$, and the image captioner will generate a sequence of words $\mathbf{w} = (w_1, w_2, ..., w_T)$. Usually, a VQA model's input is an image and a question tuple $(\mathbf{q}, \mathbf{i})$, and the output is $\mathbf{a} \in \mathcal{A}$ (an answer from a fixed set of answers -- exactly like classification, except with usually a couple of thousand choices). How do we bridge the difference?

Apparently, we can't just build a question encoder with random weights, and hope neural network will magically give us an answer. Instead of being "brain-dead", we can try to use it a bit and think: even though we are not able to ask an image captioner a question directly, we can think about what we want to get as the answer to a question. Here are some common VQA questions to an image: "What color is the wall?" "What position is the man playing?" "How many toilets are there?". The answers are "Red", "Pithcer", and "Two". 

Now we take a closer look at the image -- yes, it's a picture, but we can think of it differently. 

(Introduce image as a set of attributes, first-order, second-order, abstract, etc.) (Put airplane image there) (Then after saying image = set of features/attributes, introduce RSA to pick an item from a set)

## RSA: Bayesian Game to Select Item from Set





[^1]:  Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." *arXiv preprint arXiv:1909.11942* (2019).
[^2]:  Nie, Allen, Erin D. Bennett, and Noah D. Goodman. "Dissent: Sentence representation learning from explicit discourse relations." *arXiv preprint arXiv:1710.04334* (2017).
[^3]: Nie, Allen, Reuben Cohn-Gordon, and Chris Potts. "Pragmatic Issue-Sensitive Image Captioning ." arXiv preprint arXiv:2004.14451 (2020).