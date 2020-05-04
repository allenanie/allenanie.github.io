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

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/fig5_large.png?raw=true" style="width:120%"> <br> <span>Figure 1: Examples from the MS COCO dataset. We can produce captions that answer questions about objects and attribute of objects.</span> </p>

We make a 6-layer encoder-decoder Transformer with 55M parameters that is trained on MS COCO dataset that obtained SoTA CIDEr score, which has NEVER seen any VQA question/answer before, answer VQA questions in the image captions. The base caption shows what the Transformer originally would have outputted without our method (last column). The penultimate column (Issue-sensitive caption) shows what our method would produce. 

Since image captioner can be considered a very general model learning the association between text and image (grounding text into image scenes), it avoids making potential mistakes VQA model might have made. For example, the fourth image asks "What color is the sky?". A SoTA VQA model will answer "Gray." (If you want to try it -- here's a link to a [live demo](https://vqa.cloudcv.org/)). However, this is neither a very informative nor helpful answer -- in fact, **the sky is not gray** in this picture, it is only gray because it's **a black and white photo of the sky**. Luckily, because an image captioning model has more general-domain knowledge, it picks up on that and spells out the right answer.

This is perhaps the best reason for repurposing an image captioning model to answer question beyond "just because we can". An image captioning model has the chance to fully learn the world and the knowledge within (text and vision). It does not need to live under VQA's data bias where answer is always one or two words. With our method (and some tuning), it can answer any visual questions without training on QA data.

**Imagine if we can train an ultra-large joint text-image model on a generative objective (just like BERT), then we can repurpose such model to answer visual questions. Bayesian decoding to Ultra-large Image Captioner is similar to what fine-tuning is to BERT.** Both offer valuable ways to repurpose very powerful models to do our bidding.

After this proper motivation, all we need is how to **coerce the model to be question-aware**. In the paper, we refer to this as **"issue-sensitive"**.

## Asking a Question without Asking a Question

Let's formalize our setting a bit. It's an image captioner, so the input is  just an image: $\mathbf{i}$, and the image captioner will generate a sequence of words $\mathbf{w} = (w_1, w_2, ..., w_T)$. Usually, a VQA model's input is an image and a question tuple $(\mathbf{q}, \mathbf{i})$, and the output is $\mathbf{a} \in \mathcal{A}$ (an answer from a fixed set of answers -- exactly like classification, except with usually a couple of thousand choices). How do we bridge the difference?

Apparently, we can't just build a question encoder with random weights, and hope neural network will magically give us an answer. Instead of being "brain-dead", we can try to use it a bit and think: even though we are not able to ask an image captioner a question directly, we can think about what we want to get as the answer to a question. Here are some common VQA questions to an image: "What color is the wall?" "What position is the man playing?" "How many toilets are there?". The answers are "Red", "Pithcer", and "Two". 

Now we take a closer look at the image -- yes, it's a picture, but we can think of it differently:

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/airplane.png?raw=true" style="width:90%"> <br> <span>Figure 2: A photo of an airplane</span> </p>

Cognitively, what does this image mean to us? Clearly it's not a set of pixels to our mind -- when we look at this image, we see **objects**: `{"sky", "airplane", "fence", "runway/tarmac", "clouds" ...}`. Each object is made of smaller objects -- for example, `airplane` contains smaller objects: `{wheel, engine, wing, horizontal tail, vertical tail}`. Each object can have **attributes**: `{one, two, white}` -- the composition of attribute and object will give us `{two wheels, two engines, two wings, two horizontal tails, one vertical tail}`.

Not just objects, smaller objects, and their attributes, our brain also understands the relationship between objects -- the **relationship** between `airplane` and `runway` is `taking off / landing`. Traditionally object recognition datasets focus solely on objects (such as MS COCO). [Visual Genome project](http://visualgenome.org/) actually has all three (object/region, attribute, relationship) annotated, but does not provide actual captions.

Any fact-based visual questions around this image will not deviate from these three categories. **This tells us that our goal is to coerce the caption model to produce captions that contain words related to objects/attributes/relationships that the visual question is referring to**. In our paper, we refer to this as the "resolution" of an "issue". To put it more explicitly, if the visual question is: "How many wheels does the airplane have?", the caption needs to contain "two wheels" (In Figure 1, we showed the caption model is capable of resolving questions regarding quantity).

To recap, hopefully I've convinced you an equivalence between an image, and its cognitive representation -- which is a set of objects, their attributes, and their relationships. To formalize this, we introduce $\psi(i)$ = `{airplane, airplane flying, photo, black & white photo, ....}` The resulting set is  prohibitively large -- we can roughly think this is the caption model encoder's internal knowledge of this image. **Then it is the decoder's job to select which subset of these set elements to output**. We've now reduced the task of decoding to item section from a set.

## RSA: A Bayesian Game to Select Items from Set

A disclaimer: I'm giving a very operational/engineering view of RSA (Rational Speech Act) framework, which **focuses not on what RSA is but what RSA does**. RSA is developed as a Psycholinguistic framework and it successfully replicated/recovered/recreated many human linguistic phenomenon. A more linguistic view is presented [here](https://www.problang.org/). 

RSA is a Bayesian game assuming rational agents and the following "winning" condition: given a set of sets (a list of images), the Player1 needs to pick the best item (utterance) from the set to represent it, so that Player2 has the highest chance of picking out the right set after seeing Player1's pick. In Linguistics, this is often called a "reference game", and the item Player1 picks out satisfies a "communicative goal". A more detailed description can be found [here](https://web.stanford.edu/class/linguist130a/materials/ling130a-handout-02-18-rsa.pdf).

So, let me introduce the formalism of RSA, which relies on Bayes theorem. In RSA, we have to rely on the raw form of Bayes' theorem. Let's first define $S_0(\mathbf{w}\vert\mathbf{i})$, aka literal speaker. This is our image captioner, a conditional probability distribution. The RSA calculation is to compute two probabilities (we refer to them as pragmatic listener and pragmatic speaker) recursively:


$$
\begin{align*}
L_1(\mathbf{i}|\mathbf{w}) &= \frac{S_0(\mathbf{w}|\mathbf{i}) P(\mathbf{i})}{P(\mathbf{w})} = \frac{S_0(\mathbf{w}|\mathbf{i}) P(\mathbf{i})}{\sum_{i \in \mathcal{I}} S_0(\mathbf{w}|\mathbf{i}) P(\mathbf{i})} \\
S_1(\mathbf{w}|\mathbf{i}) &= \frac{L_1(\mathbf{i}|\mathbf{w}) P(\mathbf{w})}{P(\mathbf{i})} = \frac{L_1(\mathbf{i}|\mathbf{w}) P(\mathbf{w})}{\sum_{w \in \mathcal{V}} L_1(\mathbf{i}|\mathbf{w}) P(\mathbf{w})} \\
\end{align*}
$$

In RSA books/papers, you often see the simplified version (skipping the normalization term):

$$
\begin{align*}
L_1(\mathbf{i}\vert\mathbf{w}) &\propto S_0(\mathbf{w}|\mathbf{i}) P(\mathbf{i}) \\
S_1(\mathbf{w}\vert\mathbf{i}) &\propto L_1(\mathbf{i}|\mathbf{w})P(\mathbf{w})
\end{align*}
$$

If this seems confusing, just replace $L_1$ and $S_1$ with the probability symbol $P$ and it should make more sense. Looking at this formula, except that we recognize it's just Bayes theorem, it's very hard to have an intuition over what it does. Allow me to use an example (a probability table) to illustrate how RSA works and flesh out the intuition.

We show a probability table, where rows correspond to a cognitive representation of images (denoted as $\mathbf{i}$) (each image is a set that has many items), and the column is the utterance that we generate (denoted as $\mathbf{w}$). 

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/RSA_fig.png?raw=true" style="width:100%"> <br> <span>Figure 3: Viewing RSA computation as Probability Tables.</span> </p>



[^1]:  Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." *arXiv preprint arXiv:1909.11942* (2019).
[^2]:  Nie, Allen, Erin D. Bennett, and Noah D. Goodman. "Dissent: Sentence representation learning from explicit discourse relations." *arXiv preprint arXiv:1710.04334* (2017).
[^3]: Nie, Allen, Reuben Cohn-Gordon, and Chris Potts. "Pragmatic Issue-Sensitive Image Captioning ." arXiv preprint arXiv:2004.14451 (2020).