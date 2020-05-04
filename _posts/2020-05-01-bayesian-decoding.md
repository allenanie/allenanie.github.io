---
published: true
layout: post
draft: true
title: "Bayesian Neural Decoding: <br> &mdash; Using Visual Questions to Control Caption Generation"
---

## Preface

In the past 2 years, controlling generative models (such as GANs and VAEs) have been widely studied in the Computer Vision literature. The idea is that once these large capacity neural network models learn the data manifold of millions of images, it has "inherent" knowledge about the world (in the form of understanding "skin tone", "hair style" in celebrity faces, or "brightness", "camera angles" for bird images). Then through some post-training manipulation for GAN/VAE, we can bring these disentangled aspect of the image out to the surface.

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/RSA_GAN_VAE.png?raw=true" style="width:120%"> <br> <span>Figure 1: (a) from OpenAI GLOW demo; (b) from ICLR 2020 paper: On the "steerability" of generative adversarial networks.</span> </p>

Despite such impressive result, controlling generative models in text has been far more difficult. Unlike images, where humans have an intuitive idea on what to control: rotation angle, brightness, blueness of picutres; age, beard, hair color for faces, we have no intuitive idea on what to control for text beyond simple ideas like sentiment or tense[^1]. Follow-up work like Lample et al.[^2] leveraged the domain where the data is collected to add additional attributes like gender, race, age, and interests (like music, book, movie, etc.), and trained a neural network from scratch conditioned on these attributes.

Although these efforts have greatly expanded our capability to manipulate text, they lose the aspect of discovering “naturally learned disentangled representation” by modeling the data distribution. Also, any work that requires an attribute classifier is inherently unscalable.

In this post, I will show you how we can control the output from a model that is trained on a joint distribution of images and text. It builds up a conditional distribution $P(\mathbf{w}\vert\mathbf{i})$ where we can control the generation of text by systematically varying the image input to this distribution. We do not rely on additional attribute classifiers and allow the control of text generation through a simple visual question inputted by the user. 

Since conditioning on images allows us to (again) from an intuitive understanding of what aspect of  text we want to control, we can use our understanding in images (various objects, behaviors) as ground truth to dictate what we want in text. Our proposed method also only uses a trained generative model, which means any aspect we are able to control, is inherently learned by the generative model by learning from the conditional distribution.

Previous works that studied this setting (and used methods somewhat similar to ours) mostly focused on the concept of "granularity" -- how detailed can the text be [^3][^4]. This is only one attribute of text to control. In our work, we will try to control the model to generate text on an infinite amount of attributes. The work described by this post is my new paper with Reuben Cohn-Gordon and Chris Potts[^5].


<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/fig5_large.png?raw=true" style="width:120%"> <br> <span>Figure 2: Examples from applying our method to a SoTA 6-layer encoder-decoder Transformer with 55M parameters image captioner trained on the MS COCO dataset. The base caption shows what the Transformer originally would have outputted without our method (last column). The penultimate column (Issue-sensitive caption) shows what our method would produce. We can produce captions that answer questions about objects and attribute of objects.</span> </p>

By asking visual questions, such as "what color is the sky", we control the image captioner to focus on the peculiar nature of this photo (that this is a black & white photo). It instead of describing what is the scene of this picture ("airplane taking off"), to describe the meta-level information of this picture: ("a black and white photo of an airplane").

Examining controllable text generation in a joint (image, text) domain solves the problem of not knowing what aspect of text to control, and dense annotations in image datasets allows us to quickly verify how well we can control the generated text. I will then describe how we control the caption generation by asking a question (of course, a pre-trained VQA model is involved).

## Asking a Question without Asking a Question

Let's formalize our setting a bit. It's an image captioner, so the input is  just an image: $\mathbf{i}$, and the image captioner will generate a sequence of words $\mathbf{w} = (w_1, w_2, ..., w_T)$. Usually, a VQA model's input is an image and a question tuple $(\mathbf{q}, \mathbf{i})$, and the output is $\mathbf{a} \in \mathcal{A}$ (an answer from a fixed set of answers -- exactly like classification, except with usually a couple of thousand choices). How do we bridge the difference?

Apparently, we can't just build a question encoder with random weights, and hope neural network will magically give us an answer. Instead of being "brain-dead", we can try to use it a bit and think: even though we are not able to ask an image captioner a question directly, we can think about what we want to get as the answer to a question. Here are some common VQA questions to an image: "What color is the wall?" "What position is the man playing?" "How many toilets are there?". The answers are "Red", "Pithcer", and "Two". 

Now we take a closer look at the image -- yes, it's a picture, but we can think of it differently:

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/airplane.png?raw=true" style="width:90%"> <br> <span>Figure 3: A photo of an airplane</span> </p>

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

We show an idealized example of how RSA works, where rows correspond to a cognitive representation of images (denoted as $\mathbf{i}$) (each image is a set that has many items), and the column is the utterance that we generate (denoted as $\mathbf{w}$). You can image the first row `{cap, mountain}` as an image of a person wearing a cap climbing a mountain. The second row `{cap, skiing}` as an image of a person wearing a cap skiing in snow. The column corresponds to the utterance/word you would pick to describe the image (in this simplified setting, we only get to pick ONCE). In other words, row represents data (images), column represents vocabulary space (utterances). When we train a generic caption model, we obtain $S_0(\mathbf{w}\vert\mathbf{i})$ -- given an image, which word would you output to describe this image? 

<p style="text-align: center"><img src="https://github.com/windweller/windweller.github.io/blob/master/images/bayesian_decoding/RSA_comp_fig.png?raw=true" style="width:100%"> <br> <span>Figure 4: Viewing RSA computation as Probability Tables.</span> </p>

Without any additional constraint, for the first row, an image of a person wearing a cap climbing a mountain, I can choose either "cap" or "mountain" to describe it, hence P("cap" \| i = 1) = P("mountain"\| i=1) in the top-left table for $S_0$. Now we add constraints -- what if we add 5 more images (the next 5 rows) to the first image, what should we output 



[^1]: Hu, Z., Yang, Z., Liang, X., Salakhutdinov, R., & Xing, E. P. (2017, August). Toward controlled generation of text. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 1587-1596). JMLR. org.
[^2]: Lample, G., Subramanian, S., Smith, E., Denoyer, L., Ranzato, M. A., & Boureau, Y. L. (2018). Multiple-attribute text rewriting.
[^3]:
[^4]:
[^5]: Nie, Allen, Reuben Cohn-Gordon, and Chris Potts. "Pragmatic Issue-Sensitive Image Captioning ." arXiv preprint arXiv:2004.14451 (2020).