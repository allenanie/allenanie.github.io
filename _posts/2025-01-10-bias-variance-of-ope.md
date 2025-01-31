---
layout: post
title: On Bias and Variance Decomposition of Offline Policy Evaluation Estimators
published: true
draft: false
render_with_liquid: false
---

Note: This blog post is an unpublished partial draft of a larger work, jointly written with [Aishwarya Mandyam](https://aishwaryamandyam.com/).

## Introduction

Evaluation is a critical component of learning contextual bandit policies that can be deployed in high-risk settings. One way to perform this evaluation is to directly deploy it in the setting of interest, or use an accurate simulator evaluate how well this policy will do when deployed. Unfortunately, directly deploying this policy may be unsafe or unethical (i.e. deploying in a hospital environment), and we may not have accurate simulators of the deployment setting. In this case, we turn to Off-policy Evaluation (OPE), which estimates the value of the learned policy using an offline or cached dataset of samples that arise from a behavior policy. The challenge is that this behavior policy is distinct from the target policy, and there may be limited coverage of the behavior dataset in areas of the state-action space where the target policy frequents. Several strategies have been proposed to mitigate this dataset distribution shift.

## Categories of OPE estimators

There are three broad categories of OPE estimators. The first is the direct method (DM)[1], which in the contextual bandit setting, approximates the reward function, and uses that reward model to simulate the target policy. Typically, if the reward function estimate is good (as in the reward model is fully realizable, and there is enough data to estimate it), DM is a great option for OPE. When the reward estimate is bad, DM estimators typically have high bias. 

The second option is importance sampling (IS)[2], in which you re-weight samples in the behavior dataset as if they appeared from the target policy. The weight used is an importance sampling ratio, typically written as $\rho$, which compares how likely a sample is to occur in the target policy and the behavior policy. IS-based OPE estimators typically have high variance.

## Notation

$\pi_e$ : target, or learned policy. Our goal is to evaluate this policy. 

$\pi_b$ : behavior policy. Our offline dataset contains samples that arise from this behavior policy. 

$\rho = \frac{\pi_e(a|s)}{\pi_b(a|s)}$ : importance sampling ratio comparing the likelihood of observing a sample in the target policy to the likelihood of observing it in the behavior policy

$S,A$ : state and action spaces

$s, a, r$ : observed state, action, and reward

$N$: number of samples

$d$ : state distribution

$R(s,a)$: the true reward function used for the factual dataset, with mean $\bar{R}(s,a)$ and variance $\sigma^2(s,a)$. 

$D_1 \sim \mathcal{D}$: dataset used to fit the OPE estimate

$D_{\hat{R}} \sim \mathcal{D}$: dataset used to fit the reward model

$V = V(\pi_e) = \mathbb{E}_{s \sim  d, a\sim\pi_e}[R(s, a)]$: the true return of the model. $\hat V$ is an estimator for this estimand.

## Direct Method

The direct method is  $\hat{V}^{DM} = \frac{1}{N}\sum_{i=1}^N \sum_{a \in A}\pi_e(a|s_i) \hat{R}(s_i, a_i)$. 

### Bias Derivation

First we use the linearity of expectation and definition of expectation over $D_1$. Then, we use the mean of the distribution $R$. Finally, we use the definition of the value function and the policy value. 

$$
\begin{align}\mathbb{E}_{D_1 \sim\mathcal{D}, D_{\hat{R}} \sim \mathcal{D}}[\hat{V}^{\text{DM}}] &= \mathbb{E}_{D_1 \sim\mathcal{D}} \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}} \left[ \frac{1}{N}\sum_{s_i \in D_1} \sum_{a \in A} \pi_e(a|s_i) \hat{R}(s_i,a) \right] \\
&= \mathbb{E}_{D_1 \sim \mathcal{D}} \left[ \frac{1}{N} \sum_{j=1}^{N} \sum_{a \in A} \pi_e(a|s_i) \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}} [\hat{R}(s_i,a) ] \right] \\
&= \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{D_1 \sim\mathcal{D}} \left[ \sum_{a \in A} \pi_e(a|s_i) \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}} [\hat{R}(s_i,a) ] \right] \\
&= \frac{1}{N} \sum_{i=1}^{N} \sum_{s \in \mathcal{S}} d(s) \left( \sum_{a \in A} \pi_e(a|s) \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}} [\hat{R}(s,a) ] \right) \\
&= \frac{1}{N} \sum_{i=1}^{N} \sum_{s \in \mathcal{S}} d(s) \left( \sum_{a \in A} \pi_e(a|s) \bar{R}(s,a) \right) \\
&= \frac{1}{N} \sum_{i=1}^{N} \sum_{s \in \mathcal{S}} d(s) V^{\pi_e}(s_i) \\
&= \frac{1}{N} \sum_{i=1}^{N} V(\pi_e) \\
&= V(\pi_e)\end{align}
$$

### Variance Derivation

By law of total variance, we can decompose the variance of the direct method estimator into two terms:

$$
\begin{align}
& \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}, D_{1} \sim \mathcal{D}}[\hat{V}^{\text{DM}}] = \underbrace{\mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}} \left[\hat{V}^{\text{DM}}\right]}{(1)} + \underbrace{\mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \mathbb{V}_{D_{\hat{R}} \sim \mathcal{D}} \left[ \hat{V}^{\text{DM}} \right]}_{(1')}
\end{align}
$$

We can substitute the intermediate result from the proof of the bias of DM into (1):

$$

\begin{align}
(1) &= \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}}\big[ \hat{V}^{\text{DM}} \big] \right] = \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \hat{R}(d_0,\pi_e) \right]
\end{align}
$$

For (1'), we first consider the inner variance with respect to $D_{\hat{R}}$ assuming $\hat{R}$ is given:

$$
\begin{align}
\mathbb{V}_{D_{\hat{R}} \sim \mathcal{D}} \left[ \hat{V}^{\text{DM}} \right] &= \mathbb{V}_{D_{\hat{R}} \sim \mathcal{D}} \left[ \frac{1}{N} \sum_{i=1}^{N} \sum_{a \in A} \pi_e(a|s_i) \hat{R}(s_i,a) \right] \\
&= \frac{1}{{N}^2} \sum_{i=1}^{N} \mathbb{V}_{s_i \sim d_0} \left[ \hat{R}(s_i,\pi_e) \right] \\
&= \frac{1}{N} \mathbb{V}_{s \sim d_0} \left[ \hat{R}(s,\pi_e) \right]
\end{align}
$$

Substituting this into (1') :

$$

\begin{align}
& (1') = \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \mathbb{V}_{D_{\hat{R}} \sim \mathcal{D}} \left[ \hat{V}^{\text{DM}} \right] \\
&= \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \frac{1}{N} \mathbb{V}_{s \sim d_0} \left[ \hat{R}(s,\pi_e) \right] \right] \\
&= \frac{1}{N} \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e)^2 \right] - \mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e) \right]^2 \right]  && \\
&= \frac{1}{N} \Bigg( \underbrace{\mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e)^2 \right]}_{(2)} - \underbrace{\mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}}\left[\mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e) \right]^2 \right]}_{(2')} \Bigg)
\end{align}
$$

$$
\begin{align}
(2) &= \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e)^2 \right] \\
&= \mathbb{E}_{s \sim d_0} \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \hat{R}(s,\pi_e)^2 \right] \\
&= \mathbb{E}_{s \sim d_0} \left[ v^{\pi_e}(s)^2 + \mathbb{V}_{D_{\hat{R}}\sim\mathcal{D}}[\hat{R}(s,\pi_e)] \right] \\
&= \mathbb{E}_{s \sim d_0} \left[ {v^{\pi_e}(s)}^2 \right] + \mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right] \\
&= \mathbb{E}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right]^2 + \mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] + \mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right] &&  \\
&= {v(\pi_e)}^2 + \mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] + \mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right]  \\
\\
(2') &= \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}}\left[\mathbb{E}_{s \sim d_0} \left[ \hat{R}(s,\pi_e) \right]^2 \right] \\
&= \mathbb{E}_{D_{\hat{R}} \sim\mathcal{D}}\left[\hat{R}(d_0,\pi_e)^2 \right] \\
&= v(\pi_e)^2 + \mathbb{V}_{D_{\hat{R}}\sim\mathcal{D}}[\hat{R}(d_0,\pi_e)]
\end{align}
$$

Thus,

$$

\begin{align}
(2) - (2') &= \mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] + \mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right] - \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}}\left[\hat{R}(d_0,\pi_e) \right] \\
\end{align}
$$

Putting everything together, we have:

$$

\begin{align}
& \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}, D_{\hat{R}} \sim \mathcal{D}}[\hat{V}^{\text{DM}}] = (1) + (1') \\
&= \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} \left[ \hat{R}(d_0,\pi_e) \right] \\
&+ \frac{1}{N} \left(\mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] + \mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right] - \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}}\left[\hat{R}(d_0,\pi_e) \right]\right) \\
&= \frac{1}{N}\mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] + \frac{1}{N}\mathbb{E}_{s \sim d_0} \left[ \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}} [ \hat{R}(s,\pi_e) ] \right] + \big(1-\frac{1}{N}\big) \mathbb{V}_{D_{\hat{R}} \sim\mathcal{D}}\left[\hat{R}(d_0,\pi_e) \right] \\
&= \frac{1}{N}\mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] \frac{1}{N}\mathbb{E}_{s \sim d_0} \bigg[ \mathbb{E}_{a \sim \pi_e} \Big[\pi_e(a|s) \, \sigma_R^2(s,a) \, \mathbb{E}_{D{\hat{R}} \sim \mathcal{D}}\big[\frac{1}{N_{s,a}(D_{\hat{R}})}\big] \Big] \bigg] \\
& \qquad \qquad \qquad \qquad \;\; + \big(1-\frac{1}{N}\big) \mathbb{E}_{s \sim d_0}\mathbb{E}_{a \sim \pi_e} \left[d_0(s) \, \pi_e(a|s) \, \sigma_R^2(s,a) \, \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}}\Big[\frac{1}{N_{s,a}(D_{\hat{R}})}\Big] \right] \\
&= \frac{1}{N}\mathbb{V}_{s \sim d_0} \left[ {v^{\pi_e}(s)} \right] \\
&+ \mathbb{E}_{s \sim d_0}\mathbb{E}_{a \sim \pi_e} \left[\Big(\frac{1}{N} + \big(1-\frac{1}{N}\big)d_0(s) \Big) \, \pi_e(a|s) \, \sigma_R^2(s,a) \, \mathbb{E}_{D_{\hat{R}} \sim \mathcal{D}}\Big[\frac{1}{N_{s,a}(D_{\hat{R}})}\Big] \right]
\end{align}

$$

## Importance Sampling

The importance sampling (IS) estimator is: $\hat V^{\text{IS}} = \frac{1}{N}\sum_{i=1}^N \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} r_i$

## Bias Derivation

Importance sampling estimator is unbiased.

$$
\begin{align}\mathbb{E}_{D \sim \mathcal{D}}[\hat V^{\text{IS}}] &= \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Bigg[\frac{1}{N}\sum_{i=1}^N \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} r_i \Bigg] \\
&= \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Bigg[ \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} r_i \Bigg] \\
&= \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{s_i \sim d} \Bigg[ \pi_b(a_i|s_i) \frac{ \pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} r_i \Bigg] \\
&= \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{s_i \sim d} \Bigg[  \pi_e(a_i|s_i) r_i \Bigg] \\
&= \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{s_i \sim d, a_i \sim \pi_e(s_i)} \Big[ r_i \Big] \\
&= \frac{1}{N}\sum_{i=1}^N V(\pi_e) \\
&= V(\pi_e)
\end{align}
$$

## Variance Derivation

However, Importance sampling estimator is known for having a relatively large variance.

$$
\begin{align}
\mathbb{V}_{D \sim \mathcal{D}}[\hat{V}^{\text{IS}}] &= \mathbb{V}_{s_i \sim d, a_i \sim \pi_b} \Bigg[ \frac{1}{N}\sum_{i=1}^N \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} R(s_i, a_i) \Bigg] \\
&= \frac{1}{N^2} \mathbb{V}_{s_i \sim d, a_i \sim \pi_b} \Bigg[ \sum_{i=1}^N \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} R(s_i, a_i) \Bigg] \\
&= \frac{1}{N} \mathbb{V}_{s_i \sim d, a_i \sim \pi_b} \Bigg[  \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} R(s_i, a_i) \Bigg]\\
\end{align}
$$

Step 24 to 25 is because we know each sample is i.i.d. Let $w_i = \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)}$, the **empirical variance estimator** is: 

$\hat{\mathbb{V}}_{D \sim \mathcal{D}}[\hat{V}^{\text{IS}}] = \frac{1}{N} \sum_{i=1}^N \big (w_i R(s_i, a_i) - \hat V^{\text{IS}} \big)^2$

### Bonus Content (Weighted Importance Sampling)

Weighted importance sampling estimator is also called a self-normalizing importance sampling estimator. It is a biased estimator but has smaller variance compared to IS.

WIS can be defined as (using $w_i = \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)}$)

$$
\hat V^{\text{WIS}} = \frac{\sum_{i=1}^N w_i R(s_i, a_i)}{\sum_{i=1}^N w_i} = \sum_{i=1}^N \frac{w_i}{\sum_{i=1}^N w_i} R(s_i, a_i)
$$

We show a quick derivation of their bias and variance.

**Bias Derivation**

$$
\begin{align}\mathbb{E}_{D \sim \mathcal{D}}[\hat V^{\text{WIS}}] &= \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Bigg[\sum_{i=1}^N \frac{w_i}{\sum_{i=1}^N w_i} R(s_i, a_i) \Bigg] \\
&= \sum_{i=1}^N \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Bigg[ \frac{w_i}{\sum_{i=1}^N w_i} R(s_i, a_i) \Bigg] \\
&= \sum_{i=1}^N \frac{1}{\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [\sum_{i=1}^N w_i]} \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Bigg[ w_i R(s_i, a_i) \Bigg] \\
&=\sum_{i=1}^N \frac{1}{\sum_{i=1}^N\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [w_i]} \mathbb{E}_{s_i \sim d, a_i \sim \pi_b}\Bigg[ \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} R(s_i, a_i) \Bigg] \\
&=\sum_{i=1}^N \frac{1}{\sum_{i=1}^N\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [w_i]} V(\pi_e) \\
&= \frac{1}{\sum_{i=1}^N\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [w_i]} \sum_{i=1}^N V(\pi_e)
\end{align}
$$

Step (29) to step (30) followed the IS bias derivation. We can see that WIS estimator is only unbiased when $\sum_{i=1}^N\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [w_i] = N$, which means $\mathbb{E}_{s_i \sim d, a_i \sim \pi_b} [w_i] = 1$. If the evaluation policy $\pi_e$ and behavior policy $\pi_b$ takes the exact same action, then this estimator is unbiased.  

**Consistency Corollary**

This is a biased estimator

$$
\hat V^{\text{WIS}}  =  \underbrace{\frac{1}{\frac{1}{N}\sum_{i=1}^N w_i}}_{a} \cdot \underbrace{\frac{1}{N} \sum_{i=1}^N  w_iR(s_i, a_i)}_{b}
$$

We can show that $\hat V^{\text{WIS}}$ is consistent.

$$
\lim_{N \rightarrow \infty }\frac{1}{N}\sum_{i=1}^N w_i \stackrel{a.s.}{\rightarrow} \mathbb{E}_{s_i \sim d, a_i \sim \pi_b} \Big[ \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} \Big] = \mathbb{E}_{s_i \sim d} \Big[ \pi_e(a_i|s_i)\Big] = 1
$$

(Note: I’m not sure if it’s strong consistency or weak consistency here)

As we can see, $a \rightarrow 1$ as $N \rightarrow \infty$. We can also apply the strong law of large numbers on $b$ and see $b \rightarrow V(\pi_e)$ as $N \rightarrow \infty$.  Therefore, $\hat V^{\text{WIS}} \rightarrow V(\pi_e)$ as $N \rightarrow \infty$.

**Variance Derivation**

Variance of an WIS estimator requires derivation over $\frac{w_i}{\sum_{i}w_i}$, which in its essence, a ratio — the variance of this ratio through the sampled dataset $D$ has an effect on the overall variance of WIS. In order to work this part out, we will first very quickly discuss the multivariate delta method:

For ease of discussion, we define two dummy variables $X = \frac{1}{N} \sum_{i} w_i R(s_i, a_i)$ and $Y = \frac{1}{N} \sum_i w_i$. It’s easy to see $\hat V^{\text{WIS}} = X/Y$. Let $\mu_x = \mathbb{E}[X]$ and $\mu_y = \mathbb{E}[Y]$. We can define a two-variable function $f(x, y) = x/y$ and its first-order Taylor Expansion to be:

$$
f(X, Y) \approx f(\mu_x, \mu_y) + \nabla_X f(\mu_x, \mu_y)(X - \mu_x) + \nabla_Y f(\mu_x, \mu_y)(Y-\mu_y)
$$

If we take variance on both side, we get:

$$
\mathbb{V}_{D \sim \mathcal{D}} \Bigg[\frac{X}{Y} \Bigg] \approx \Big(\nabla_X f(\mu_x, \mu_y) \Big)^2 \mathbb{V}[X] + \Big(\nabla_Y(\mu_x, \mu_y) \Big)^2 \mathbb{V}[Y] + 2 \nabla_X f(\mu_x, \mu_y) \nabla_Y f(\mu_x, \mu_y) \mathrm{Cov}(X, Y)
$$

Note that $\nabla_X f = \frac{1}{Y}$, therefore, $\nabla_X f(\mu_x, \mu_y) = \frac{1}{\mu_y}$. Similarly, $\nabla_Y f = \frac{X}{Y^2}$, therefore $\nabla_Y f(\mu_x, \mu_y) = \frac{\mu_x}{\mu_y^2}$. Using that, we can write out the equation above as below. We also assume $w_i \sim W, R(s_i, a_i) \sim R$ (where both $W$ and $R$ have randomness over $D$).

$$
\begin{align}
\mathbb{V}_{D \sim \mathcal{D}}\Bigg[\frac{X}{Y} \Bigg] &\approx \frac{1}{\mu_x^2}\frac{1}{N}\mathbb{E}\Big[(WR - \mu_x)^2 \Big] + \frac{\mu_x^2}{\mu_y^4} \frac{1}{N} \mathbb{E} \Big[(W - \mu_y)^2 \Big] \\ &- 2\frac{1}{\mu_y} \frac{\mu_x}{\mu_y^2} \mathbb{E} \Big[W^2 R \Big] + 2 \frac{1}{\mu_y} \frac{\mu_x}{\mu_y^2} \mathbb{E} \Big[WR \Big] \mathbb{E} \Big[W \Big] \\
&=\frac{1}{\mu_y^2} \frac{1}{N} \Bigg[ \mathbb{E} \Big[ W^2 R^2\Big] + \frac{\mu_x^2}{\mu_y^2} \Big[ W^2\Big] - 2 \frac{\mu_x}{\mu_y} \mathbb{E} \Big[ W^2 R\Big] \Bigg] \\
&= \frac{1}{\mu_y^2} \frac{1}{N} \mathbb{E} \Big[ (WR - \frac{\mu_x}{\mu_y}W)^2\Big] \\
&= \frac{1}{N} \frac{\mathbb{E} \Big[ W^2 (R - \frac{\mu_x}{\mu_y})^2 \Big]}{\mu_y^2}
\end{align}
$$

Note that $\mu_x / \mu_y = V^{\text{WIS}}$, therefore, we can write this as:

$$
\mathbb{V}_{D \sim \mathcal{D}} [\hat V^{\text{WIS}}] \approx \frac{1}{N}\frac{\mathbb{E}_{D \sim \mathcal{D}} \Bigg[ w_i^2 \Big(R(s_i, a_i) - \hat V^{\text{WIS}} \Big)^2 \Bigg]}{\mathbb{E}_{D \sim \mathcal{D}}\big[w_i \big]^2}
$$

The empirical variance estimator for this is:

$$
\begin{align}
\hat{\mathbb{V}}_{D \sim \mathcal{D}}[\hat V^{\text{WIS}}] &= \frac{1}{N} \frac{\frac{1}{N} \sum_{i=1}^N w_i^2 \Big(R(s_i, a_i) - \hat V^{\text{WIS}} \Big)^2 }{(\frac{1}{N} \sum_{i=1}^N w_i)^2} \\
&= \sum_{i=1}^N \Big(\frac{w_i}{\sum_{i=1}^N w_i}\Big)^2 \Big( R(s_i, a_i) - \hat V^{\text{WIS}} \Big)^2
\end{align}
$$

Note that unlike other variance estimator, we do not have the $1/N$ term after summation. 

Reference: https://www.stat.cmu.edu/~hseltman/files/ratio.pdf


