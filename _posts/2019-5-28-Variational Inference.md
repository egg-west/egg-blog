---
layout: post
comments: true
title: "Procedure of Variational Inference"
date: 2019-5-26 00:04:00
tags: probability
image:
---

> Proof-friendly Bayesian inference

<!--more-->
## Recap of maximum likelihood:
There is a presumable parametralized distribution $P(x|\theta)$, and some sample $${x_i}$$. Maximum likelihood is a tool to estimate the most likely $$P(x|\theta)$$ (by ensure the $$\theta$$)

$$
L(\theta) = \mathop{\Pi}^{n}_{t=1} p(x_i|\theta)\\
\max_{\theta}\ L(\theta)
$$

In a simple case, the $$\theta$$ can be solve by $$\frac{\delta\ log\ L(\theta)}{\delta \theta}$$.

## Variational Inference

VI is a part of Variational Bayes theory.
**Formulation** We observed dataset(y), and weights($$\theta$$) of an approximator. $$p(\theta| y_{1:N})$$ is the **posterior** and $$p(\theta)$$ is the **prior**.  $$p(y_{1:N} | \theta)$$ is the **likelihood**. 

Several choices at this time:
1. build a model: choose prior & choose likelihood 

2. compute the posterior **computational complexity: high dimension data/weights**
3.  report a summary, e.g. posterior means and (co)variances **computational complexity**

Our goal is to find a optimal $$q^*(\theta)$$  for $$p(\theta| y)$$ (the posterior), and one way to do this is to shorten the **distance** between $$q()$$ and $$p()$$. 
1. Gold standard: MCMC **eventually accurate but can be slow**

2. **Variational Bayes  (VB)**: $f$ is Kullback-Leibler divergence $KL(q(\cdot) || p(\cdot | y)) = q log(q/p)$, so we got

   $$
   q^* = \mathop{argmin}_{q\in Q} KL(q(\cdot) || p(\cdot|y))\\
   
   KL(q||p) := \int q(\theta)\ log \frac{q(\theta)}{p(\theta|y)} d\theta \\
   
   := \int q(\theta)log \frac{q(\theta) p(y)}{p(\theta, y)} d\theta = \int q(\theta) \log p(y) d\theta - \int q(\theta) \log(p/q) d\theta \\
   = \log p(y) - \int q(\theta) \log \frac{p(\theta, y)}{q(\theta)} d\theta
   $$

   And we find the the second term of the last equation contains known things. This is called **Evidence lower bound (ELBO)**. A trick in the last step is that, for a distribution $$q(\theta)$$, the integral would be 1. 

   Then we have $$KL \leq 0$$ (exercise Bishop 2006 Sec 1.6.1), so $$\log p(y) \leq ELBO$$ . $$q^* = argmax _{q\in Q} ELBO(q) $$

   > Why KL?
   >
   > Super easy derivative

Choice of "NICE" distributions (assume a distribution)

**Mean-field variational Bayes (MFVB) ** to approximate $$q(\theta)$$

$$
Q_{MFVB} := {q: q(\theta) = \mathop{\Pi}^{J}_{j=1} q_j(\theta_j)}
$$

- often also exponential family
- *Not* a modeling assumption (**Emphasize!!**). Only some approximation of model's factorization.

## How to solve

Now, we have problem: 

$$
q^* = \mathop{argmin}_{q\in Q_{MFVB}} KL(q || p( |y))
$$

There are solutions:

- Coordinate descent **(slow)**
- Stochastic variational inference (SVI) [Hoffman et al., 2013]
- Automatic differentiation variational inference (ADVI) [kucukellbir et al., 2017]

## Relative term

**Latent Variables**: invisible variables that makes the model more tractable

**Bayesian networks**, the weight of a NN is a distribution, so every time to do forward computation, the results can be different

## Ref

[icml VI tutorial](https://www.youtube.com/watch?v=DYRK0-_K2UU) that answer more questions

> why mean filed [46:30]

[deep learning with VI](https://www.youtube.com/watch?v=h0UE8FzdE8U)

