---
layout: post
comments: true
title: "CUAD Resource Management"
date: 2018-10-13 12:15:00
tags: generative-model
image: "three-generative-models.png"
---

> To control the army in your GPU!

<!--more-->

CUDA is the best way to utilize GPUs. For example, to  build a system, design a database or a deep learning system etc, CUDA can accelerate them and save some lives.

{: class="table-of-content"}
* TOC
{:toc}


## Processes Management

Here is a quick summary of the difference between GAN, VAE, and flow-based generative models:
1. Generative adversarial networks: GAN provides a smart solution to model the data generation, an unsupervised learning problem, as a supervised one. The discriminator model learns to distinguish the real data from the fake samples that are produced by the generator model. Two models are trained as they are playing a [minimax](https://en.wikipedia.org/wiki/Minimax) game.
2. Variational autoencoders: VAE inexplicitly optimizes the log-likelihood of the data by maximizing the evidence lower bound (ELBO).
3. Flow-based generative models: A flow-based generative model is constructed by a sequence of invertible transformations. Unlike other two, the model explicitly learns the data distribution p(\mathbf{x}) and therefore the loss function is simply the negative log-likelihood.

![Categories of generative models]({{ '/assets/images/three-generative-models.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Comparison of three categories of generative models.*


## Memory Management

We should understand two key concepts before getting into the flow-based generative model: the Jacobian determinant and the change of variable rule. Pretty basic, so feel free to skip.


### Communication and Cooperation

Given a function of mapping a $$n$$-dimensional input vector $$\mathbf{x}$$ to a $$m$$-dimensional output vector, $$\mathbf{f}: \mathbb{R}^n \mapsto \mathbb{R}^m$$, the matrix of all first-order partial derivatives of this function is called the **Jacobian matrix**, $$\mathbf{J}$$ where one entry on the i-th row and j-th column is $$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$.

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\[6pt]
\vdots & \ddots & \vdots \\[6pt]
\frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \\[6pt]
\end{bmatrix}
$$

The determinant is one real number computed as a function of all the elements in a squared matrix. Note that the determinant *only exists for **square** matrices*. The absolute value of the determinant can be thought of as a measure of *"how much multiplication by the matrix expands or contracts space".*

The determinant of a nxn matrix $$M$$ is:

$$
\det M = \det \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & & \vdots \\
a_{n1} & a_{n2} & \dots & a_{nn} \\
\end{bmatrix} = \sum_{j_1 j_2 \dots j_n} (-1)^{\tau(j_1 j_2 \dots j_n)} a_{1j_1} a_{2j_2} \dots a_{nj_n}
$$

where the subscript under the summation $$j_1 j_2 \dots j_n$$ are all permutations of the set {1, 2, ..., n}, so there are $$n!$$ items in total; $$\tau(.)$$ indicates the [signature](https://en.wikipedia.org/wiki/Parity_of_a_permutation) of a permutation.

The determinant of a square matrix $$M$$ detects whether it is invertible: If $$\det(M)=0$$ then $$M$$ is not invertible (a *singular* matrix with linearly dependent rows or columns; or any row or column is all 0); otherwise, if $$\det(M)\neq 0$$, $$M$$ is invertible.

The determinant of the product is equivalent to the product of the determinants: $$\det(AB) = \det(A)\det(B)$$. ([proof](https://proofwiki.org/wiki/Determinant_of_Matrix_Product))


### NCCL

Let's review the change of variable theorem specifically in the context of probability density estimation, starting with a single variable case. 

Given a random variable $$z$$ and its known probability density function $$z \sim \pi(z)$$, we would like to construct a new random variable using a 1-1 mapping function $$x = f(z)$$. The function $$f$$ is invertible, so $$z=f^{-1}(x)$$. Now the question is *how to infer the unknown probability density function of the new variable*, $$p(x)$$?

$$
\begin{aligned}
& \int p(x)dx = \int \pi(z)dz = 1 \scriptstyle{\text{   ; Definition of probability distribution.}}\\
& p(x) = \pi(z) \left\vert\frac{dz}{dx}\right\vert = \pi(f^{-1}(x)) \left\vert\frac{d f^{-1}}{dx}\right\vert = \pi(f^{-1}(x)) \vert (f^{-1})'(x) \vert
\end{aligned}
$$

By definition, the integral $$\int \pi(z)dz$$ is the sum of an infinite number of rectangles of infinitesimal width $$\Delta z$$. The height of such a rectangle at position $$z$$ is the value of the density function $$\pi(z)$$. When we substitute the variable, $$z = f^{-1}(x)$$ yields $$\frac{\Delta z}{\Delta x} = (f^{-1}(x))'$$ and $$\Delta z =  (f^{-1}(x))' \Delta x$$. Here $$\vert(f^{-1}(x))'\vert$$ indicates the ratio between the area of rectangles defined in two different coordinate of variables $$z$$ and $$x$$ respectively.

The multivariable version has a similar format:

$$
\begin{aligned}
\mathbf{z} &\sim \pi(\mathbf{z}), \mathbf{x} = f(\mathbf{z}), \mathbf{z} = f^{-1}(\mathbf{x}) \\
p(\mathbf{x}) 
&= \pi(\mathbf{z}) \left\vert \det \dfrac{d \mathbf{z}}{d \mathbf{x}} \right\vert  
= \pi(f^{-1}(\mathbf{x})) \left\vert \det \dfrac{d f^{-1}}{d \mathbf{x}} \right\vert
\end{aligned}
$$

where $$\det \frac{\partial f}{\partial\mathbf{z}}$$ is the Jacobian determinant of the function $$f$$. The full proof of the multivariate version is out of the scope of this post; ask Google if interested ;)


## Something Interesting

Being able to do good density estimation has direct applications in many machine learning problems, but it is very hard. For example, since we need to run backward propagation in deep learning models, the embedded probability distribution (i.e. posterior $$p(\mathbf{z})\vert\mathbf{x})$$) is expected to be simple enough to calculate the derivative easily and efficiently. That is why Gaussian distribution is often used in latent variable generative models, even through most of real world distributions are much more complicated than Gaussian. 

Here comes a **Normalizing Flow** (NF) model for better and more powerful distribution approximation. A normalizing flow transforms a simple distribution into a complex one by applying a sequence of invertible transformation functions. Flowing through a chain of transformations, we repeatedly substitute the variable for the new one according to the change of variables theorem and eventually obtain a probability distribution of the final target variable.

![Normalizing flow]({{ '/assets/images/normalizing-flow.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Illustration of a normalizing flow model, transforming a simple distribution $$p_0(\mathbf{z}_0)$$ to a complex one $$p_K(\mathbf{z}_K)$$ step by step.*


As defined in Fig. 2,

$$
\begin{aligned}
\mathbf{z}_{i-1} &\sim p_{i-1}(\mathbf{z}_{i-1}) \\
\mathbf{z}_i &= f_i(\mathbf{z}_{i-1})\text{, thus }\mathbf{z}_{i-1} = f_i^{-1}(\mathbf{z}_i) \\
p_i(\mathbf{z}_i) 
&= p_{i-1}(f_i^{-1}(\mathbf{z}_i)) \left\vert \det\dfrac{d f_i^{-1}}{d \mathbf{z}_i} \right\vert
\end{aligned}
$$

Then let's convert the equation to be a function of $$\mathbf{z}_i$$ so that we can do inference with the base distribution.

$$
\begin{aligned}
p_i(\mathbf{z}_i) 
&= p_{i-1}(f_i^{-1}(\mathbf{z}_i)) \left\vert \det\dfrac{d f_i^{-1}}{d \mathbf{z}_i} \right\vert \\
&= p_{i-1}(\mathbf{z}_{i-1}) \left\vert \det \color{red}{\Big(\dfrac{d f_i}{d\mathbf{z}_{i-1}}\Big)^{-1}} \right\vert & \scriptstyle{\text{; According to the inverse func theorem.}} \\
&= p_{i-1}(\mathbf{z}_{i-1}) \color{red}{\left\vert \det \dfrac{d f_i}{d\mathbf{z}_{i-1}} \right\vert^{-1}} & \scriptstyle{\text{; According to a property of Jacobians of invertible func.}} \\
\log p_i(\mathbf{z}_i) &= \log p_{i-1}(\mathbf{z}_{i-1}) - \log \left\vert \det \dfrac{d f_i}{d\mathbf{z}_{i-1}} \right\vert
\end{aligned}
$$

(\*) A note on the *"inverse function theorem"*: If $$y=f(x)$$ and $$x=f^{-1}(y)$$, we have:

$$
\dfrac{df^{-1}(y)}{dy} = \dfrac{dx}{dy} = (\dfrac{dy}{dx})^{-1} = (\dfrac{df(x)}{dx})^{-1}
$$


(\*) A note on *"Jacobians of invertible function"*: The determinant of the inverse of an invertible matrix is the inverse of the determinant: $$\det(M^{-1}) = (\det(M))^{-1}$$, [because](#jacobian-matrix-and-determinant) $$\det(M)\det(M^{-1}) = \det(M \cdot M^{-1}) = \det(I) = 1$$.


Given such a chain of probability density functions, we know the relationship between each pair of consecutive variables. We can expand the equation of the output $$\mathbf{x}$$ step by step until tracing back to the initial distribution $$\mathbf{z}_0$$.

$$
\begin{aligned}
\mathbf{x} = \mathbf{z}_K &= f_K \circ f_{K-1} \circ \dots \circ f_1 (\mathbf{z}_0) \\
\log p(\mathbf{x}) = \log \pi_K(\mathbf{z}_K) 
&= \log \pi_{K-1}(\mathbf{z}_{K-1}) - \log\left\vert\det\dfrac{d f_K}{d \mathbf{z}_{K-1}}\right\vert \\
&= \log \pi_{K-2}(\mathbf{z}_{K-2}) - \log\left\vert\det\dfrac{d f_{K-1}}{d\mathbf{z}_{K-2}}\right\vert - \log\left\vert\det\dfrac{d f_K}{d\mathbf{z}_{K-1}}\right\vert \\
&= \dots \\
&= \log \pi_0(\mathbf{z}_0) - \sum_{i=1}^K \log\left\vert\det\dfrac{d f_i}{d\mathbf{z}_{i-1}}\right\vert
\end{aligned}
$$

The path traversed by the random variables $$\mathbf{z}_i = f_i(\mathbf{z}_{i-1})$$ is the **flow** and the full chain formed by the successive distributions $$\pi_i$$ is called a **normalizing flow**. Required by the computation in the equation, a transformation function $$f_i$$ should satisfy two properties:
1. It is easily invertible.
2. Its Jacobian determinant is easy to compute.




![MAF and IAF]({{ '/assets/images/MAF-vs-IAF.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 10. Comparison of MAF and IAF.  The variable with unknown density is in green while the known one is in red.*

Computations of the individual elements $$\tilde{x}_i$$ do not depend on each other, so they are easily parallelizable (only one pass using MADE). The density estimation for a known $$\tilde{\mathbf{x}}$$ is not efficient, because we have to recover the value of $$\tilde{z}_i$$ in a sequential order, $$\tilde{z}_i = (\tilde{x}_i - \tilde{\mu}_i(\tilde{\mathbf{z}}_{1:i-1})) / \tilde{\sigma}_i(\tilde{\mathbf{z}}_{1:i-1})$$, thus D times in total.

{: class="info"}
| | Base distribution | Target distribution | Model | Data generation | Density estimation |
| ---------- | ---------- | ---------- | ---------- |---------- | ---------- |
| MAF | $$\mathbf{z}\sim\pi(\mathbf{z})$$ | $$\mathbf{x}\sim p(\mathbf{x})$$ | $$x_i = z_i \odot \sigma_i(\mathbf{x}_{1:i-1}) + \mu_i(\mathbf{x}_{1:i-1})$$ | Sequential; slow | One pass; fast |
| IAF | $$\tilde{\mathbf{z}}\sim\tilde{\pi}(\tilde{\mathbf{z}})$$ | $$\tilde{\mathbf{x}}\sim\tilde{p}(\tilde{\mathbf{x}})$$ | $$\tilde{x}_i  = \tilde{z}_i \odot \tilde{\sigma}_i(\tilde{\mathbf{z}}_{1:i-1}) + \tilde{\mu}_i(\tilde{\mathbf{z}}_{1:i-1})$$ | One pass; fast | Sequential; slow |
| ---------- | ---------- | ---------- | ---------- |---------- | ---------- |


## VAE + Flows

if we want to model the posterior $$p(\mathbf{z}\vert\mathbf{x})$$ as a more complicated distribution rather than simple Gaussian. Intuitively we can use normalizing flow to transform the base Gaussian for better density approximation. The encoder then would predict a set of scale and shift terms $$(\mu_i, \sigma_i)$$ which are all functions of input $$\mathbf{x}$$. Read the [paper](https://arxiv.org/abs/1809.05861) for more details if interested.

---

*Welcome to discuss and suggest about contents relative to  this blog*

See yah  :>


## Reference

[1] Danilo Jimenez Rezende, and Shakir Mohamed. ["Variational inference with normalizing flows."](https://arxiv.org/abs/1505.05770) ICML 2015.

[2] [Normalizing Flows Tutorial, Part 1: Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang.

[3] [Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows](https://blog.evjang.com/2018/01/nf2.html) by Eric Jang.

[4] [Normalizing Flows](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html) by Adam Kosiorek.

[5] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. ["Density estimation using Real NVP."](https://arxiv.org/abs/1605.08803) ICLR 2017.

[6] Laurent Dinh, David Krueger, and Yoshua Bengio. ["NICE: Non-linear independent components estimation."](https://arxiv.org/abs/1410.8516) ICLR 2015 Workshop track.

[7] Diederik P. Kingma, and Prafulla Dhariwal. ["Glow: Generative flow with invertible 1x1 convolutions."](https://arxiv.org/abs/1807.03039) arXiv:1807.03039 (2018).

[8] Germain, Mathieu, Karol Gregor, Iain Murray, and Hugo Larochelle. ["Made: Masked autoencoder for distribution estimation."](https://arxiv.org/abs/1502.03509) ICML 2015.

[9] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. ["Pixel recurrent neural networks."](https://arxiv.org/abs/1601.06759) ICML 2016.

[10] Diederik P. Kingma, et al. ["Improved variational inference with inverse autoregressive flow."](https://arxiv.org/abs/1606.04934) NIPS. 2016.

[11] George Papamakarios, Iain Murray, and Theo Pavlakou. ["Masked autoregressive flow for density estimation."](https://arxiv.org/abs/1705.07057) NIPS 2017.

[12] Jianlin Su, and Guang Wu. ["f-VAEs: Improve VAEs with Conditional Flows."](https://arxiv.org/abs/1809.05861) arXiv:1809.05861 (2018).

[13] Van Den Oord, Aaron, et al. ["WaveNet: A generative model for raw audio."](https://arxiv.org/abs/1609.03499) SSW. 2016.


