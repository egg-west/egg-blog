---
layout: post
comments: true
title: "Neuromodulation: A new approach for meta learning"
date: 2019-1-1 00:04:00
tags: generative-model
image: "neuromodulation_exps.jpg","neuromodulation_connection.jpg"
---

> Another inspiration comes from neuro-science is applied on state-of-the-art problem 

<!--more-->

Reinforcement Learning once been denounced by training on environment it is test. Things got different now. Want to read more about Meta learning? This [blog](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html) written by an Openai researcher is my best recommendation.

**Meta learning** methods are trying to find a way to "learn to learn", which can be treated as how to modify the parameter of a trained model to fit the new environment ASAP. Specifically, meta Reinforcement learning requires a guider to guide the agent and enable the agent **adaptive behavior**.

The solution of this paper called **Neuromodulation** as shown in fig 1 below, the top network is the controller mentioned above and this sort of connection are called **Neuromodulatory connecction.**  This is inspired by a kind of biological structure (Bargmann & Marder(2013), Marder et al.(2014) that is relative to continuous control of human, motor driving for example.


## Meta-RL problem setting

Since there is a controller now, we may have different policy in every different episode. So we got $\pi^i$ to notate the policy on $i_{th}$ episode (environment).

## Methods
blank

## Experiments

The researchers build up 3 different difficulties and type of meta-learn tasks. NMD Net is implemented in Actor-Critic algorithm. Seen from below results we can see that in all 3 tasks, NMD Net gains slightly higher reward than RNN baseline and NMD Net is far more stable. I noticed that RNN is getting more unstable through the steps, maybe it is caused by RNN's limit on its capture ability of **long-term dependence**. On the contrary, NMD Net gets more stable through the steps.


---