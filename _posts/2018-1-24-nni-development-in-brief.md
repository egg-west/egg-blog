---
layout: post
comments: true
title: "NNI development: In brief"
date: 2019-1-22 00:04:00
tags: generative-model
image: "neuromodulation_exps.jpg"
---

> What you need to know is $O(\theta)$ and $\Omega(\theta)$, NNI will handle every other things.

<!--more-->

AutoML seems to be far away from personal research. But now it comes like a thunder sort of hamburger, that not only have structure but also the material: it works and easy!

## Go through NNI

`nnictl config.yml mask-r-cnn.py` this magical command will slash a slight wave on east coast of China and make a tsunami on the whole Pacific, coming with this are simple simple web UI with inspect diagrams. Life never gets better. 

The first time I got touch with AutoML is a [tool](https://www.automl.org/) produced by University of Freiburg and sadly to say that was not-opensource. Not soon after that Neural Architecture Search (NAS) got very popular and researchers proposed diverse generative algorithm for generating nice model in efficiency. Several months ago, I met AutoKeras(ref) and feel quiet interested but not that happy to use it in my work. At this time, everything gets different. NNI is a complete system for AutoML rather than a tool.

The source code of NNI contains 3 parts, dividing by different element in NNI system. `nni_manager/`, `sdk/`, `webui/`. `nni_manager/` includes codes for backend of nni system,  maintaining some **typescript** code for service while training. `webui` maintains frontend code of webUI. `sdk/` are the director for python scripts that package the whole system into a python package, in which the *tuners* and *assessor* are placed here. Except for these 3 parts, codes for `nnictl` lands on this dir. 

NNI is under develop so during exploring NNI, I met little inconvenient. When I tried to use a tunner that requires outside lib, the `nnictl` shows nothing different from works fine. So I spend some time to [fix it and made a commit](https://github.com/Microsoft/nni/commit/0405a426ccbd330d4577e14bfdfbcb987657809c). 



## A brief look at this system: with  the mnist-annotation example

This part is down out of picture issue. would be fixed ASAP





## Go in development - tuner 

> grid search is the best, no debate approved.

Developers have implemented n tuners and all placed in `source/sdk/pynni/nni` . These diverse tuners can satisfy different training framework and tuning tools.

To clarify the way to develop a tuner, a ABCTuner shows below to make a abstract-level glance at nni tuners.

```python
from nni.tuner import Tuner
import nni.metis_tuner.lib_data as lib_data
import nni.metis_tuner.lib_constraint_summation as lib_constraint_summation
# optional
from enum import Enum, unique
from multiprocessing.dummy import Pool as ThreadPool

### at leat 3 function requires implementation
class ABCTuner(Tuner):
	def __init__(self):
		pass

    def generate_parameters(self, parameter_id):
        pass

    def receive_trial_result(self, parameter_id, parameters, value):
        pass
    def update_search_space(self, search_space):
        """Update the search space of tuner. Must override.
        	search_space: JSON object
        """
        pass

    '''optional'''
    def is_valid(self, search_space):
        '''
        Check the search space is valid: only contains 'choice' type
        '''
        if not len(search_space) == 1:
            raise RuntimeError('BatchTuner only supprt one combined-paramreters key.')
        
        for param in search_space:
            param_type = search_space[param][TYPE]
            if not param_type == CHOICE:
                raise RuntimeError('BatchTuner only supprt one combined-paramreters type is choice.')
            else:
                if isinstance(search_space[param][VALUE], list):
                    return search_space[param][VALUE]
                raise RuntimeError('The combined-paramreters value in BatchTuner is not a list.')
        return None
```

The most simple tuner is *BatchTuner*, searching hyperparameter in an interval. Given an interval for learning rate $r \in {x| x \in [0.0001, 0.1]}$, *BatchTuner* would search inside. Similar, *GridSearch* is one of the most common tuner to find an optimal combination of hyperparameters. 

Except for some linear search, to satisfy exist work, hyperopt, for example, *HyperoptTuner* is implemented for lending powerful algorithm in hyperopt. hyperopt is a popular tool for hyperparameter tunning, which works well in distributed system.

*EvolutionTuner* applies simple evolution theory on parameter search.

*NetworkmorphismTuner* utilizes 

*metisTuner* is recently added by researcher HuiXue in System group, MSRA. Got attention when it's under review.This tuner is based on work (ref) that makes a tuner for optimizing **tail latencies** in cloud system.

***Besides implemented tuners, several interesting work is reviewed below.***

iTuned(ref) is a parameter tuner for improve **I/O performance** for databases. The formulation is as below

$X^*  = argmin_{1\leq i \leq n} y(X^{(i)}) $

In which $X^*$ is the optimal parameter. We define a *improvement* $IP(X)$
$$
IP(X) = y(X^*) - y(X)\ \ \  if\ y(X)<y(X^*)\\
=0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ otherwise
$$


The main algorithm is listed below:

```
1 Initialize based on Latin Gypercube Sampling
2 before stopping conditino, do
3	sample X_next = argmax(EIP(X))  x\in DOM
4	Exucute the experiment with X_next and got new sample
5	update the GRS and X* with the new sample. Go to Line2;

```





## Go in development - assessor 

> Good judgement comes from experience, and experience comes from bad judgement.	
>
> ​																--Jim Horning



## Reference

[1] Auto-Keras: Efficient Neural Architecture Search with Network Morphism. Haifeng Jin et, al.

[2] Tuning database configuration Parameters with iTuned, Songyun Duan et, al.

















