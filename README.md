# Naive Bayes classifier
This repository contains implemention of Gaussian and Multinomial Naive Bayes Classificator using maximum a posteriori estimation.

***

## What is Bayes classifier

In task of classification for $n$ pairs of observations and target values, the goal is to assign new observation to the most probable unknown before target value. Observation can be described as vector $x$ containg $p$ features and target values as $y$ belonging to one of $k$ classes. Solution to this task is modelling probability of belonging to a given class based on set features what can be written as $P(Y = k | X = x) $. Thanks to the Bayes theorem this conditional probability called *a posteriori* can be written as
$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}.$$
$P(Y)$ is *a priori* probability and can be unsderstood as frequency of occuring for given class, $P(X|Y)$ is called likelihood. Since goal is to maximize *a posteriori* so there's only a need to maximize nominator, this approach is called **maximum a posteriori estimation**. Other approach is to maximize only $P(X|Y)$ clause which is **maximum likelihood estimation** but this estimation is not implemented in the repository.  
Whereas calculating $P(Y)$ is an easy task, calculating likelihood distributions for each class is computanionaly exspensive because there are multidimensional distributions for every combination of features values.
$$P(x_1,x_2,...,x_p|k)$$

## Naive assumption

Naive assumption states that, every feature is independent and that gives *p* distributions for each class.
$$P(x_1|k) \cdot P(x_2|k) \cdot ... \cdot P(x_p|k)$$
For easier computations it is better to take logarithm of this value, what is called **loglikelihood**. 
$$log(P(x_1|k)) + log(P(x_2|k)) + ... + log(P(x_p|k))$$

## Gaussian Naive Bayes

This approach assumes that every $P(x_i|k)$ distribution is gaussian so it is needed to find mean value $\mu$ and standard deviation $\sigma$ for every feature.
$$P(x_i|k) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x_i - \mu}{\sigma} \right)^2\right)$$

## Multinomial Naive Bayes

In this approach every distibuiton $P(x_i|k)$ is represented as normalized histogram, it means that every continous feature must be represented as discrete values representing ranges called bins.
