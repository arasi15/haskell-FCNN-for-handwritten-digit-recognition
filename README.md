# haskell-FCNN-handwritten-Recognition

This is a course project of haskell.  
A fully connected neural network is realized for handwritten recognition trained on MNIST dataset.  

## Dataset  

MNIST: 28x28

## FCNN architecture   



![image.png](https://i.loli.net/2020/07/21/3Q2dNtIfpgmUZiA.png)



| input   | fc-layer1 | fc-layer2 | fc-layer3 |
| ------- | --------- | --------- | --------- |
| 28x28x1 | 784x256   | 256x256   | 256x10    |

## Training Setup

|     Setup     | Content |
| :-----------: | :-----: |
|   Optimizer   |   SGD   |
| Learning Rate |  0.002  |
|   Iteration   |  10000  |

## Loss Function

1. $Softmax=\frac{e^{x_i}}{\sum_{0}^{9}{e^{j}}}$
2. $CrossEntropy=\sum_{0}^{9}-label_{i}\times ln(softmax_{i})$

## BackPropagation

1. $\frac{dLoss}{drelu\_feat3_{i}}=\frac{-d\sum{y_i\times ln(\frac{e^{x_{i}}}{\sum{e^{x_{j}}}})}}{dx_k}=\sum{y_i}\frac{dln(\sum{{e^{x_{j}}}})-dln(e^{x_i})}{dx_k}=\sum{y_i}\times(\frac{e^{x_k}}{\sum{e^{x_j}}}-\frac{dx_i}{dx_k})=\sum{y_i}\times(softmax_k-\frac{dx_i}{dx_k})=softmax_k-label_k$
2.$\frac{drelu\_{f_k}}{df_k}=\begin{cases}0 f<0\\1 f>0\\  \end{cases}$
3. $\frac {dLoss} {dfeat3}=relu^′ (softmax)-label$
4. $\frac{dLoss}{dW3}=\frac{dLoss}{dfeat3}×\frac{dfeat3}{dW3}=[relu\_feat2]^T×[relu^′ (softmax)-label]$
5. $\frac{dLoss}{dW2}=\frac{dLoss}{dfeat3}×\frac{dfeat3}{dfeat2}×\frac{dfeat2}{dW2}=[relu_feat1]^T×relu^′ ([relu^′ (softmax)-label]×[W3]^T)$
6. $\frac{dLoss}{dW1}=\frac{dLoss}{dfeat3}×\frac{dfeat3}{dfeat2}×\frac{dfeat2}{dfeat1}×\frac{dfeat1}{dW1}=[input]^T×relu′\{relu^′ [[relu^′ (softmax)-label]×[W3]^T ]×W2^T\}$

## Accuracy

90.14%

![image.png](https://i.loli.net/2020/07/21/heTS5kWZqQwB74l.png)

## Reference

https://www.cnblogs.com/simbon/p/8040432.html
