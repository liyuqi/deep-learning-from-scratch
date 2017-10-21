# ゼロから作る Deep Learning

---

![表紙](https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png)

---

本リポジトリはオライリー・ジャパン発行書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』のサポートサイトです。

Contents 

Preface 

### Chapter 1 Introduction to Python 
#### 1.1 What is Python 
#### 1.2 Python Installation 
* 1.2.1 Python Version 
* 1.2.2 External Library to Use 
* 1.2.3 Anaconda Distribution 
#### 1.3 Python Interpreter 
* 1.3.1 Arithmetic Calculation 
* 1.3.2 Data Types 
* 1.3.3 Variables 
* 1.3.4 List 
* 1.3.5 Dictionary 
* 1.3.6 Boolean 
* 1.3.7 if statement 
* 1.3.8 for statement 
* 1.3.9 Function 
#### 1.4 Python script file 
* 1.4.1 Save in file 
* 1.4.2 Class 
#### 1.5 NumPy 
* 1.5.1 NumPy import 
* 1.5.2 Generation of NumPy sequence 
* 1.5.3 Arithmetic calculation of NumPy 
* 1.5.4 N-dimensional array of NumPy
* 1.5.5 Broadcast 
* 1.5.6 Element Access 
#### 1.6 Matplotlib 
* 1.6.1 Simple Graph Drawing 
* 1.6.2 pyplot Function 
* 1.6.3 Image Display 
#### 1.7 Summary 

### Chapter 2 Perceptron 
#### 2.1 What is the Perceptron 
#### 2.2 Simple Logic Circuit 
* 2.2.1 AND Gate 
* 2.2.2 NAND Gate and OR Gate 
#### 2.3 Perceptron Implementation 
* 2.3.1 Simple Implementation 
* 2.3.2 Introduction of Weights and Bias 
* 2.3.3 Implementation by Weight and Bias 
#### 2.4 Limitations of Perceptron 
* 2.4.1 XOR Gate 
* 2.4.2 Linear and Nonlinear 
#### 2.5 Multilayer Perceptron 
* 2.5.1 Combination of Existing Gates 
* 2.5.2 Implementation of XOR Gate 
#### 2.6 NAND to Computer 
#### 2.7 Summary 

### Chapter 3 Neural Network
#### 3.1 Perceptron to Neural Network  
* 3.1.1 Neural network example
* 3.1.2 Review of Perceptron 
* 3.1.3 Activation Function Appearance 
#### 3.2 Activation Function 
* 3.2.1 Sigmoid Function 
* 3.2.2 Step Function Implementation 
* 3.2.3 Step Function Graph 
* 3.2.4 Implementation of Sigmoid Function 
* 3.2.5 Comparison of sigmoid function and step function 
* 3.2.6 Nonlinear function 
* 3.2.7 ReLU function 
#### 3.3 Multidimensional array calculation 
* 3.3.1 Multidimensional array 
* 3.3.2 Matrix inner product 
* 3.3.3 Inner product of neural network 
#### 3.4 Three layer neural network implementing 
* 3.4.1 symbols confirmation 
* 3.4.2 implementation of signal transmission in each layer summary 
* 3.4.3 implementation design 
#### 3.5 output layer 
* 3.5.1 identity function and softmax function Notes on mounting 
* 3.5.2 softmax function 
* 3.5.3 Features of SoftMax Function neurons 
* 3.5.4 output layer Number of 
#### 3.6 Handwritten Numeral Recognition
* 3.6.1 MNIST data set 
* 3.6.2 Neural network inference processing 
* 3.6.3 Batch processing 
#### 3.7 Summary 

### Chapter 4 Learning neural network 
#### 4.1 Learning from data 
* 4.1.1 Data driving 
* 4.1.2 Training data and test data 
#### 4.2 Loss function 
#### 4.2. 1 squared error 
* 4.2.2 cross entropy error 
* 4.2.3 mini batch learning 
* 4.2.4 [batch supported version] implementing cross entropy error 
* 4.2.5 why set loss function? 
#### 4.3 Numerical differentiation 
* 4.3.1 Differential 
* 4.3.2 Example of numerical differentiation 
* 4.3.3 Partial derivative 
#### 4.4 Gradient 
* 4.4.1 Gradient method 
* 4.4.2 Slope for neural network 
#### 4.5 Implementation of learning algorithm  
* 4.5.1 Class of two-layer neural network
* 4.5.2 Implementation of mini batch learning 
* 4.5.3 Test data Evaluation 
#### 4.6 Summary

### Chapter 5 Error Back Propagation Method 
#### 5.1 Calculation Graph 
* 5.1.1 Solving in Calculation Graph 
* 5.1.2 Local Calculation 
* 5.1.3 Why is it solved with calculation graph? 
#### 5.2 Chain Rate 
* 5.2.1 Back Propagation of Calculation Graph 
* 5.2.2 Chain Rate 
* 5.2.3 Chain Rate and Calculation Graph 
#### 5.3 Back Propagation 
* 5.3.1 Back Propagation of Addition Node 
* 5.3.2 Back Propagation of Multiplication Node 
* 5.3.3 Example 
#### 5.4 Simple Layer Implementation 
* 5.4.1 Multiplication Layer Implementation 
* 5.4.2 Implementation of Addition Layer 
#### 5.5 Activation Function Layer Implementation 
* 5.5.1 ReLU Layer 
* 5.5.2 Sigmoid Layer 
#### 5.6 A.ne / Softmax Layer Implementation 
* 5.6.1 A.ne layer  
* 5.6.2 Batch version A.ne layer
* 5.6.3 Softmax-with-Loss layer 
#### 5.7 Implementation of error back propagation method 
* 5.7.1 Overall view of learning of neural network 
* 5.7.2 Implementation of neural network corresponding to 
#### error back propagation method# 5.7.3 Confirmation of gradient of 
#### error back propagation method# 5.7.4 Learning using error back propagation method 
#### 5.8 Summary 

### Chapter 6 learning Technique 
#### 6.1 Updating parameters 
* 6.1.1 Talk of adventurers 
* 6.1.2 SGD 
* 6.1.3 Disadvantages of SGD 
* 6.1.4 Momentum 
* 6.1.5 AdaGrad 
* 6.1.6 Adam 
* 6.1.7 Which update method to use? 
* 6.1.8 Comparison of updating methods with MNIST dataset 
#### 6.2 Initial value of weight 
* 6.2.1 Setting initial value of weight to 0? 
* 6.2.2 Activation distribution of hidden layer 
* 6.2.3 Initial value of weight for ReLU 
* 6.2.4 Comparison of weight initial values by MNIST data set 
#### 6.3 Batch Normalization 
* 6.3.1 Algorithm of Batch Normalization
* 6.3.2 Evaluation of Batch Normalization 
* 7.4.2 Development by im2col 
#### 6.4 Regularization 
* 6.4.1 Over learning
* 6.4.2 Weight decay 
* 6.4.3 Dropout 
#### 6.5 Verification of hyperparameters 
* 6.5.1 Verification data 
* 6.5.2 Optimization of hyperparameters 
* 6.5.3 Implementation of hyperparameter optimization 
#### 6.6 Summary 

### Chapter 7 Convolutional neural network 
#### 7.1 Overall structure 
#### 7.2 Convolutional layer 
* 7.2.1 Problems with all tie layers 
* 7.2.2 Convolution operation 
* 7.2.3 Padding 
* 7.2.4 Stride 
* 7.2.5 Three-dimensional data convolution operation 
* 7.2.6 Thinking in blocks 
* 7.2.7 Batch processing 
#### 7.3 Pooling layer 
* 7.3.1 Pooling layer Features 
#### 7.4 Convolution / Pooling layer implementation 
* 7.4.1 Four-dimensional array 
* 7.4.4 Pooling layer 
#### 7.5 Implementation of CNN
* 7.4.3 Convolution layer implementation 
#### 7.6 CNN visualization 
* 7.6.1 First layer weight visualization 
* 7.6.2 Information extraction by hierarchical structure 
#### 7.7 Typical CNN 
* 7.7.1 LeNet 
* 7.7.2 AlexNet 
#### 7.8 Conclusion 

### Chapter 8 Deep learning 
#### 8.1 Deepening the network 
* 8.1.1 To a deeper network
* 8.1.2 To further increase recognition accuracy 
* 8.1.3 Motivation to deepen layers 
#### 8.2 Small history of deep learning 
* 8.2.1 ImageNet 
* 8.2.2 VGG 
* 8.2.3 GoogLeNet 
#### 8.2 .4 ResNet 
#### 8.3 Acceleration of deep learning 
* 8.3.1 Problems to be addressed 
* 8.3.2 Acceleration by GPU 
* 8.3.3 Distributed learning 
* 8.3.4 Bit reduction of calculation accuracy 
#### 8.4 Practical example of deep learning
* 8.4.1 Object detection  
* 8.4.2 Segmentation
* 8.4.3 Image caption generation 
#### 8.5 Future of deep learning 
* 8.5.1 Image style conversion 
* 8.5.2 Image generation 
* 8.5.3 Automatic operation 
* 8.5.4 Deep Q-Network (reinforcement learning) 
#### 8.6 Conclusion 

### Appendix A Softmax-with - Loss layer calculation graph 
* A.1 Propagation 
* A.2 Backpropagation 
* A.3 Summary 

#### Reference 
* Python / NumPy 
* calculation graph (error back propagation method) 
* Online lesson (Deep Learning) 
* method of updating 
* parameter Initial value of weight parameter 
* Batch Normalization / Dropout 
* Hyperparameter 
* Optimization 
* CNN Visualization 
* Representative Network 
* Dataset 
* Calculation Acceleration MNIST Dataset Accuracy Ranking and Maximum * Accuracy Approach 
* Deep Learning Application




ソースコードの解説は本書籍をご覧ください。

## 必要条件
ソースコードを実行するには、下記のソフトウェアがインストールされている必要があります。

* Python 3.x
* NumPy
* Matplotlib

※Pythonのバージョンは、3系を利用します。

## 実行方法

各章のフォルダへ移動して、Pythonコマンドを実行します。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。

## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。

