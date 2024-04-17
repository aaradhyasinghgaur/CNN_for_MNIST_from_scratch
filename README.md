# CNN_for_MNIST_from_scratch

# Table of Contents
1. [Introduction](#introduction)
2. [Tools and Technology](#toolsandtechnology)
3. [Network Architecture](#networkarchitecture)
4. [Training and Testing](#trainingandtesting)
   
-----

## Introduction
This repository contains source code for Convolutional Neural Network to train and predict MNIST dataset which is a dataset for handwritten digits from 0-9 , using C++ as programming language. It is purely made using standard library for C++ without any third party library . It contains all the basic features of a CNN such as multiple layers.

-----------

## Tools and Technology
C++ , Standard Library for C/C++ , VS Code Editor , Machine Learning , Image/data Processing .

---------

## Network Architecture
This framework contains multiple layers of Convolutional Neural Network such as convolution layer , fully connected layer (dense layer) , activation layer , activation functions etc. It also contains a data handling file to store the binary data of MNIST in suitable data structure for easier processing when passed among different layers.

Below is the brief discription of each file/layer and its function :-

**1.) data.h -** This container encapsulates the data in feature vector and label of each image of size 28 * 28 and declares some setter and getter functions to set and retrieve the data whenever asked.

**2.) datahandler.h -** It handles the data (feature vector , label) and stores it a data structure. It describes several functions such as reading and storing the feature vector and label and splitting the data according to our needs and counting classes.

**3.) layer.h -** This is an abstract class which declares basic functions such as forward and backward propogation of each layer in a network.

**4.) convolution.h -** It initialises dimensions of input , kernel(filters) , output and random values for weights and biases. It introduces forward and backward propogation functions for co-relation and convolution operations . It is used to learn spatial hierarchies of features from input data.

**5.) activation.h -** It is an activation layer which is defined after each layer which introduces non-linearity into the model , for model to learn more complex features to predict correct class for each data.

**6.) activation_functions.h -** Several activation functions are introduced in this file for various different machine learning tasks , such as tanh , ReLu , Sigmoid and Softmax . For MNIST dataset particularly which is multi-class classification , functions like Sigmoid and Softmax are used.

**7.) dense.h -** It is also known as fully-connected layer . Flatten data (multi-dimensional feature maps into a one-dimensional array) is passed through dense layer for the final prediction of the output class.

**8.) loss.h -** Various different loss/cost functions are defined in this file which are used for several different machine learning tasks , such as mse(Mean squared error) and binary-cross entropy which are used for binary clasification. For MNIST particularly catagorical cross-entropy is used to measure the error for each data-point(image) using actual probability and predicted probability for each class .

**9.) matrix.h -** It includes matrix realted operations such as transpose , initialising matrix with random values and falttening the data.

**10.) network.h -** This file contains network architecture for our model with various functions such as reading the training and test dataset and training , validation and test functions to train and test the model.

-----------

## Training and Testing 

**network.cpp -** It includes training and testing functions to train and test the dataset.

**Training -** The dataset contains 60,000 images for training and 10,000 for testing . 
I've splitted the training dataset further into two parts validation dataset for 6000 images and training dataset for 54,000.
For each mini batch of 2000 images the model is being trained for epoch = 10 and learning rate = 0.001
After training the complete dataset of 54,000 images the validation set giing the accuracy of 96.75 %.

![Example Image](https://github.com/kyra-09/CNN_for_MNIST_from_scratch/blob/main/Screenshot%20(152).png)

**Testing -** On testing the dataset of 10,000 images it is giving final accuracy of 97 % which is described is more than good accuracy for MNIST dataset.

-----------





