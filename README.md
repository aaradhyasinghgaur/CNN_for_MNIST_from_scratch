# CNN_for_MNIST_from_scratch

# Table of Contents
1. [Introduction](#introduction)
2. [Tools and Technology](#toolsandtechnology)
3. [Network Architecture](#networkarchitecture)
4. [Training and Testing](#trainingandtesting)

## Introduction
This repository contains source code for Convolutional Neural Network to train and predict MNIST dataset which is a dataset for handwritten digits from 0-9 , using C++ as programming language. It is purely made using standard library for C++ without any third party library . It contains all the basic features of a CNN such as multiple layers.

## Tools and Technology
C++ , Standard Library for C/C++ , VS Code Editor , Machine Learning , Image/data Processing .

## Network Architecture
This framework contains multiple layers of Convolutional Neural Network such as convolution layer , fully connected layer (dense layer) , activation layer , activation functions etc. It also contains a data handling file to store the binary data of MNIST in suitable data structure for easier processing when passed among different layers.

Below is the brief discription of each file/layer and its function :-

**1.) data.h -** This container encapsulates the data in feature vector and label of each image of size 28 * 28 and declares some setter and getter functions to set and retrieve the data whenever asked.

**2.) datahandler.h -** It handles the data (feature vector , label) and stores it a data structure. It describes several functions such as reading and storing the feature vector and label and splitting the data according to our needs and counting classes.

**3.) layer.h -** This is an abstract class which declares basic functions such as forward and backward propogation of each layer in a network.

**4.)

