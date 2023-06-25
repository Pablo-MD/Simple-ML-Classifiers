# Simple-ML-Classifiers
This file contains a simple implementation of 3 simple yet commonly used classifiers in the field of data analysis and machine learning, these being a Naïve Bayes, a Logistic Regression and One vs Rest Classifiers. 

These implementations are made in python with the only use of numpy (and a function from scipy). They are not intended to be a substitute for obviously optimize and efficient implementation found in modules such as sklearn, but rather to showcase how these models work in a simple way.

All 3 of them are instantiate and then recieve a set of parameters and a dataset used in a training phase, once training, they call the method classify recieving a test dataset to return an output with the predictions.
