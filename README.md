# MSc-Multiscale-image-representation-in-Deep-Learning-code
The code used in mini-dissertation entitle 'Multiscale image representation in Deep Learnin'. This is the code used to decompose the images and to train and compare the neural networks. 

rmpa - This is a python class written by Mark de Lancey and can be found on his Github repository

DPV.py - Applies DPT to all the images in parallel, determine eacg pixel of each images DPV and stores these DPVs as DogDPV.gz and CatDPV.gz

NeuralNetOG.py - Loads the original images, does some image preprocessing and then builds and train an neural network on these images. The model is saved as NN_OG.h5

NeuralNetDPV.py - Loads the original and augmented images, does some image preprocessing and then builds and train an neural network on these images. The model is saved as NN_DPV.h5

TestProbs.py - Loads the models and uses a validation set to determine an otimal cut-off using the predicted probabilities to increase the performance of the model.
