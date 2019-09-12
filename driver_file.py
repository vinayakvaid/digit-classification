import numpy as np
import random
import matplotlib.pyplot as plt
import classifier_nn
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

def main():
    """

    :return:
    """

    data = loadmat('./20by20 Data/ex4data1.mat')
    X = data["X"]
    y = data["y"]
    encoder = OneHotEncoder(sparse=False)
    y_matrix = encoder.fit_transform(y)
    train_data= (X,y_matrix)
    test_data = ()

    # # Loading and reading data
    # # The data is in such a way that "0" digit is labeled as "10", while
    # # the digits "1" to "9" are labeled as "1" to "9" in their natural order.
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #
    # # We will perform some checks to ascertain number of training, test and cross validation examples
    # # in this data set.
    # print()
    # print("Number of training examples : " + str(mnist.train.num_examples))
    # print("Number of test examples : " + str(mnist.test.num_examples))
    # print("Number of cross validation examples : " + str(mnist.validation.num_examples))
    # print("Number of pixel values in a single example : " + str(mnist.train.images[1].shape[0])),print()

    #Visualizing data
    for i in range(1,6):
        plt.figure(num=i)
    #     #plt.imshow(mnist.train.images[random.randint(1,int(mnist.train.num_examples))].reshape(28,28),cmap="gist_gray")
        plt.imshow(np.matrix(X[random.randint(1, 5000)]).reshape(20, 20),
                  cmap="gist_gray")
    plt.show()

    # Specifying X and y matrices for train and test data
    # train_data = mnist.train.next_batch(10000)
    X_train = train_data[0]
    y_train = train_data[1]
    print("Shape of X_train : " + str(X_train.shape))
    print("Shape of y_train : " + str(y_train.shape)),print()

    # test_data = mnist.test.next_batch(1000)
    # X_test = test_data[0]
    # y_test = test_data[1]
    # print("Shape of X_test : " + str(X_test.shape))
    # print("Shape of y_test : " + str(y_test.shape)),print()

    # Calling neural network model
    layers = [400,25,10]
    classifier_nn.model_nn(train_data,test_data,layers,y)

main()


