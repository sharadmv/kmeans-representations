import pickle
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from path import Path

from spherecluster import SphericalKMeans

from kmeans import load_cifar10, ZCA

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--num-centers', type=int, default=20)
    argparser.add_argument('--top-centers', type=int, default=None)
    argparser.add_argument('--random', action='store_true')
    argparser.add_argument('--seed', type=int, default=1337)
    argparser.add_argument('--logfile', type=str)

    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # mnist = fetch_mldata('MNIST original')
    # X, y = mnist.data / 255., mnist.target
    # Xtrain, Xtest = X[:60000], X[60000:]
    # ytrain, ytest = y[:60000], y[60000:]
    (Xtrain, ytrain), (Xvalid, yvalid), (Xtest, ytest) = load_cifar10()
    Xtrain, Xvalid, Xtest = Xtrain / 255., Xvalid / 255., Xtest / 255.
    Xtrain = Xtrain.reshape([Xtrain.shape[0], -1])
    Xvalid = Xvalid.reshape([Xvalid.shape[0], -1])
    Xtest = Xtest.reshape([Xtest.shape[0], -1])

    zca = ZCA().fit(Xtrain)
    Xtrain = zca.transform(Xtrain)
    Xtest = zca.transform(Xtest)

    print("Training classifier...")
    cf = LogisticRegression()
    cf.fit(Xtrain, ytrain)
    train_accuracy = cf.score(Xtrain, ytrain)
    test_accuracy = cf.score(Xtest, ytest)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
