import pickle
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from path import Path

from spherecluster import SphericalKMeans

from zca import ZCA

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

    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data / 255., mnist.target
    Xtrain, Xtest = X[:60000], X[60000:]
    ytrain, ytest = y[:60000], y[60000:]

    zca = ZCA().fit(Xtrain)
    Xtrain = zca.transform(Xtrain)
    Xtest = zca.transform(Xtest)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.random:
        centers = Xtrain[np.random.choice(Xtrain.shape[0], replace=False, size=args.num_centers)]
    else:
        kmeans = SphericalKMeans(args.num_centers, verbose=1, n_init=5)
        dump_path = Path("out") / ("kmeans-%u-%u.pkl" % (args.num_centers, args.seed))
        if dump_path.exists():
            with open(dump_path, 'rb') as fp:
                centers = pickle.load(fp)
        else:
            kmeans.fit(Xtrain)
            centers = kmeans.cluster_centers_
            with open(dump_path, 'wb') as fp:
                pickle.dump(centers, fp)

    print("Calculating distances...")
    Xtrain = cdist(Xtrain, centers, "cosine")
    Xtest = cdist(Xtest, centers, "cosine")
    Xtrain, Xtest = 1 - Xtrain, 1 - Xtest


    print("Sorting...")
    if args.top_centers is not None:
        furthest = (-Xtrain).argsort(axis=1)[:, args.top_centers:]
        rows = np.array([[i] * furthest.shape[1] for i in range(Xtrain.shape[0])]).ravel()
        cols = furthest.ravel()
        if len(rows) > 0:
            Xtrain[rows, cols] = 0

        furthest = (-Xtest).argsort(axis=1)[:, args.top_centers:]
        rows = np.array([[i] * furthest.shape[1] for i in range(Xtest.shape[0])]).ravel()
        cols = furthest.ravel()
        if len(rows) > 0:
            Xtest[rows, cols] = 0

    print("Training classifier...")
    cf = LogisticRegression()
    cf.fit(Xtrain, ytrain)
    train_accuracy = cf.score(Xtrain, ytrain)
    test_accuracy = cf.score(Xtest, ytest)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    if args.logfile is not None:
        with open(args.logfile, 'a') as fp:
            print("\t".join(map(str, [args.num_centers, args.top_centers, not args.random, args.seed, train_accuracy, test_accuracy])), file=fp)
