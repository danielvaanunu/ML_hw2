import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (206418642, 206968281)
        self.labels = []
        self.labelled_points = []
        self.hyperplanes = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """

        self.labels = np.unique(y)
        if len(self.labelled_points) != 0:
            # if we run the code again we need to make labelled points empty
            self.labelled_points = []
        i = 0
        for row in X:
            temp = [row, y[i]]
            self.labelled_points.append(temp)
            i += 1

        # hyperplane
        K = len(self.labels)
        W = [[0] * len(X[0])] * K

        flag = True
        index = 0
        good_pred_list = []

        while flag:
            observe_x = X[index]
            observe_y = y[index]

            maximum = np.inner(W[0], observe_x)
            pred_label = 0
            for label in range(1, K):
                temp = np.inner(W[label], observe_x)
                if temp > maximum:
                    maximum = temp
                    pred_label = label

            if observe_y != pred_label:
                W[observe_y] = list(np.add(W[observe_y], observe_x))
                W[pred_label] = list(np.subtract(W[pred_label], observe_x))
                good_pred_list = []
            else:
                good_pred_list.append(observe_x)

            if len(good_pred_list) == len(X):
                flag = False

            if index == (len(X) - 1):
                index = 0
            else:
                index += 1

        self.hyperplanes = W


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        predicted_labels = np.zeros(len(X), dtype=np.uint8)
        i = 0
        for row in X:
            observe_x = row
            maximum = np.inner(self.hyperplanes[0], observe_x)
            pred_label = 0
            for label in range(1, len(self.labels)):
                temp = np.inner(self.hyperplanes[label], observe_x)
                if temp > maximum:
                    maximum = temp
                    pred_label = label
            predicted_labels[i] = pred_label
            i += 1
        return predicted_labels


if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_206418642_206968281.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
