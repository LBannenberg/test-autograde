from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd


def hello_world():
    return "Hello!"


def get_iris(test_size=0.4):
    X, y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X, y, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    pass
