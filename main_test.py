import main
import pandas as pd


def test_hello_world():
    assert main.hello_world() == 'Hello World!'


def test_hello_world_again():
    assert main.hello_world() == 'Hello World!'


def test_iris_split():
    X, y, X_train, X_test, y_train, y_test = main.get_iris(test_size=0.4)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert X_train.shape == (90, 4)
    assert X_test.shape == (60, 4)
    assert y_train.shape == (90,)
    assert y_test.shape == (60,)
    X, y, X_train, X_test, y_train, y_test = main.get_iris(test_size=0.6)
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert X_train.shape == (60, 4)
    assert X_test.shape == (90, 4)
    assert y_train.shape == (60,)
    assert y_test.shape == (90,)
