"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
from regression import utils, logreg
import pytest
import numpy as np


@pytest.fixture
def nsclc_data():
    return utils.loadDataset(split_percent=0.7)


@pytest.fixture
def gradient_test():
    return np.load("./test/gradient.npy")


@pytest.fixture
def loss_test():
    return np.load("./test/loss.npy")


def test_updates(nsclc_data, gradient_test, loss_test):
    """
    Test whether the losses and gradient match previous manually computed values.
    Also make sure that the loss is going down generally over a large number of iterations; previously this number produced a reasonable fit accuracy of ~70-80%
    """

    np.random.seed(0)

    X_train, X_test, y_train, y_test = nsclc_data

    regressor = logreg.LogisticRegression(
        X_train.shape[1], max_iter=10000, learning_rate=0.0001, batch_size=200, tol=1e-6
    )

    X_train_intercept = np.c_[X_train, np.ones(len(X_train))]
    gradient = regressor.calculate_gradient(X_train_intercept, y_train)

    loss = regressor.loss_function(X_train_intercept, y_train)

    assert np.allclose(
        gradient, gradient_test  # gradient test previously manually computed
    ), "Gradient values do not match manually calculated values"

    assert np.allclose(
        loss, loss_test
    ), "Loss values do not match calculated values"  # loss test previously manually computed

    regressor.train_model(X_train, y_train, X_test, y_test)  #

    assert np.mean(regressor.loss_history_val[:1000]) > np.mean(
        regressor.loss_history_val[-1000:]
    ), "Loss did not go down over the run"


def test_predict(nsclc_data):
    """
    Test whether the weights change after being fit and whether the accuracy goes up on average over the run.
    """

    X_train, X_test, y_train, y_test = nsclc_data
    regressor = logreg.LogisticRegression(
        X_train.shape[1], max_iter=10000, learning_rate=0.001, batch_size=200, tol=1e-6
    )

    w = regressor.W
    regressor.train_model(X_train, y_train, X_test, y_test)
    trained_w = regressor.W

    assert not np.allclose(
        w, trained_w
    ), "W should have changed over the run"  # weights not same

    preds = regressor.make_prediction(
        np.c_[X_test, np.ones(len(X_test))]
    )  # put on intercept

    preds = np.rint(preds)  # round probs to 0, 1

    correct = (
        np.bitwise_and(preds == 1, y_test == 1).sum()
        + np.bitwise_and(preds == 0, y_test == 0).sum()
    )
    
    accuracy = correct / len(X_test)
    assert (
        accuracy > 0.5
    ), "Accuracy should have gone up over the run"  # generally just see how many predictions were in the true category
