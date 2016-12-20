from helpers import calculate_mse
import numpy as np
import scipy.sparse as sp


def baseline_global_mean(train, test):
    """ Baseline method: use the global mean.

    :param train: train data array of shape (num_items, num_users)
    :param test: test data array of shape (num_items, num_users)
    :return: global_mean, the average of all the ratings, the RMSE on the train and test sets
    """
    # Compute the global mean
    global_mean = train.sum() / train.nnz

    # Compute the RMSE for test and train
    tst_nz_indices = test.nonzero()
    mse_test = 1 / test.nnz * calculate_mse(test[tst_nz_indices].toarray()[0], global_mean)
    tr_nz_indices = train.nonzero()
    mse_train = 1 / train.nnz * calculate_mse(train[tr_nz_indices].toarray()[0], global_mean)
    return global_mean, np.sqrt(mse_train), np.sqrt(mse_test)


def baseline_user_mean(train, test):
    """ Baseline method: use the user means as the prediction.

    :param train: train data array of shape (num_items, num_users)
    :param test: test data array of shape (num_items, num_users)
    :return: array of user's means with shape = (num_users,). The RMSE on the train and test sets
    """
    # Compute mean for every users
    means = np.array(train.sum(axis=0) / train.getnnz(axis=0))[0]

    # Compute the RMSE for test and train
    tst_nz_idx = test.nonzero()
    mse_test = 1 / len(tst_nz_idx[1]) * calculate_mse(test[tst_nz_idx].toarray()[0], means[tst_nz_idx[1]])
    tr_nz_idx = train.nonzero()
    mse_train = 1 / len(tr_nz_idx[1]) * calculate_mse(train[tr_nz_idx].toarray()[0], means[tr_nz_idx[1]])
    return means, np.sqrt(mse_train), np.sqrt(mse_test)


def baseline_item_mean(train, test):
    """ Baseline method: use item means as the prediction.

    :param train: train data array of shape (num_items, num_users)
    :param test: test data array of shape (num_items, num_users)
    :return: array of item's means with shape = (num_items,). The RMSE on the train and test sets
    """
    # Compute mean for every users
    means = np.array(train.sum(axis=1).T / train.getnnz(axis=1))[0]

    # Compute the RMSE for test and train
    tst_nz_idx = test.nonzero()
    mse_test = 1 / len(tst_nz_idx[0]) * calculate_mse(test[tst_nz_idx].toarray()[0], means[tst_nz_idx[0]])
    tr_nz_idx = train.nonzero()
    mse_train = 1 / len(tr_nz_idx[0]) * calculate_mse(train[tr_nz_idx].toarray()[0], means[tr_nz_idx[0]])
    return means, np.sqrt(mse_train), np.sqrt(mse_test)
