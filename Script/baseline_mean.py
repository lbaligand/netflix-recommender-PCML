from helpers import calculate_mse
import numpy as np
import scipy.sparse as sp



def baseline_global_mean(train, test):
    """baseline method: use the global mean.
    Args:
        train:
            Train data array of shape (num_items, num_users)
        test:
            Test data array of shape (num_items, num_users)
    Returns:
        global_mean
        RMSE of training data
        RMSE of test data
    """
    # Compute the global mean
    global_mean = train.sum() / train.nnz

    # Compute the RMSE
    tst_nz_indices = test.nonzero()
    mse_test = 1 / test.nnz * calculate_mse(test[tst_nz_indices].toarray()[0], global_mean)
    tr_nz_indices = train.nonzero()
    mse_train = 1 / train.nnz * calculate_mse(train[tr_nz_indices].toarray()[0], global_mean)
    return global_mean, np.sqrt(mse_train), np.sqrt(mse_test)

    # baseline_global_mean(train, test)


def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction.
    Args:
        train:
            Train data array of shape (num_items, num_users)
        test:
            Test data array of shape (num_items, num_users)
    Returns:
        means:
            Array of user's means. shape = (num_users,)
        RMSE of training data
        RMSE of test data
    """
    # Compute mean for every users
    means = np.array(train.sum(axis=0) / train.getnnz(axis=0))[0]

    # Compute the RMSE
    tst_nz_idx = test.nonzero()
    mse_test = 1 / len(tst_nz_idx[1]) * calculate_mse(test[tst_nz_idx].toarray()[0], means[tst_nz_idx[1]])
    tr_nz_idx = train.nonzero()
    mse_train = 1 / len(tr_nz_idx[1]) * calculate_mse(train[tr_nz_idx].toarray()[0], means[tr_nz_idx[1]])
    return means, np.sqrt(mse_train), np.sqrt(mse_test)

    # _, rmse_tr, rmse_te = baseline_user_mean(train, test)
    # print("RMSE on train data: {}. RMSE on test data: {}".format(rmse_tr, rmse_te))


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction.
    Args:
        train:
            Train data array of shape (num_items, num_users)
        test:
            Test data array of shape (num_items, num_users)
    Returns:
        means:
            Array of item's means. shape = (num_items,)
        RMSE of training data
        RMSE of test data
    """
    # Compute mean for every users
    means = np.array(train.sum(axis=1).T / train.getnnz(axis=1))[0]

    # Compute the RMSE
    tst_nz_idx = test.nonzero()
    mse_test = 1 / len(tst_nz_idx[0]) * calculate_mse(test[tst_nz_idx].toarray()[0], means[tst_nz_idx[0]])
    tr_nz_idx = train.nonzero()
    mse_train = 1 / len(tr_nz_idx[0]) * calculate_mse(train[tr_nz_idx].toarray()[0], means[tr_nz_idx[0]])
    return means, np.sqrt(mse_train), np.sqrt(mse_test)

    # _, rmse_tr, rmse_te = baseline_item_mean(train, test)
    # print("RMSE on train data: {}. RMSE on test data: {}".format(rmse_tr, rmse_te))
