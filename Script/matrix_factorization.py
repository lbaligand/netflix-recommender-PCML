import numpy as np
import scipy.sparse as sp
from helpers import calculate_mse,init_MF,build_index_groups,compute_error


def matrix_factorization_SGD(train, test,gamma,lambda_user,lambda_item,num_features):
    """matrix factorization by SGD."""
    # define parameters

    num_epochs = 25  # number of full passes through the train set
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            e_dn = train[d, n] - (item_features[:, d].T).dot(user_features[:, n])
            grad_user = e_dn * item_features[:, d] - lambda_user * user_features[:, n]
            grad_item = (e_dn * user_features[:, n] - lambda_item * item_features[:, d])

            item_features[:, d] = item_features[:, d] + gamma * grad_item
            user_features[:, n] = user_features[:, n] + gamma * grad_user

        regularized_term = lambda_user / 2 * np.linalg.norm(user_features) + lambda_item / 2 * np.linalg.norm(
            item_features)
        rmse = compute_error(train, user_features, item_features, nz_train)

        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)

    print("TEST")

    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

    return user_features, item_features

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""

    I = np.eye(item_features.shape[0])
    item_features_nz = item_features[:, nz_user_itemindices]
    train_nz = train[nz_user_itemindices]

    Ai = item_features_nz @ item_features_nz.T + lambda_user * I * nnz_items_per_user
    Vi = item_features_nz @ train_nz

    updated_user_features = np.linalg.solve(Ai, Vi)

    return updated_user_features.reshape(item_features.shape[0], )


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""

    I = np.eye(user_features.shape[0])
    user_features_nz = user_features[:, nz_item_userindices]
    train_nz = train[:, nz_item_userindices]

    Ai = user_features_nz @ user_features_nz.T + lambda_item * I * nnz_users_per_item
    Vi = user_features_nz @ train_nz.T

    updated_item_features = np.linalg.solve(Ai, Vi)

    return updated_item_features.reshape(user_features.shape[0], )





def ALS(train, test, lambda_user, lambda_item, k):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = k  # K in the lecture notes
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    error_table = []
    rmse = 0

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    error_list[0] = 1000

    # Calculate arguments for the update of Z and W
    nnz_items_per_user = train.getnnz(axis=0)
    nnz_users_per_item = train.getnnz(axis=1)
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)

    while (abs(error_list[0] - error_list[1]) > stop_criterion):

        # Fix W (item), estimate Z (user)
        for i, nz_user_itemindices in nz_col_rowindices:
            user_features[:, i] = update_user_feature(train[:, i], item_features, lambda_user, nnz_items_per_user[i],
                                                      nz_user_itemindices)

        # Fix Z, estimate W
        for j, nz_item_userindices in nz_row_colindices:
            item_features[:, j] = update_item_feature(train[j], user_features, lambda_item, nnz_users_per_item[j],
                                                      nz_item_userindices)

        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))
        error_list[change] = compute_error(train, user_features, item_features, nz_train)
        error_table.append(error_list[change])

        print("RMSE on train data: {}".format(error_list[change]))

        if (change == 1):
            change = 0
        else:
            change = 1

    print("Converged")
    print()

    nz_row_te, nz_col_te = test.nonzero()
    nz_test = list(zip(nz_row_te, nz_col_te))
    if len(nz_test)==0:
        rmse_test=-1
    else:
        rmse_test = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse_test))

    return user_features, item_features, error_table, rmse_test