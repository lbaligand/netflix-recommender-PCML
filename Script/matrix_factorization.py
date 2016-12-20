import numpy as np
import scipy.sparse as sp
from helpers import calculate_mse, init_MF, build_index_groups, compute_error


def matrix_factorization_SGD(train, test, gamma, lambda_user, lambda_item, num_features):
    """ Matrix factorization using Stochastic Gradient Descent(SGD).

    :param train: train data matrix of size (num_items, num_users)
    :param test: test data matrix of size (num_items, num_users)
    :param gamma: step size to update the gradient
    :param lambda_user: weight of the regularizer for user_features
    :param lambda_item: weight of the regularizer for item_features
    :param num_features: number of features for the factorization, also called k
    :return: matrix decomposition learned after a fixed number of epochs.
             user_features, item_features of size (num_features, num_users) and (num_features, num_items) respectively
    """
    # Define parameters
    num_epochs = 25  # number of full passes through the train set
    errors = [0]

    # Set seed
    np.random.seed(988)

    # Initialize matrices
    user_features, item_features = init_MF(train, num_features)

    # Find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # Shuffle the training rating indices
        np.random.shuffle(nz_train)

        # Decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # Calculating the gradient for user and item
            e_dn = train[d, n] - item_features[:, d].T.dot(user_features[:, n])
            grad_user = e_dn * item_features[:, d] - lambda_user * user_features[:, n]
            grad_item = (e_dn * user_features[:, n] - lambda_item * item_features[:, d])

            # Update item_features and user_features with the gradients
            item_features[:, d] = item_features[:, d] + gamma * grad_item
            user_features[:, n] = user_features[:, n] + gamma * grad_user

        # Compute the new RMSE on training set
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        # Add the computed RMSE in the array of errors
        errors.append(rmse)

    print("TEST")
    # Compute the RMSE on the test set
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

    return user_features, item_features


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """ Update user features matrix by fixing item features matrix

    :param train: train data matrix of size (num_items, num_users)
    :param item_features: matrix of size(num_features, num_items) representing the items in the matrix factorization
    :param lambda_user: weight of the regularizer for user_features
    :param nnz_items_per_user: number of non zero items per user
    :param nz_user_itemindices: non zero item indices for each user
    :return: the updated user features matrix of size (num_features, num_users)
    """
    # Initializing the non zero parameters
    item_features_nz = item_features[:, nz_user_itemindices]
    train_nz = train[nz_user_itemindices]

    # Calculate the update of the user features matrix using the analytical formula derived in the report
    I = np.eye(item_features.shape[0])
    Ai = item_features_nz @ item_features_nz.T + lambda_user * I * nnz_items_per_user
    Vi = item_features_nz @ train_nz
    updated_user_features = np.linalg.solve(Ai, Vi)

    return updated_user_features.reshape(item_features.shape[0], )


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """ Update item features matrix by fixing user features matrix

    :param train: train data matrix of size (num_items, num_users)
    :param user_features: matrix of size(num_features, num_users) representing the users in the matrix factorization
    :param lambda_item: weight of the regularizer for item_features
    :param nnz_users_per_item: number of non zero users per item
    :param nz_item_userindices: non zero user indices for each item
    :return: the updated item features matrix of size (num_features, num_items)
    """
    # Initializing the non zero parameters
    user_features_nz = user_features[:, nz_item_userindices]
    train_nz = train[:, nz_item_userindices]

    # Calculate the update of the item features matrix using the analytical formula derived in the report
    I = np.eye(user_features.shape[0])
    Ai = user_features_nz @ user_features_nz.T + lambda_item * I * nnz_users_per_item
    Vi = user_features_nz @ train_nz.T
    updated_item_features = np.linalg.solve(Ai, Vi)

    return updated_item_features.reshape(user_features.shape[0], )


def ALS(train, test, lambda_user, lambda_item, num_features):
    """ Matrix factorization using Alternating Least Squares (ALS).

    :param train: train data matrix of size (num_items, num_users)
    :param test: test data matrix of size (num_items, num_users)
    :param lambda_user: weight of the regularizer for user_features
    :param lambda_item: weight of the regularizer for item_features
    :param num_features: number of features for the factorization, also called k
    :return: user_features, item_features of size (num_features, num_users) and (num_features, num_items) respectively.
             error_table containing the RMSEs after every iteration until it converges to the stopping criterion.
             rmse_test that is -1 if there is no test set.
    """
    # Define initial parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    error_table = []

    # Set seed
    np.random.seed(988)

    # Initialize the factorization matrices
    user_features, item_features = init_MF(train, num_features)

    # Create an array of size 2 in order to keep in memory the previous and the present RMSE to know if we have reached
    # the stop_criterion
    error_list[0] = 1000

    # Calculate arguments for the update of Z and W
    nnz_items_per_user = train.getnnz(axis=0)
    nnz_users_per_item = train.getnnz(axis=1)
    nz_train_indices, nz_row_colindices, nz_col_rowindices = build_index_groups(train)

    while abs(error_list[0] - error_list[1]) > stop_criterion:

        # Fix W (item), estimate Z (user)
        for i, nz_user_itemindices in nz_col_rowindices:
            user_features[:, i] = update_user_feature(train[:, i], item_features, lambda_user, nnz_items_per_user[i],
                                                      nz_user_itemindices)

        # Fix Z, estimate W
        for j, nz_item_userindices in nz_row_colindices:
            item_features[:, j] = update_item_feature(train[j], user_features, lambda_item, nnz_users_per_item[j],
                                                      nz_item_userindices)

        # Create a list of non zero indices of the training set
        nz_row, nz_col = train.nonzero()
        nz_train_indices = list(zip(nz_row, nz_col))

        # Store the RMSE
        error_list[change] = compute_error(train, user_features, item_features, nz_train_indices)
        error_table.append(error_list[change])

        print("RMSE on train data: {}".format(error_list[change]))

        # Update the index of the array to not overwrite the previous RMSE
        if (change == 1):
            change = 0
        else:
            change = 1

    print("Converged\n")

    # Create a list of non zero indices of the test set
    nz_row_te, nz_col_te = test.nonzero()
    nz_test = list(zip(nz_row_te, nz_col_te))

    # Check if the test is non null, otherwise we set its RMSE to -1
    if len(nz_test) == 0:
        rmse_test = -1
    else:
        rmse_test = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse_test))

    return user_features, item_features, error_table, rmse_test
