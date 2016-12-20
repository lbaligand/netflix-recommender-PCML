import numpy as np
from helpers import build_index_groups
import scipy
import scipy.io
import scipy.sparse as sp


def add_removed_elements(features, valid_indices, total_elements):
    """ Add removed users or items by replacing it with the mean

    :param features: The feature matrix that has to be refill
    :param valid_indices: indices of items/users that have more than the minimum number of ratings
    :param total_elements: number total of elements, i.e. either num_users or num_items
    :return: the full features matrix of shape (num_features, num_users/items)
    """
    # Initialize with the mean of its row
    full_features = np.zeros((features.shape[0], total_elements)) + np.mean(features, axis=1).reshape((-1, 1))

    # Complete the initialized matrix with only the valid columns
    for i, valid_col in enumerate(valid_indices):
        full_features[:, valid_col] = features[:, i]
    return full_features


def unvalid_indexes(total_elements, valid_indices):
    """ Return the corresponding unvalid elements indices by removing the elements of valid_indices in a data set of
    size total_elements

    :param total_elements: number total of elements, i.e. either num_users or num_items
    :param valid_indices: indices of items/users that have more than the minimum number of ratings
    :return: The corresponding unvalid elements indices
    """
    return np.delete(range(total_elements), valid_indices)


def fill_added_user_features(full_item_features, full_user_features, users_idx_to_calculate, train_full, lambda_user,
                             nnz_items_per_user, nz_user_itemindices):
    """ Update user features matrix for a specific set of indices

    :param full_item_features: item features matrix from the matrix factorization, of shape (num_features, num_items)
    :param full_user_features: user features matrix from the matrix factorization, of shape (num_features, num_users)
    :param users_idx_to_calculate: the specific set of indices of users to calculate
    :param train_full: full train data set of size (num_items, num_users)
    :param lambda_user: the weight of the regularizer for the user features matrix
    :param nnz_items_per_user: number of non zero items per user
    :param nz_user_itemindices: non zero user indices for each item
    :return: the full user features matrix used for to compute the prediction. shape (num_features, num_items)
    """
    # Number of features
    K = full_item_features.shape[0]

    for user_idx in users_idx_to_calculate:
        # Find the non zero columns of the item features matrix for a specific user
        nz_item_features_per_user = full_item_features[:, nz_user_itemindices[user_idx]]

        # Calculate the update of the user features matrix using the analytical formula derived in the report
        A = nz_item_features_per_user @ nz_item_features_per_user.T + nnz_items_per_user[
                                                                          user_idx] * lambda_user * np.eye(K)

        # Find the ratings of non zero items for a specific user
        ratings_per_user_nz_items = train_full[nz_user_itemindices[user_idx], user_idx]
        B = nz_item_features_per_user @ ratings_per_user_nz_items
        full_user_features[:, user_idx] = np.linalg.solve(A, B)[:, 0]
    return full_user_features


def fill_added_item_features(full_item_features, full_user_features, items_idx_to_calculate, train_full, lambda_item,
                             nnz_users_per_item, nz_item_userindices):
    """ Update item features matrix for a specific set of indices

    :param full_item_features: item features matrix from the matrix factorization, of shape (num_features, num_items)
    :param full_user_features: user features matrix from the matrix factorization, of shape (num_features, num_users)
    :param items_idx_to_calculate: the specific set of indices of items to calculate
    :param train_full: full train data set of size (num_items, num_users)
    :param lambda_item: the weight of the regularizer for the item features matrix
    :param nnz_users_per_item: number of non zero users per item
    :param nz_item_userindices: non zero item indices for each user
    :return:
    """
    # Number of features
    K = full_user_features.shape[0]

    for d in items_idx_to_calculate:
        # Find the non zero columns of the user features matrix for a specific item
        nz_user_features_per_item = full_user_features[:, nz_item_userindices[d]]

        # Calculate the update of the item features matrix using the analytical formula derived in the report
        A = nz_user_features_per_item @ nz_user_features_per_item.T + nnz_users_per_item[d] * lambda_item * np.eye(K)

        # Find the ratings of non zero users for a specific item
        ratings_per_item_nz_users = train_full[d, nz_item_userindices[d]]
        B = nz_user_features_per_item @ ratings_per_item_nz_users.T
        full_item_features[:, d] = np.linalg.solve(A, B)[:, 0]
    return full_item_features


def construct_full_features(predicted_user_features, predicted_item_features,
                            valid_users_idx, valid_items_idx,
                            min_num_ratings, train_full,
                            lambda_user, lambda_item):
    """ Construct the full user and item features matrix to match the size in the prediction

    :param predicted_user_features: predicted user features matrix that has to be filled
    :param predicted_item_features: predicted item features matrix that has to be filled
    :param valid_users_idx: indices of valid users
    :param valid_items_idx: indices of valid items
    :param min_num_ratings: minimum number of ratings
    :param train_full: full training data set
    :param lambda_user: weight of the regularizer for user_features
    :param lambda_item: weight of the regularizer for item_features
    :return: the full user and item features matrix of shapes (num_features, num_users) and (num_features, num_items)
    """
    # Check for the base case
    if min_num_ratings == 0:
        full_user_features = predicted_user_features
        full_item_features = predicted_item_features

    else:
        total_num_items, total_num_users = train_full.shape

        # Add columns for the deleted user and items
        full_user_features = add_removed_elements(predicted_user_features, valid_users_idx, total_num_users)
        full_item_features = add_removed_elements(predicted_item_features, valid_items_idx, total_num_items)

        # Select the unvalid indexes
        added_users = unvalid_indexes(total_num_users, valid_users_idx)
        added_items = unvalid_indexes(total_num_items, valid_items_idx)

        # Find the number of non zero for items and users
        nnz_items_per_user = train_full.getnnz(axis=0)
        nnz_users_per_item = train_full.getnnz(axis=1)

        # Create the non zero item indices for each user and the non zero user indices for each item
        nz_user_itemindices = []
        nz_item_userindices = []
        nz_ratings, nz_row_colindices, nz_col_rowindices = build_index_groups(train_full)
        for row, colindices in nz_row_colindices:
            nz_item_userindices.append(colindices)
        for col, rowindices in nz_col_rowindices:
            nz_user_itemindices.append(rowindices)

        # Update the features matrices in order to converge to a better prediction than the everage
        full_item_features = fill_added_item_features(full_item_features, full_user_features, added_items, train_full,
                                                      lambda_item, nnz_users_per_item, nz_item_userindices)
        full_user_features = fill_added_user_features(full_item_features, full_user_features, added_users, train_full,
                                                      lambda_user, nnz_items_per_user, nz_user_itemindices)

    return full_user_features, full_item_features
