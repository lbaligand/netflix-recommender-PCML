import numpy as np
from helpers import build_index_groups
import scipy
import scipy.io
import scipy.sparse as sp

def add_removed_elements(features, valid_indices, num_elements):
    """Add removed users or items by replacing it with the mean
    """
    full_features = np.zeros((features.shape[0], num_elements)) + np.mean(features, axis=1).reshape((-1, 1))
    for i, Vi in enumerate(valid_indices):
        full_features[:, Vi] = features[:, i]
    return full_features


def unvalid_indexes(total_elements, valid_indices):
    """ Return the corresponding unvalid elements indices
    if we remove the elements of valid_indices in a data set of size total_elements"""
    return np.delete(range(total_elements), valid_indices)


def fill_added_user_features(full_item_features, full_user_features, users_idx_to_calculate, train_full, lambda_user,
                             nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix for a specific set of indices"""
    num_users = train_full.shape[1]
    K = full_item_features.shape[0]

    # Calculate usin full_item_features or item_features ?

    for user_idx in users_idx_to_calculate:
        nz_item_features_per_user = full_item_features[:, nz_user_itemindices[user_idx]]
        A = nz_item_features_per_user @ nz_item_features_per_user.T + nnz_items_per_user[
                                                                          user_idx] * lambda_user * np.eye(K)
        ratings_per_user_nz_items = train_full[nz_user_itemindices[user_idx], user_idx]
        B = nz_item_features_per_user @ ratings_per_user_nz_items
        full_user_features[:, user_idx] = np.linalg.solve(A, B)[:, 0]
    return full_user_features


def fill_added_item_features(full_item_features, full_user_features, items_idx_to_calculate, train_full, lambda_item,
                             nnz_users_per_item, nz_item_userindices):
    """update item feature matrix for a specific set of indices"""
    num_items = train_full.shape[0]
    K = full_user_features.shape[0]

    # Calculate usin full_user_features or user_features ?

    for d in items_idx_to_calculate:
        nz_user_features_per_item = full_user_features[:, nz_item_userindices[d]]
        A = nz_user_features_per_item @ nz_user_features_per_item.T + nnz_users_per_item[d] * lambda_item * np.eye(K)
        ratings_per_item_nz_users = train_full[d, nz_item_userindices[d]]
        B = nz_user_features_per_item @ ratings_per_item_nz_users.T
        full_item_features[:, d] = np.linalg.solve(A, B)[:, 0]
    return full_item_features


def constuct_full_features(predicted_user_features, predicted_item_features,
                           valid_users_idx, valid_items_idx,
                           min_num_ratings, train_full,
                           lambda_user, lambda_item):
    if min_num_ratings == 0:
        full_user_features = predicted_user_features
        full_item_features = predicted_item_features

    else:
        total_num_items, total_num_users = train_full.shape

        full_user_features = add_removed_elements(predicted_user_features, valid_users_idx, total_num_users)
        full_item_features = add_removed_elements(predicted_item_features, valid_items_idx, total_num_items)

        added_users = unvalid_indexes(total_num_users, valid_users_idx)
        added_items = unvalid_indexes(total_num_items, valid_items_idx)

        nnz_items_per_user = train_full.getnnz(axis=0)
        nnz_users_per_item = train_full.getnnz(axis=1)

        nz_user_itemindices = []
        nz_item_userindices = []
        nz_ratings, nz_row_colindices, nz_col_rowindices = build_index_groups(train_full)
        for row, colindices in nz_row_colindices:
            nz_item_userindices.append(colindices)
        for col, rowindices in nz_col_rowindices:
            nz_user_itemindices.append(rowindices)

        full_item_features = fill_added_item_features(full_item_features, full_user_features, added_items, train_full,
                                                      lambda_item, nnz_users_per_item, nz_item_userindices)
        full_user_features = fill_added_user_features(full_item_features, full_user_features, added_users, train_full,
                                                      lambda_user, nnz_items_per_user, nz_user_itemindices)

    return full_user_features, full_item_features