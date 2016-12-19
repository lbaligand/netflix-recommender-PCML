# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:] # First line is ID,prediction
    return preprocess_data(data)

def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings



def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""

    # initalization
    prediction = (item_features.T).dot(user_features)
    x, y = zip(*nz)

    # remove zero elements
    prediction_nz = prediction[x, y]
    data_nz = data[x, y]

    # rmse
    return np.sqrt(calculate_mse(data_nz, prediction_nz).sum() / (data.nnz))


from scipy.sparse import lil_matrix


def normalize(ratings):
    # array of train and test
    alg_ratings = ratings.todense()

    # index with zeros
    I_ratings = np.zeros((ratings.shape[0], ratings.shape[1]))
    I_ratings[alg_ratings > 0] = 1

    # mean and std dev column-wise
    mean_ratings_col = np.sum(alg_ratings, axis=0) / np.diff(ratings.tocsc().indptr)
    std_dev_col = np.std(alg_ratings, axis=0)

    # remove nan
    mean_ratings_col = np.nan_to_num(mean_ratings_col)

    # normalizing
    alg_ratings_norm = (alg_ratings - mean_ratings_col) / std_dev_col

    # normalizing and discarding previous zero values
    normalized_ratings = lil_matrix(np.multiply((alg_ratings_norm), I_ratings))

    return normalized_ratings, mean_ratings_col, std_dev_col


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        ratings:
            The given loaded data that corresponds to the ratings of shape (num_items, num_users)
        num_users_per_item:
            Number of users corresponding to every items. shape = (num_items,)
        num_items_per_user:
            Number of items corresponding to every users. shape = (num_users,)
        p_test:
            Probability that one rating is in the test data
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    Returns:
        valid_ratings:
            Ratings that corresponds to the conditions
        train:
            The splitted train data
        test:
            The splitted test data
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # initialization
    train = valid_ratings.copy()
    test = valid_ratings.copy()
    train_full = ratings.copy()
    test_full = ratings.copy()

    # split the data and return train and test data: for every valid user,
    # create a binary mask with probability p_test and set the non zero value
    # of the test sparse matrix to zero.
    for u in range(valid_ratings.shape[1]):
        non_zero_ratings = valid_ratings[:, u].nonzero()
        mask_test = np.random.choice(2, non_zero_ratings[0].shape, p=[1 - p_test, p_test]).astype(bool)
        test_idxs = non_zero_ratings[0][mask_test]
        train_idxs = non_zero_ratings[0][~mask_test]
        train[test_idxs, u] = 0
        test[train_idxs, u] = 0

    # split the data and return train and test data before threshold.
    for u in range(ratings.shape[1]):
        non_zero_ratings = ratings[:, u].nonzero()
        mask_test = np.random.choice(2, non_zero_ratings[0].shape, p=[1 - p_test, p_test]).astype(bool)
        test_idxs = non_zero_ratings[0][mask_test]
        train_idxs = non_zero_ratings[0][~mask_test]
        train_full[test_idxs, u] = 0
        test_full[train_idxs, u] = 0

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test, valid_users, valid_items, train_full, test_full

def init_MF(train, num_features):  # CHRIS' Function
    """init the parameter for matrix factorization."""

    # initalization
    num_items, num_users = train.shape
    item_features = np.random.randn(num_features, num_items)
    user_features = np.random.randn(num_features, num_users)

    # mean of for each item
    sums_train = train.sum(axis=1).reshape(num_items, )
    counts_train = np.diff(train.tocsr().indptr)  # counts number of non zero value for each row
    mean_train_item = sums_train / counts_train

    item_features[0, :] = mean_train_item

    return user_features, item_features

import csv

def create_submission_csv(predictions,sample_submission_filename, submission_filename):
    sample_data = read_txt(sample_submission_filename)[1:]
    sample_data = [deal_line(line) for line in sample_data]
    with open(submission_filename, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for user,item,fake_rating in sample_data:
            writer.writerow({'Id': "r{}_c{}".format(user,item), 'Prediction': predictions[item-1,user-1]})
        #WARNING NEW LINE IN OUPTU FILE
