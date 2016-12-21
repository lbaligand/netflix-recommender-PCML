# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """ Plot the statistics result on raw rating data.

    :param ratings: ratings matrix of shape (num_items, num_users)
    :return: num_items_per_user of size num_user containing number of items per user and num_users_per_item of size
             num_item containing number of users per item
    """
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 2000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """ Visualize the train and test data.

    :param train: train dataset
    :param test: test dataset
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.5)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.5)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()


def plot_min_ratings(parameter_array, rmse_train_array, rmse_test_array):
    """ Plot the RMSE test and train with respect to the minimum number of ratings

    :param parameter_array: minimum number of ratings values
    :param rmse_train_array: train RMSE for each minimum number of ratings
    :param rmse_test_array: test RMSE for each minimum number of ratings
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(parameter_array, rmse_train_array, color='red')
    ax1.set_xlabel("minimum number of ratings")
    ax1.set_ylabel("RMSE train")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(parameter_array, rmse_test_array, color='red')
    ax2.set_xlabel("minimum number of ratings")
    ax2.set_ylabel("RMSE test")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("minimum_number_of_ratings")
    plt.show()


def plot_number_features(parameter_array, rmse_train_array, rmse_test_array):
    """ Plot the RMSE test and train with respect to number of features

    :param parameter_array: number of features values
    :param rmse_train_array: train RMSE for each number of features
    :param rmse_test_array: test RMSE for each number of features
    """
    fig = plt.figure()

    # Plot for RMSE train
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(parameter_array, rmse_train_array, color='red')
    ax1.set_xlabel("number of features")
    ax1.set_ylabel("RMSE train")
    ax1.grid()

    # Plot for RMSE test
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(parameter_array, rmse_test_array, color='red')
    ax2.set_xlabel("number of features")
    ax2.set_ylabel("RMSE test")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("number_of_features_errors")
    plt.show()


def plot_every_lambda(results_test_lambda, lambda_table_user, lambda_table_item):
    """ Plot the test RMSE with respect to the lambdas of item for every lambda user

    :param results_test_lambda: test RMSEs for every combinations of lambdas
    :param lambda_table_user: regularizer parameter values in ALS for the user features matrix
    :param lambda_table_item: regularizer parameter values in ALS for the item features matrix
    """
    # Initialize parameters
    width = 12
    height = 8
    fig = plt.figure(figsize=(width, height))
    ax1 = fig.add_subplot(1, 1, 1)
    x = np.array(["b", "g", "r", "c", "m", "y", "k", "pink", "orange", "indigo", "brown", "gray", "lightblue"])
    lambda_new_axis = np.linspace(lambda_table_item.min(), lambda_table_item.max(), 100)

    # For a fixed lambda user, plot the test RMSE with respect to the lambdas of item
    for i, row in enumerate(results_test_lambda):
        s = np.poly1d(np.polyfit(lambda_table_item, row, 7))
        results_smooth = s(lambda_new_axis)
        ax1.plot(lambda_new_axis, results_smooth, color=x[i], label="lambda user = {}".format(lambda_table_user[i]))

    ax1.set_xlabel("lambda item")
    ax1.set_ylabel("rmse test")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    ax1.grid()
    ax1.set_title("Test sample")
    plt.savefig("estimation_lambda_complete")
