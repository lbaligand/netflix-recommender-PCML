# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data.
    
    Args:
        ratings: The ratings matrix of shape (num_item, num_user)
    
    Return:
        num_items_per_user: Array of size num_user containing number of items
                            per user.
        num_users_per_item: Array of size num_item containing number of users
                            per item.             
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
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
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


def plot_parameter(parameter_array, rmse_train_array, rmse_test_array):
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(parameter_array, rmse_train_array, color='red')
    ax1.set_xlabel("number of features")
    ax1.set_ylabel("RMSE train")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(parameter_array, rmse_test_array, color='red')
    ax2.set_xlabel("number of features")
    ax2.set_ylabel("RMSE test")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("number_of_features_errors")
    plt.show()


def plot_every_lambda(results_test_lambda, lambda_table_user, lambda_table_item):
    width = 12
    height = 8
    fig = plt.figure(figsize=(width, height))
    ax1 = fig.add_subplot(1, 1, 1)
    x = np.array(["b", "g", "r", "c", "m", "y", "k", "pink", "orange", "indigo", "brown", "gray", "lightblue"])
    lambda_new_axis = np.linspace(lambda_table_item.min(), lambda_table_item.max(), 100)

    for i, row in enumerate(results_test_lambda):
        s = np.poly1d(np.polyfit(lambda_table_item, row, 7))
        results_smooth = s(lambda_new_axis)
        # results_smooth = itp.spline(lambda_table_item, row, lambda_new_axis)
        ax1.plot(lambda_new_axis, results_smooth, color=x[i], label="lambda user = {}".format(lambda_table_user[i]))
        # ax1.plot(lambda_table_item, row, color = x[i], label="lambda user = {}".format(lambda_table_user[i]))

    ax1.set_xlabel("lambda item")
    ax1.set_ylabel("rmse test")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    ax1.grid()
    ax1.set_title("Test sample")
    plt.savefig("estimation_lambda_complete")


