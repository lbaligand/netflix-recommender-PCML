import numpy as np
from helpers import split_data,create_submission_csv,load_data
from plots import plot_raw_data
from matrix_factorization import ALS
from postprocessing import constuct_full_features

path_dataset = "../Data/data_train.csv"
ratings = load_data(path_dataset).T
print("Number of items: {}, Number of users: {}.".format(
        ratings.shape[0],ratings.shape[1]))
num_items_per_user, num_users_per_item = plot_raw_data(ratings)

min_num_ratings = 20
valid_ratings, train, test, valid_users, valid_items, train_full, test_full = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test = 0)

lambda_user = 0.15
lambda_item = 0.04
k = 10
predicted_user_features_kten, predicted_item_features_kten, error_table, rmse_test = ALS(train, test, lambda_user, lambda_item, k)
lambda_user = 0.8
lambda_item = 0.01
k = 20
predicted_user_features_ktwenty, predicted_item_features_ktwenty, error_table, rmse_test = ALS(train, test, lambda_user, lambda_item, k)

full_user_features_ktwenty, full_item_features_ktwenty = constuct_full_features(predicted_user_features_ktwenty, predicted_item_features_ktwenty,
                                                                valid_users, valid_items, min_num_ratings, train_full,
                                                                lambda_user, lambda_item)
full_user_features_kten, full_item_features_kten = constuct_full_features(predicted_user_features_kten, predicted_item_features_kten,
                                                                valid_users, valid_items, min_num_ratings, train_full,
                                                                lambda_user, lambda_item)

value = 0.46
full_prediction_from_two = np.multiply(full_item_features_ktwenty.T @ full_user_features_ktwenty, value) + np.multiply(full_item_features_kten.T @ full_user_features_kten, 1-value)

create_submission_csv(full_prediction_from_two,"../Data/sampleSubmission.csv","./submission.csv")
