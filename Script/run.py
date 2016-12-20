import numpy as np
from helpers import split_data, create_submission_csv, load_data
from matrix_factorization import ALS
from plots import plot_raw_data
from postprocessing import construct_full_features

# Load the provided Data Set in the Data folder
path_dataset = "../Data/data_train.csv"
ratings = load_data(path_dataset).T
print("Number of items: {}, Number of users: {}.".format(ratings.shape[0], ratings.shape[1]))

# Plot the statistics result on raw rating data
num_items_per_user, num_users_per_item = plot_raw_data(ratings)

# Split the data to have a train and a test set
min_num_ratings = 20
valid_ratings, train, test, valid_users, valid_items, train_full, test_full = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test=0)

# Initialize parameters for the first ALS with k = 10
lambda_user = 0.15
lambda_item = 0.04
k = 10
predicted_user_features_k10, predicted_item_features_k10, error_table, _ = ALS(train, test, lambda_user, lambda_item,
                                                                               k)

# Initialize parameters for the second ALS
lambda_user = 0.8
lambda_item = 0.01
k = 20
predicted_user_features_k20, predicted_item_features_k20, error_table, _ = ALS(train, test, lambda_user,
                                                                               lambda_item, k)

# Reconstruct the full features matrices in order to have the matching prediction size
full_user_features_k10, full_item_features_k10 = construct_full_features(predicted_user_features_k10,
                                                                        predicted_item_features_k10,
                                                                        valid_users, valid_items, min_num_ratings,
                                                                        train_full,
                                                                        lambda_user, lambda_item)
full_user_features_k20, full_item_features_k20 = construct_full_features(predicted_user_features_k20,
                                                                        predicted_item_features_k20,
                                                                        valid_users, valid_items,
                                                                        min_num_ratings, train_full,
                                                                        lambda_user, lambda_item)

# Compute the prediction using the weighted predictions found in the previous two ALS
weight = 0.46
full_prediction_from_two = np.multiply(full_item_features_k20.T @ full_user_features_k20, weight) \
                           + np.multiply(full_item_features_k10.T @ full_user_features_k10, 1 - weight)

# Create the submission file in the current folder
create_submission_csv(full_prediction_from_two, "../Data/sampleSubmission.csv", "./submission.csv")
