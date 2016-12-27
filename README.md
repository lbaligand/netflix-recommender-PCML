# ML prject 2 Recommended systems
Adrian Pace
Christophe Windler
Louis Baligand

Please make sure that the train data is in the /Data folder

The prediction can be optained by running the script/run.py. It will create the submission file script/submission.csv.

The folder data contains the sample_submission and the data_train csv files.

The folder script contains our implementation:

baseline_mean.py contains the methods calculating the baseline means
helpers.py contains the functions provided in lab10 and some custom functions such as normalization, data split, matrix initialization and submission file creation
matrix_factorization.py contains the ALS and SGD implementations
parameter_search.py contains the methods that we use to estimate the different paramenters (lambdas, gamma, number of features)
plots.py contains the methods that constructs our graphs
postprocessing.py contains the method to add the removed elements (that had too few ratings) back to the features matrix
script.ipynb is the notebook containing the code used for our parameter search.
run.py contains the script for calculating our prediction
