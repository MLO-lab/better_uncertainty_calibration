import os
import pickle
import torch


# Open file with pickled variables
def unpickle_probs(file, verbose=0):
    with open(file, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)

    if verbose:
        print("y_preds_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_preds_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels

    return ((y_probs_val, y_val), (y_probs_test, y_test))
