# Bagging

The first part of this project is the implementation of bagging algorithm taught by Dr. Soumya Ray in EECS 440.

## Structure
```bag.py``` is the main file to be executed to run bagging algorithm

## Usage

```shell
python3 bag.py folder_path use_full_sample learning_algo n_iter
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```learning_algo``` takes a string among "dtree", "nbayes" and "logreg", representing which learning algorithm to use.

```n_iter``` takes a positive integer, representing the number of iterations to perform bagging.