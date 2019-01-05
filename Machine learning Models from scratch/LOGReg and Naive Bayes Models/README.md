# Naive Bayes Net

The first part of this project is the implementation of naive Bayes algorithm taught by Dr. Soumya Ray in EECS 440.

## Structure
```nbayes.py``` is the main file to be executed to run naive Bayes algorithm

## Usage

```shell
python3 nbayes.py folder_path use_full_sample max_depth use_gain_ratio
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```n_bins``` takes value in non-negative integers greater than 2.

```m_estimate``` takes value in {0, 1}.

# Logistic Regression

The second part of this project is the implementation of logistic regression algorithm taught by Dr. Soumya Ray in EECS 440.

## Structure
```logreg.py``` is the main file to be executed to run logistic regression.

## Usage

```shell
python3 logreg.py folder_path use_full_sample lambda
```

```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```lambda``` is a non-negative real number that sets value for required constant lambda.