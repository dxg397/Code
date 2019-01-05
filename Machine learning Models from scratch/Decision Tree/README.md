# Decision Tree Learner

This is an implementation of ID3 decision tree algorithm for EECS 440 taught by Dr. Soumya Ray.

## Structure
```dtree.py``` is the main file to be executed
```DT_Model.py``` is the model class file that builds a ID3 decision tree

## Usage

```shell
python3 dtree.py folder_path use_full_sample max_depth use_gain_ratio
```

where ```use_full_sample``` takes value in {0, 1}, with 0 meaning do a k-fold cross validation (defaulted to 5) and 1 meaning train the model on all datapoints.

```max_depth``` takes value in non-negative integers with 0 representing "grow a full tree".

```use_gain_ratio``` takes value in {0, 1}
