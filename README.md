This repository includes basic implementations of k-Nearest Neighbors (KNN) and Gaussian Naive Bayes in Python.

The KNN implementation provided can handle both weighted and normal (unweighted) versions, where the current weighted version is distance based.

## Requirements
- Python 3

The implementation does not require any external libraries.

## Usage
You can find an example of how to use the models in the `main.py` file. You can modify the following parameters to fit your use case:
`runs`, `split_ratio`, `file_name`. 

You will also need to the change the implementation of the `transform(line)` function based on your data file. This function is used to convert each row of the data file into a 
feature vector. A more detailed explanation about the function can be found in the docstring of the function itself.

## TODO
Improve the implementation by allowing users to pass their own weighting function when using KNN.