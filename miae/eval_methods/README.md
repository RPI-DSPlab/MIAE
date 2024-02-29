# `mia/eval_methods` Directory

This directory contains the evaluation methods used in the Membership Inference Attack (MIA) project. The main purpose of these methods is to evaluate the performance of the implemented attacks.

## Files

- `sample_metric_eval.py`: This file contains the `SampleMetric` class which is used to evaluate the performance of the attacks on a per-sample basis. It provides methods to read metrics from a file, print the length of the scores, plot the score distribution for training and testing data, and plot the score distribution of two different `SampleMetric` instances.

    To use the `SampleMetric` class, you need to provide the training and testing score dictionaries, as well as the training and testing data loaders during initialization. The score dictionaries should be in the format `{sample_index: score}`.


