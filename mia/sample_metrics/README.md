# Sample Metrics
This directory contains hardness metric that can be used to evaluate
example difficulty for a given dataset. The metrics implemented are:

# Design:
Example metrics are implemented as a class that inherits from the
`ExampleMetric` class. The `ExampleMetric`, which is an abstract class.

## Prediction Depth `prediction_depth.py`
Link to the paper: [Deep Learning Through the Lens of Example Difficulty](https://arxiv.org/abs/2106.09647).

> Asserting that the final prediction is effectively determined in earlier layers of a model, before the output, we
estimate the depth at which a prediction is made for a given input as follows 2
:
> 1. We construct k-NN classifier probes from the embeddings of the training set after particular layers of
the network, including the input and the final softmax. The placement of k-NN probes is described in
Appendix A.5. We use k = 30 in the k-NN probes. Appendix A.4 establishes that the k-NN accuracies
we report are insensitive to k over a wide range.
> 2. A prediction is defined to be made at a depth L = l if the k-NN classification after layer L = l − 1 is
different from the network’s final classification, but the classifications of k-NN probes after every layer
> 3. L ≥ l are all equal to the final classification of the network. Data points consistently classified by all
k-NN probes are determined to be (effectively) predicted in layer 0 (the input) 3
.
It is worth noting that the prediction depth can be calculated for all data points: both in the training
and validation splits. This leads to two notions of computational difficulty:
• The difficulty of predicting the (given) class for an input (in the training split)
• The difficulty of making a prediction for an input, unseen in advance (from the validation split)
We examine both notions of computational difficulty in this paper and use the distinction between them to
describe different forms of example difficulty in Section 
