# LRCN_model for Human Activity Classification
A Video Classification model used for Human activity recognition that uses the CNN-RNN architecture mentioned in LRCN paper. This model was trained on a subset of Epic Kitchen dataset.

link to paper - https://arxiv.org/pdf/1411.4389.pdf

This model was used for binary classification, number of classes will change for multi-class classification. Also, due to computational limitations, Videos were split into smaller videos of length 10 seconds. This can be changed accoding to the computational power available.

Note - The path for video dataset must be relative to the location of this script, else frame extraction will not execute.
