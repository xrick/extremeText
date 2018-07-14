#!/usr/bin/env bash

DATASET_NAME="Wiki10-31K"
FILES_PREFIX="wiki10"

# Complete tree params
PARAMS="-lr 0.2 -epoch 30 -arity 16 -dim 500 -l2 0.0001 -wordsWeights -neg 5"

# K-Means params
PARAMS="-lr 0.2 -epoch 30 -arity 2 -dim 500 -l2 0.0001 -wordsWeights -treeType kmeans -neg 5"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
