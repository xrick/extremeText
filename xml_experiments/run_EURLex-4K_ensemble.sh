#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"

# K-Means params
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans -neg 5 -ensemble 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
