#!/usr/bin/env bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"

# K-Means params
PARAMS="-lr 0.05 -epoch 30 -arity 16 -dim 500 -treeType kmeans -l2 0.0001 -wordsWeights"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
