#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"

# Complete tree params
PARAMS="-lr 0.5 -epoch 20 -arity 32 -dim 250 -l2 0.001 -wordsWeights"

# K-Means params
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans"

# K-Means with pick one heuristic
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans -pickOne"

# K-Means with negative-sampling
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans -neg 5"

bash run_xml_log.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
