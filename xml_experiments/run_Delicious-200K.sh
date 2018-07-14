#!/usr/bin/env bash

DATASET_NAME="Delicious-200K"
FILES_PREFIX="deliciousLarge"

# Complete tree params
PARAMS="-lr 0.05 -epoch 40 -arity 4 -dim 500 -l2 0.0001 -wordsWeights"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
