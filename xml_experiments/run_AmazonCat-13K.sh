#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"

# Huffman tree params
PARAMS="-lr 0.05 -epoch 15 -arity 64 -dim 250 -treeType huffman -l2 0.0001 -wordsWeights"

# Complete tree params
PARAMS="-lr 0.2 -epoch 30 -arity 16 -dim 500 -l2 0.0001 -wordsWeights"

# K-Means params
PARAMS="-lr 0.2 -epoch 15 -arity 2 -dim 500 -l2 0.001 -wordsWeights -treeType kmeans"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
