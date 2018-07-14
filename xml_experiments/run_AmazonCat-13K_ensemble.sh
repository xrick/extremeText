#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"

# Huffman tree params
PARAMS="-lr 0.05 -epoch 20 -arity 64 -dim 250 -treeType huffman -bagging 0.5 -ensemble 3 -wordsWeights"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
