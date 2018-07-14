#!/usr/bin/env bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"

# Huffman tree params
PARAMS="-lr 0.05 -epoch 40 -arity 16 -dim 300 -treeType huffman -bagging 0.5 -ensemble 3 -wordsWeights"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
