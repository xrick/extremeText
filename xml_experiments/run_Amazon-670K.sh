#!/usr/bin/env bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"
PARAMS="-lr 0.05 -epoch 30 -arity 16 -dim 300 -treeType huffman"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
