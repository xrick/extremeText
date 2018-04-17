#!/usr/bin/env bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"
PARAMS="-lr 0.05 -epoch 40 -arity 16 -dim 300 -treeType huffman -bagging 0.5 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
