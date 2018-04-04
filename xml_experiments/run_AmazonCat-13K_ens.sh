#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"
PARAMS="-lr 0.05 -epoch 20 -arity 64 -treeType huffman -bagging 0.5 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
