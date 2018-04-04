#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"
PARAMS="-lr 0.05 -epoch 15 -arity 64 -treeType huffman"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
