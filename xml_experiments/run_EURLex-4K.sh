#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"
PARAMS="-lr 0.5 -epoch 20 -arity 32 -dim 300 -treeType huffman"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
