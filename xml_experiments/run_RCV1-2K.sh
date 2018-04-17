#!/usr/bin/env bash

DATASET_NAME="RCV1-2K"
FILES_PREFIX="rcv1x"
PARAMS="-lr 0.05 -epoch 20 -arity 16 -dim 300 -treeType huffman"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
