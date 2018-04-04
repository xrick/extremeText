#!/usr/bin/env bash

DATASET_NAME="RCV1-2K"
FILES_PREFIX="rcv1x"
PARAMS="-lr 0.05 -epoch 30 -arity 16 -dims 300 -treeType huffman -bagging 0.5 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
