#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"
PARAMS="-lr 0.5 -epoch 20 -arity 32 -dim 300 -randomTree -bagging 1.0 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
