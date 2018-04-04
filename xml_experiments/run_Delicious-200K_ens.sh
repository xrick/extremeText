#!/usr/bin/env bash

DATASET_NAME="Delicious-200K"
FILES_PREFIX="deliciousLarge"
PARAMS="-lr 0.05 -epoch 40 -arity 4 -dim 1000 -randomTree -bagging 1.0 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
