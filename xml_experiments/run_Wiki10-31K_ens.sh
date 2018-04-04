#!/usr/bin/env bash

DATASET_NAME="Wiki10-31K"
FILES_PREFIX="wiki10"
PARAMS="-lr 0.2 -epoch 30 -arity 16 -dims 300 -random_tree -bagging 1.0 -nBase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
