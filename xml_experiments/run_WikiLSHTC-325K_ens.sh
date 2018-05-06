#!/usr/bin/env bash

DATASET_NAME="WikiLSHTC-325K"
FILES_PREFIX="wikiLSHTC"
PARAMS="-lr 0.07 -epoch 20 -arity 16 -dim 1000 -randomTree -bagging 1.0 -nbase 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"


