#!/usr/bin/env bash

DATASET_NAME="WikiLSHTC-325K"
FILES_PREFIX="wikiLSHTC"
PARAMS="-lr 0.07 -epoch 20 -arity 16 -dims 1000"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"


