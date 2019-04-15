#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 500 -l2 0.001 -wordsWeights -treeType kmeans -weightsThr 1.0"
PARAMS="-loss plt -lr 0.5 -epoch 10 -arity 2 -dim 300 -l2 0.001 -wordsWeights -treeStructure trees/eurlex_tree2"
bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-loss brt -lr 0.1 -epoch 10 -arity 2 -dim 300 -l2 0.001 -wordsWeights -treeStructure trees/eurlex_tree2"
bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-loss brt -lr 0.5 -epoch 10 -arity 2 -dim 300 -l2 0.001 -wordsWeights -treeStructure trees/eurlex_tree2 -neg 10"
bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-loss brt -lr 0.5 -epoch 10 -arity 2 -dim 300 -l2 0.001 -wordsWeights -treeStructure trees/eurlex_tree2 -neg 100"
bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-loss brt -lr 0.5 -epoch 10 -arity 2 -dim 300 -l2 0.001 -wordsWeights -treeStructure trees/eurlex_tree2 -neg 5"
bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"