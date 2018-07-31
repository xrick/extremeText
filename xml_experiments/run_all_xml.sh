#!/usr/bin/env bash

#CMD=sbatch
CMD=bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"

PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -treeType complete"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -treeType complete"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType complete"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -treeType kmeans"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType huffman"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType complete"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
#
#PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"