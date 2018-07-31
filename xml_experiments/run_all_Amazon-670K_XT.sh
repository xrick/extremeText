#!/usr/bin/env bash

CMD=sbatch
#CMD=bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"

LR=0.2
EPOCH=40
DIM=500
L2=0.003

# XT tree types
#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType complete"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType complete -randomTree"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType huffman"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType kmeans"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# XT tree types without reg and tf-idf
#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType complete"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType complete -randomTree"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType huffman"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType kmeans"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# XT kmeans with reg + tf-idf
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -treeType kmeans"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType kmeans"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# XT huffman with reg + tf-idf
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -treeType huffman"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType huffman"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Old update version
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType huffman -oldUpdate"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType kmeans -oldUpdate"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType huffman -oldUpdate"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType kmeans -oldUpdate"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
