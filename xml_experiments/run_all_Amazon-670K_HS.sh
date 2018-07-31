#!/usr/bin/env bash

CMD=sbatch
#CMD=bash

DATASET_NAME="Amazon-670K"
FILES_PREFIX="amazon"

LR=0.2
EPOCH=40
DIM=500
L2=0.003

# Pick one tree types
#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType complete -pickOne"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType complete -randomTree -pickOne"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType huffman -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -wordsWeights -treeType kmeans -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Pick one tree types without reg and tf-idf
#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType complete -pickOne"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

#PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType complete -randomTree -pickOne"
#${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType huffman -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType kmeans -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Pick one kmeans with reg + tf-idf
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -treeType kmeans -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType kmeans -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Pick one huffman with reg + tf-idf
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -l2 ${L2} -treeType huffman -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType huffman -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Old update version
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType huffman -oldUpdate -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -wordsWeights -treeType kmeans -oldUpdate -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType huffman -oldUpdate -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"

PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -treeType kmeans -oldUpdate -pickOne"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"


# Vanilla fastText
PARAMS="-lr ${LR} -epoch ${EPOCH} -arity 2 -dim ${DIM} -loss hs"
${CMD} run_xml_log.sl $DATASET_NAME $FILES_PREFIX "$PARAMS"
