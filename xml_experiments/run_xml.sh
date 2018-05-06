#!/usr/bin/env bash

DATASET_NAME=$1
FILES_PREFIX=$2
PARAMS=$3

mkdir -p models

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
BIN=${SCRIPT_DIR}/../fasttext

if [ ! -e datasets4fastText ]; then
    git clone https://github.com/mwydmuch/datasets4fastText.git
fi

if [ ! -e $FILES_PREFIX ]; then
    bash datasets4fastText/xml_repo/get_${DATASET_NAME}.sh
fi

if [ ! -e ${BIN} ]; then
    cd ${SCRIPT_DIR}/..
    make
    cd -
fi

TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train
TEST=${FILES_PREFIX}/${FILES_PREFIX}_test

if [ ! -e $TRAIN ]; then
    TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train0
    TEST=${FILES_PREFIX}/${FILES_PREFIX}_test0
fi

mkdir -p models
MODEL="models/${FILES_PREFIX}_$(echo $PARAMS | tr ' ' '_')"

if [ ! -e ${MODEL}.bin ]; then
    $BIN supervised -input ${TRAIN} -output $MODEL -loss plt $PARAMS -thread 4
fi

$BIN test ${MODEL}.bin ${TEST} 1
$BIN test ${MODEL}.bin ${TEST} 3
$BIN test ${MODEL}.bin ${TEST} 5
