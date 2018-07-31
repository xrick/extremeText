#!/usr/bin/env bash

DATASET_NAME=$1
FILES_PREFIX=$2
PARAMS=$3
QUANTIZE_PARAMS=$4

THREADS=8

mkdir -p models

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
BIN=${SCRIPT_DIR}/../fasttext

if [ ! -e datasets4fastText ]; then
    git clone https://github.com/mwydmuch/datasets4fastText.git
    cd datasets4fastText
    git checkout with_features_values
    cd ..
fi

if [ ! -e $FILES_PREFIX ]; then
    bash datasets4fastText/xml_repo/get_${DATASET_NAME}.sh
fi

if [ ! -e ${BIN} ]; then
    cd ${SCRIPT_DIR}/..
    make -j
    cd -
fi

TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train
TEST=${FILES_PREFIX}/${FILES_PREFIX}_test

if [ ! -e $TRAIN ]; then
    TRAIN=${FILES_PREFIX}/${FILES_PREFIX}_train0
    TEST=${FILES_PREFIX}/${FILES_PREFIX}_test0
fi

mkdir -p models
MODEL="models/${FILES_PREFIX}_$(echo $PARAMS | tr ' /' '__')"

mkdir -p output
OUTPUT="output/${FILES_PREFIX}_$(echo $PARAMS | tr ' /' '__').out"
SUMMARY_OUTPUT="output/${FILES_PREFIX}.out"

if [ ! -e $SUMMARY_OUTPUT ]; then
    echo $FILES_PREFIX > $SUMMARY_OUTPUT
    echo "params,p@1,p@3,p@5,train time,p@1 test time, p@3 test time,p@5 test time,model size" >> $SUMMARY_OUTPUT
fi

# Print header
date > $OUTPUT
echo $PARAMS >> $OUTPUT
echo "" >> $OUTPUT

# Train model
if [ ! -e ${MODEL}.bin ]; then
    { time $BIN supervised -input $TRAIN -output $MODEL -loss plt $PARAMS -thread $THREADS ; } >> $OUTPUT 2>&1
fi

# Test model
{ time $BIN test ${MODEL}.bin ${TEST} 1 ; } >> $OUTPUT 2>&1
{ time $BIN test ${MODEL}.bin ${TEST} 3 ; } >> $OUTPUT 2>&1
{ time $BIN test ${MODEL}.bin ${TEST} 5 ; } >> $OUTPUT 2>&1

echo "Model: ${MODEL}.bin" >> $OUTPUT
echo "Model size: $(ls -lh ${MODEL}.bin | grep -E '[0-9\.,]+[BMG]' -o)" >> $OUTPUT

# Print to summary file
echo -n $PARAMS >> $SUMMARY_OUTPUT
echo -n "," >> $SUMMARY_OUTPUT
grep "P@1" $OUTPUT | cut -d " " -f 2 | tr "\n" "," >> $SUMMARY_OUTPUT
grep "P@3" $OUTPUT | cut -d " " -f 2 | tr "\n" "," >> $SUMMARY_OUTPUT
grep "P@5" $OUTPUT | cut -d " " -f 2 | tr "\n" "," >> $SUMMARY_OUTPUT
grep "user" $OUTPUT | cut -f 2 | tr "\n" "," >> $SUMMARY_OUTPUT
#grep "Model size" $OUTPUT | cut -d " " -f 3 | tr "\n" "," >> $SUMMARY_OUTPUT
grep "Model size" $OUTPUT | cut -d " " -f 3 >> $SUMMARY_OUTPUT

# Model quantization
#if [ ! -e ${MODEL}.ftz ]; then
#    time $BIN quantize -output $MODEL -input $TRAIN -thread $THREADS $QUANTIZE_PARAMS
#fi
#
#time $BIN test ${MODEL}.ftz ${TEST} 1
#time $BIN test ${MODEL}.ftz ${TEST} 3
#time $BIN test ${MODEL}.ftz ${TEST} 5
#
#echo "Quantized model: ${MODEL}.ftz"
#echo "Quantized model size: $(ls -lh ${MODEL}.ftz | grep -E '[0-9\.,]+[BMG]' -o)"

# Saving documents
#$BIN save-all ${MODEL}.bin ${TRAIN} train
#$BIN save-all ${MODEL}.bin ${TEST} test

# Get probabilities for labels in the file
#time $BIN get-prob ${MODEL}.bin ${TEST} ${MODEL}_test.prob 1

# Predict labels and get probabilities for labels in the file
#time $BIN predict-prob ${MODEL}.bin ${TEST} 5 0 ${MODEL}_test.pred-prob 1