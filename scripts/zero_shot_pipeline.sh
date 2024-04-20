#!/bin/bash


if [ -z $1 ]; then
        echo "Missing filename"
        exit 0
elif [ -z $2 ]; then
        echo "Missing folder"
        exit 0
elif [ -z $3 ]; then
        echo "Missing model name"
        exit 0
elif [ -z $4 ]; then
        echo "Missing suffix"
        exit 0
fi


FILE_NAME=$1
FOLDER=$2
MODEL_NAME=$3
SUFFIX=$4

# MODEL_NAME="google/flan-t5-xxl"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
# SUFFIX="_flan"
# SUFFIX="_mistral"


echo ${FILE_NAME}

NOTE_PATH="${FOLDER}/${FILE_NAME}.csv"
SENTENCES_PATH="${FOLDER}/sentences_${FILE_NAME}.csv" #intermediate file containiing parse sentences
ENCODED_PATH="${FOLDER}/encoded_${FILE_NAME}${SUFFIX}.p" #encoded prompts
SCORES_PATH="${FOLDER}/two_step_scores_${FILE_NAME}${SUFFIX}.p" #intermediate output file containing binary prompt probability
OUTPUT_PATH="${FOLDER}/two_step_output_${FILE_NAME}${SUFFIX}.csv" #output file containing evidence



RES=$(sbatch --parsable --export=NOTE_PATH=${NOTE_PATH},ENCODED_PATH=${ENCODED_PATH},SENTENCES_PATH=${SENTENCES_PATH},MODEL_NAME=${MODEL_NAME} run_two_step_encode_notes.sbatch)
sbatch --parsable --dependency=afterok:$RES --export=ENCODED_PATH=${ENCODED_PATH},OUTPUT_PATH=$OUTPUT_PATH,SCORES_PATH=$SCORES_PATH,SENTENCES_PATH=${SENTENCES_PATH},MODEL_NAME=${MODEL_NAME} run_two_step_prompt_diagnosis.sbatch
