#!/bin/bash


if [ -z $1 ]; then
        echo "Missing filename"
        exit 0
elif [ -z $2 ]; then
        echo "Missing folder"
        exit 0
elif [ -z $3 ]; then
        echo "Missing cache path"
        exit 0
elif [ -z $4 ]; then
        echo "Missing GPT risk factors embeddings path. Run run_embed_risk_factors.sbatch"
        exit 0
fi

FILE_NAME=$1
FOLDER=$2
CACHE_PATH=$3
EMBEDDINGS_PATH=$4 #need to run run_embed_risk_factors.sbatch for this first!

echo ${FILE_NAME}

NOTE_PATH="${FOLDER}/${FILE_NAME}.csv"
ENCODED_PATH="${FOLDER}/encoded_${FILE_NAME}.p"
SENTENCES_PATH="${FOLDER}/sentences_${FILE_NAME}.csv"
OUTPUT_PATH="${FOLDER}/cbert_scores_${FILE_NAME}.csv"
MODEL_PATH="emilyalsentzer/Bio_ClinicalBERT"

 

RES=$(sbatch --parsable --export=NOTE_PATH=${NOTE_PATH},ENCODED_PATH=${ENCODED_PATH},MODEL_PATH=${MODEL_PATH},SENTENCES_PATH=${SENTENCES_PATH},CACHE_PATH=$CACHE_PATH run_retrieval_encode_notes.sbatch)
sbatch --parsable --dependency=afterok:$RES --export=ENCODED_PATH=${ENCODED_PATH},OUTPUT_PATH=${OUTPUT_PATH},MODEL_PATH=${MODEL_PATH},CACHE_PATH=${CACHE_PATH},EMBEDDINGS_PATH=${EMBEDDINGS_PATH},SENTENCES_PATH=${SENTENCES_PATH} run_retrieval_cosine_rank.sbatch
