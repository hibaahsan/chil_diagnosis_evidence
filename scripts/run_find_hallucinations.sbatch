#!/bin/bash
#
#SBATCH --job-name=find_hallucinations
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1

FILE_NAME="filename" #same as FILE_NAME passed to zero_shot_pipeline.sh
SUFFIX="_suffix" #same as SUFFIX passed to zero_shot_pipeline.sh
FOLDER="" #same as FOLDER passed to zero_shot_pipeline.sh

TRANSFORMERS_OFFLINE=1 
HF_DATASETS_OFFLINE=1
TOKENIZERS_PARALLELISM=false

ENCODED_PATH="${FOLDER}/encoded_${FILE_NAME}${SUFFIX}.p" #path to encoded input generated using zero_shot_pipeline.sh
EVIDENCE_PATH="${FOLDER}/extracted_risks_${FILE_NAME}${SUFFIX}.csv" #path to extracted risks and signs using run_exteact_signs.sbatch
OUTPUT_PATH="${FOLDER}/is_present_${FILE_NAME}${SUFFIX}.csv" #output path

cd ../src


python3 -u find_hallucinations.py -encoded_path $ENCODED_PATH -evidence_path $EVIDENCE_PATH -output_path $OUTPUT_PATH