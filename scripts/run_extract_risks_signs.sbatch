#!/bin/bash
#
#SBATCH --job-name=find_hallucinations
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1

cd ../src

FILE_NAME="filename" #same as FILE_NAME passed to zero_shot_pipeline.sh
SUFFIX="_suffix" #same as SUFFIX passed to zero_shot_pipeline.sh
FOLDER="" #same as FOLDER passed to zero_shot_pipeline.sh

EVIDENCE_PATH="${FOLDER}/two_step_output_${FILE_NAME}${SUFFIX}.csv" #this is the output generated after running zero_shot_pipeline.sh 
OUTPUT_PATH="${FOLDER}/extracted_risks_${FILE_NAME}${SUFFIX}.csv" #output path


python3 -u extract_risks_signs.py -evidence_path $EVIDENCE_PATH -output_path $OUTPUT_PATH