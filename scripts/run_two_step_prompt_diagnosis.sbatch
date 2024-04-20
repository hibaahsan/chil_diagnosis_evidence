#!/bin/bash
#
#SBATCH --job-name=prompt_surgery
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1

cd ../src

python -u two_step_prompt_diagnosis_sentences.py -encoded_path $ENCODED_PATH -output_path $OUTPUT_PATH -scores_path $SCORES_PATH -sentences_path $SENTENCES_PATH -model_name $MODEL_NAME






