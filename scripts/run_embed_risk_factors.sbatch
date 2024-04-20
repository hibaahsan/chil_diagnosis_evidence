#!/bin/bash
#
#SBATCH --job-name=embed_risk_factors
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=32G
#SBATCH --time=10:00
#SBATCH --gpus=1


INPUT_PATH="../data/gpt_risk_factors.txt"
OUTPUT_PATH="gpt_risk_factors_embed.p"
MODEL_PATH="emilyalsentzer/Bio_ClinicalBERT"
CACHE_PATH="" #Huggingface cache path

cd ../src

python -u embed_risk_factors.py -input_path $INPUT_PATH -output_path $OUTPUT_PATH -model_path $MODEL_PATH -cache_path $CACHE_PATH





