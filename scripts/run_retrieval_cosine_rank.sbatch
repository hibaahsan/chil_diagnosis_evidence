#!/bin/bash
#
#SBATCH --job-name=retrieval_cosine
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

cd ../src

python -u retrieval_cosine_rank.py -encoded_path $ENCODED_PATH -sentences_path $SENTENCES_PATH -output_path $OUTPUT_PATH -model_path $MODEL_PATH -cache_path $CACHE_PATH -rf_embeddings_path $EMBEDDINGS_PATH





