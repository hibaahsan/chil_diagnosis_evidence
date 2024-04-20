#!/bin/bash
#
#SBATCH --job-name=encode_notes
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=16G
#SBATCH --time=03:00:00

cd ../src

python -u retrieval_encode_notes.py -input_path $NOTE_PATH -encoded_path $ENCODED_PATH -sentences_path $SENTENCES_PATH -tokenizer $MODEL_PATH -cache_path $CACHE_PATH





