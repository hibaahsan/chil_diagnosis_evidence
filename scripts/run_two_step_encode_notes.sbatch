#!/bin/bash
#SBATCH --job-name=encode_notes
#SBATCH --output=../output/res_%j.txt  # output file
#SBATCH -e ../output/res_%j.err        # File to which STDERR will be written
#SBATCH --mem=16G


cd ../src

python3 -u two_step_encode_notes_sentences.py -input_path $NOTE_PATH -encoded_path $ENCODED_PATH -sentences_path $SENTENCES_PATH  -model_name $MODEL_NAME




