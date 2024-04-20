import os
import re
import nltk
import spacy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompt to extract risks/signs")
    parser.add_argument('-evidence_path', type=str, required=True, help='path to evidence')
    parser.add_argument('-encoded_path', type=str, required=True, help='path to evidence')
    parser.add_argument('-output_path', type=str, required=True, help='path to output')
    
    set_seed(42)
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evidence_df = pd.read_csv(args.evidence_path, sep='\t')

    model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', device_map='auto')
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'
    pad_side = 'left'

    print(args.evidence_path)

    print('Find hallucinations')

    template = "Read the following clinical note of a patient: {0}\
    \nQuestion: Does the patient have {1}? Answer Yes or No."
    
    with open(args.encoded_path, "rb") as f:
        data = pickle.load(f)
        
    text_df = pd.DataFrame.from_dict({'ROW_ID': data['row_id'], 'NOTE_ID': data['note_id'], 'CHUNK_ID': data['chunk_id'], 'TEXT': data['text']})
    unique_text_df = text_df[['ROW_ID', 'NOTE_ID', 'CHUNK_ID', 'TEXT']].groupby(['ROW_ID', 'NOTE_ID', 'CHUNK_ID']).first().reset_index()
    
    extracted_risks_df = pd.read_csv(args.evidence_path, sep='\t')
    
    print('Before: ', len(extracted_risks_df))
    
    extracted_risks_df = extracted_risks_df.merge(unique_text_df, on=['ROW_ID', 'NOTE_ID', 'CHUNK_ID'], how='left')
    
    print('After: ', len(extracted_risks_df))

    extracted_risks_df['IS_PRESENT'] = ''

    count=0
    

    for ix, row in extracted_risks_df.iterrows():
        ev = row['EVIDENCE']
        note = row['TEXT']

        text = template.format(note, ev)

        messages = [
        {"role": "user", "content": text}
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=1000)
        decoded = tokenizer.batch_decode(generated_ids)
        is_true = decoded[0].split('[/INST]')[1]
        is_true = is_true[:-4]

        extracted_risks_df.at[ix, 'IS_PRESENT'] = is_true


    extracted_risks_df.to_csv(args.output_path, sep='\t', index=False)




