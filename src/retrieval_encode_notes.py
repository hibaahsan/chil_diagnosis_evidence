import os
import spacy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import write_cache
from transformers import AutoTokenizer

def encode(df, tokenizer):
    texts, row_ids, note_ids, sentence_ids, encodings, masks, labels, label_texts, note_types = [], [], [], [], [], [], [], [], []
    sentences = []
    nlp = spacy.load('en_core_web_sm')


    for ix, row in df.iterrows():
        text, label_text, row_id, note_id, note_type = row['TEXT'], row['DIAGNOSIS'], row['ROW_ID'], row['NOTE_ID'], row['NOTE_TYPE']
        
        sents = list(nlp(text).sents)
        sents = [s.text for s in sents]
        x = tokenizer(sents, return_tensors='np', truncation=True, padding='max_length', max_length=128)
        num_sent = len(sents)
        sent_ids = [k for k in range(num_sent)]

        sentences.append({'ROW_ID': row_id, 
                        'NOTE_ID': note_id, 
                        'SENTENCES': sents})

        texts.extend(sents)
        encodings.append(x['input_ids'])
        masks.append(x['attention_mask'])
        labels.extend(num_sent*[1])
        label_texts.extend(num_sent*[label_text])
        row_ids.extend(num_sent*[row_id])
        note_ids.extend(num_sent*[note_id])
        sentence_ids.extend(sent_ids)
        note_types.extend(num_sent*[note_type])
    
    encodings = np.vstack(encodings)
    masks = np.vstack(masks)
    labels = np.array(labels)
    sentence_ids = np.array(sentence_ids)

    print(encodings.shape, masks.shape, len(labels), len(label_texts))

    output = {}
    output['sentence'] = texts
    output['encoding'] = encodings
    output['mask'] = masks
    output['label'] = labels
    output['label_text'] = label_texts
    output['row_id'] = row_ids
    output['note_id'] = note_ids
    output['sentence_id'] = sentence_ids
    output['note_type'] = note_types
    
    return output, sentences



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode Notes for Retrieval")
    parser.add_argument('-input_path', type=str, required=True, help='path to dataframe with notes, corresponding diagnosis, and risk factors')
    parser.add_argument('-tokenizer', type=str, required=True, help='model tokenizer')
    parser.add_argument('-encoded_path', type=str, required=True, help='path to prompt encoded data')
    parser.add_argument('-sentences_path', type=str, required=True, help='path to sentences')
    parser.add_argument('-cache_path', type=str, required=True, help='path to cache')

    
    args = parser.parse_args()
    summaries = pd.read_csv(args.input_path, sep='\t')

    print('Number of records: ', len(summaries))


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_path)
    output, sentences = encode(summaries, tokenizer)
    write_cache(args.encoded_path, output)

    sentences_df = pd.DataFrame.from_dict(sentences)
    sentences_df.to_csv(args.sentences_path, sep='\t', index=False)




