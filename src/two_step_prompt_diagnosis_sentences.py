import spacy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, AutoModelForCausalLM, set_seed
from sklearn.feature_extraction.text import TfidfVectorizer


import sys
sys.path.append('..')
from utils import mean_pooling
from dataset import DiagnosisDataset


def evidence_prompt(model, tokenizer, dataloader, device):
    text_outputs = []

    if 'flan' in tokenizer.name_or_path:
        max_length = 256
    elif 'mistral' in tokenizer.name_or_path:
        max_length = 2000

    with torch.no_grad():
            for i, batch in enumerate(dataloader):
                seq, src_mask = batch
                seq = seq.to(device)
                src_mask = src_mask.to(device)

                inputs = {'input_ids': seq, 'attention_mask': src_mask}

                outputs = model.generate(**inputs, 
                            max_new_tokens=max_length, 
                            return_dict_in_generate=True,
                            output_scores=True)
                
                text_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                text_outputs.extend(text_output)

    return text_outputs

def binary_prompt(model, tokenizer, dataloader, device):
    yes_id = tokenizer.encode('Yes', add_special_tokens=False)[0]
    no_id = tokenizer.encode('No', add_special_tokens=False)[0]
    all_scores = []

    print(no_id, yes_id)

    if 'flan' in tokenizer.name_or_path:
        max_length = 3
    elif 'mistral' in tokenizer.name_or_path:
        max_length = 1500

    with torch.no_grad():
        with tqdm(dataloader, unit="bt") as pbar: 
            for i, batch in enumerate(pbar):
                seq, src_mask = batch
                seq = seq.to(device)
                src_mask = src_mask.to(device)

                inputs = {'input_ids': seq, 'attention_mask': src_mask}

                outputs = model.generate(**inputs, output_scores=True, max_length=max_length, return_dict_in_generate=True)
                scores = outputs['scores'][0][:, [no_id,yes_id]].cpu().numpy()

                scores = softmax(scores, axis=1)

                all_scores.extend(scores[:,1])

    N = len(all_scores)
    all_scores = np.array(all_scores)
    all_preds = np.zeros(N)
    all_preds[all_scores>0.5] = 1

    pos_ix = np.argwhere(all_preds == 1).squeeze().astype(int)

    return pos_ix, all_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompt notes for diagnosis evidence")
    parser.add_argument('-encoded_path', type=str, required=True, help='path to prompt encoded data')
    parser.add_argument('-sentences_path', type=str, required=True, help='path to note sentences')
    parser.add_argument('-output_path', type=str, required=True, help='path to output. stores reasoning output of rows that survive binary prompting.')
    parser.add_argument('-scores_path', type=str, required=True, help='path to scores. stores all the scores')
    parser.add_argument('-model_name', type=str, help='model name')
    set_seed(42)
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cbert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    cbert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    cbert_model = cbert_model.to(device)

    print('Model: ', args.model_name)

    if 'flan' in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map='auto', load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map='auto')  
    elif 'mistral' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map='auto')
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.bfloat16)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'

    model.eval()

    f = open(args.encoded_path, 'rb')
    data = pickle.load(f)
    f.close()

    print('Binary prompt...')

    pos_ix = np.arange(len(data['reason_encoding']))
    
    row_ids = np.array(data['row_id'])
    note_ids = np.array(data['note_id'])
    chunk_ids = np.array(data['chunk_id'])
    label_texts = np.array(data['label_text'])
    note_type = np.array(data['note_type'])
    evidence_type = np.array(data['evidence_type'])

    dataset = DiagnosisDataset(data['binary_encoding'], data['binary_mask'])
    dataloader = DataLoader(
        dataset,
        batch_size=8, 
        shuffle=False, 
        num_workers=1)

    pos_ix, scores = binary_prompt(model, tokenizer, dataloader, device)

    print('Number of chunks with evidence: ', len(pos_ix))
        
    pos_ix = np.array(pos_ix)

    fo = open(args.scores_path, 'wb')
    pickle.dump((row_ids, note_ids, chunk_ids, scores), fo)   
    fo.close()


    #########Prompt for evidence###########
    dataset = DiagnosisDataset(data['reason_encoding'][pos_ix], data['reason_mask'][pos_ix])
    dataloader = DataLoader(
        dataset,
        # batch_size=12,
        batch_size=4, 
        shuffle=False, 
        num_workers=1)

    print('Evidence prompt...')

    evidence = evidence_prompt(model, tokenizer, dataloader, device)

    output_df = pd.DataFrame.from_dict({'ROW_ID': row_ids[pos_ix],
    'NOTE_ID': note_ids[pos_ix],
    'NOTE_TYPE': note_type[pos_ix],
    'CHUNK_ID': chunk_ids[pos_ix],
    'DIAGNOSIS': label_texts[pos_ix],
    'EVIDENCE_TYPE': evidence_type[pos_ix],
    'OUTPUT': evidence})

    if 'mistral' in args.model_name:
        output_df['OUTPUT'] = output_df['OUTPUT'].str.split('\[/INST\] ').str[1]

    output_df.to_csv(args.output_path, sep='\t', index=False)




