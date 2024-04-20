import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import mean_pooling
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

class RetrievalDataset(Dataset):  
    def __init__(self, input_ids, att_mask, diagnosis, diag_embeddings):
        self.input_ids = input_ids
        self.att_mask = att_mask
        self.diagnosis = diagnosis
        self.diag_embeddings = diag_embeddings
        
    def __len__(self):    
        return len(self.input_ids)

    def __getitem__(self, index):  
        input_id = torch.from_numpy(self.input_ids[index,:]).int()
        mask = torch.from_numpy(self.att_mask[index,:]).bool()  
        d = self.diagnosis[index]
        diag_embedding = self.diag_embeddings[d.lower()] 
        return input_id, mask, diag_embedding 

def compute_similarity(model, dataloader, device):
    all_scores = []
    model = model.to(device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        with tqdm(dataloader, unit="bt") as pbar:
            for i, batch in enumerate(pbar):
                seq, src_mask, diag_embedding = batch
                seq = seq.to(device)
                src_mask = src_mask.to(device)
                diag_embedding = diag_embedding.to(device)

                outputs = model(seq, attention_mask=src_mask, return_dict=True, output_hidden_states=True)
                last_hidden_state = outputs.last_hidden_state
                batch_embedding = mean_pooling(last_hidden_state, src_mask)
                scores = cos(batch_embedding, diag_embedding)
                all_scores.extend(scores.cpu().numpy())

    return all_scores

def get_topk_sentences(data, scores, k=20):
    df = pd.DataFrame.from_dict({'ROW_ID': data['row_id'],
                                'NOTE_ID': data['note_id'],
                                'NOTE_TYPE': data['note_type'],
                                'SENTENCE': data['sentence'],
                                'SENTENCE_ID': data['sentence_id'],
                                'DIAGNOSIS': data['label_text'],
                                'SCORE': scores})


    df['RANK'] = df.groupby(['ROW_ID', 'DIAGNOSIS'])['SCORE'].rank(method='first', ascending=False)
    df = df[df['RANK']<=k].copy()
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompt notes for diagnosis evidence")
    parser.add_argument('-encoded_path', type=str, required=True, help='path to encoded data')
    parser.add_argument('-output_path', type=str, required=True, help='path to output that contains scores and ranks')
    parser.add_argument('-sentences_path', type=str, required=True, help='path to note sentences')
    parser.add_argument('-model_path', type=str, required=True, help='hugging face model')
    parser.add_argument('-cache_path', type=str, required=True, help='path to cache')
    parser.add_argument('-rf_embeddings_path', type=str, required=True, help='path to risk factors embeddings')
    
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModel.from_pretrained(args.model_path, cache_dir=args.cache_path)

    with open(args.encoded_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(args.rf_embeddings_path, 'rb') as f: 
        rf_embeddings = pickle.load(f)

    
    dataset = RetrievalDataset(data['encoding'], data['mask'], data['label_text'], rf_embeddings)

    dataloader = DataLoader(
        dataset,
        batch_size=12, 
        shuffle=False, 
        num_workers=1)

    scores = compute_similarity(model, dataloader, device)
    top_df = get_topk_sentences(data, scores)
    top_df.to_csv(args.output_path[:-4]+'_sentences.csv', sep='\t', index=False)
    
    # id_top_df = top_df.groupby(['ROW_ID', 'NOTE_ID', 'NOTE_TYPE', 'DIAGNOSIS'])['SENTENCE_ID'].apply(list).reset_index(name='EVIDENCE_SENTENCE_IDS')
    # sent_top_df = top_df.groupby(['ROW_ID', 'NOTE_ID', 'NOTE_TYPE', 'DIAGNOSIS'])['SENTENCE'].apply(list).reset_index(name='OUTPUT')
    
    # sentences_df = pd.read_csv(args.sentences_path, sep='\t')
    # id_top_df = id_top_df.merge(sentences_df, on=['ROW_ID', 'NOTE_ID'])
    # top_df = id_top_df.merge(sent_top_df, on=['ROW_ID', 'NOTE_ID', 'NOTE_TYPE', 'DIAGNOSIS'])
    # top_df.to_csv(args.output_path, sep='\t', index=False)
















