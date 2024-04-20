import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import write_cache, mean_pooling
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class RFDataset(Dataset):  
	def __init__(self, input_ids, attn_masks):
		self.input_ids = input_ids
		self.attn_masks = attn_masks
        
	def __len__(self):    
		return len(self.input_ids)

	def __getitem__(self, index):  
		input_ids = self.input_ids[index]
		masks = self.attn_masks[index]
		return input_ids, masks

def embed_factors(rf_df, tokenizer, model, device):
	embeddings = []
	model = model.to(device)
	unique_diag = list(rf_df['DIAGNOSIS'])

	rf_df['FACTORS'] = 'risk factors of ' + rf_df['DIAGNOSIS'] + ' include ' + rf_df['FACTORS']

	factors = list(rf_df['FACTORS'])
	diagnosis = np.array(rf_df['DIAGNOSIS'])

	factors = [f.strip().lower() for f in factors]
	
	encoded_factors = tokenizer(factors, return_tensors='np', truncation=True, padding=True)

	dataset = RFDataset(encoded_factors['input_ids'], encoded_factors['attention_mask'])
	dataloader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=1)

	with torch.no_grad():
		with tqdm(dataloader, unit="bt") as pbar: 
			for i, batch in enumerate(pbar):
				input_ids, masks = batch
				input_ids = input_ids.long().to(device)
				masks = masks.bool().to(device)
				outputs = model(input_ids, attention_mask=masks, return_dict=True, output_hidden_states=True)
				last_hidden_state = outputs.last_hidden_state
				#print(last_hidden_state.shape)

				batch_embedding = mean_pooling(last_hidden_state, masks)
				batch_embedding = batch_embedding.cpu().numpy()
				embeddings.append(batch_embedding)

	embeddings = np.vstack(embeddings)	
	print('shape: ', embeddings.shape)


	factor_embeddings = {}	
	for i, d in enumerate(unique_diag):
		ixx = np.argwhere(diagnosis == d).squeeze()
		factor_embeddings[d] = embeddings[ixx]
		print(d, ixx, factor_embeddings[d].shape)

	print('Number of risk factor embeddings: ', len(factor_embeddings))

	
	return factor_embeddings

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Embed risk factors")
	parser.add_argument('-input_path', type=str, required=True, help='path to risk factors')
	parser.add_argument('-model_path', type=str, required=True, help='model')
	parser.add_argument('-output_path', type=str, required=True, help='path to save embeddings')
	parser.add_argument('-cache_path', type=str, required=True, help='path to cache')

	args = parser.parse_args()

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = AutoModel.from_pretrained(args.model_path, cache_dir=args.cache_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_path)

 
	rf_df = pd.read_csv(args.input_path, sep='\t')
	rf_df['DIAGNOSIS'] = rf_df['DIAGNOSIS'].str.lower()
	rf_df['FACTORS'] = rf_df['FACTORS'].str.lower()
	embeddings = embed_factors(rf_df, tokenizer, model, device)
	
	write_cache(args.output_path, embeddings)
    
	





