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
    parser.add_argument('-output_path', type=str, required=True, help='path to extracted risks/sign')
    
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

    print('Extracting risks and signs...')

    risk_prompt_template = "Read the following statement: The patient is at risk of intracranial hemorrhage due to presence of an endotracheal tube (ETT) in the patient's trachea which may increase the risk of complications such as aspiration and airway obstruction.\
    \nQuestion: Extract the risk factors from the statement as a list. Be concise.\nAnswer: 1. presence of endotracheal tube (ETT) in the trachea.\
    \n\n\
    Read the following statement: {0}\
    \nQuestion: Extract the risk factors from the statement as a list. Be concise.\nAnswer: "

    sign_prompt_template = "Read the following statement: A patient may have intracranial hemorrhage because of the following report - Large left subdural hematoma, extensive subarachnoid hemorrhage, right temporal effacement, left uncal herniation, and effacement of the sulci.\
    \nQuestion: Extract the signs from the statement as a list. Be concise.\nAnswer: 1. Large left subdural hematoma\n2. Extensive subarachnoid hemorrhage\n3. Right temporal effacement\n4. Left uncal herniation\n5. Effacement of the sulci\
    \n\n\
    Read the following statement: A patient may have {0} because of the following report - {1}\
    \nQuestion: Extract the signs from the statement as a list. Be concise.\nAnswer: "


    row_ids, note_ids, chunk_ids, diags, evidences, ev_counts, is_relevant, evidence_types = [], [], [], [], [], [], [], []

    for ix, row in evidence_df.iterrows():
        output = row['SENTENCE'] if 'cbert' in args.evidence_path else row['OUTPUT']
        diag = row['DIAGNOSIS']
        row_id = row['ROW_ID']
        note_id = row['NOTE_ID']
        chunk_id = row['SENTENCE_ID'] if 'cbert' in args.evidence_path else row['CHUNK_ID']
        evidence_type = 'risk' if 'cbert' in args.evidence_path else row['EVIDENCE_TYPE']


        if evidence_type == 'risk':
            text = risk_prompt_template.format(output)
        else:
            text = sign_prompt_template.format(diag, output)


        messages = [
        {"role": "user", "content": text}
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=256)
        decoded = tokenizer.batch_decode(generated_ids)
        extracted_evidence = decoded[0].split('[/INST]')[1]
        extracted_evidence = extracted_evidence[:-4]
        pattern = r'\d+\. '
        extracted_evidence = re.split(pattern, extracted_evidence)

        yes_count = 0
        ev_count = 0

        for ev in extracted_evidence:
            if ev == ' ':
                continue

            ev_count+=1
                
            ev = ev.strip().lower()
            ev = re.sub('\.$', '', ev)

            if evidence_type == 'risk':
                text = "Is {0} a risk factor of {1}? Choice: -Yes -No. Be concise.\n\nAnswer: ".format(ev, diag)
            else:
                text = "A patient is showing the following sign: {0}.\
                \nQuestion: Can the sign indicate {1}? Choice: -Yes -No. Be concise.\n\nAnswer: ".format(ev, diag)

            messages = [
                {"role": "user", "content": text}
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)


            generated_ids = model.generate(model_inputs, max_new_tokens=128)
            decoded = tokenizer.batch_decode(generated_ids)
            
            
            if 'yes' in decoded[0].split('[/INST]')[1].lower():
                yes_count+=1
                is_relevant.append(1)
            else:
                is_relevant.append(0)

            
            row_ids.append(row_id)
            note_ids.append(note_id)
            chunk_ids.append(chunk_id)
            diags.append(diag)
            evidences.append(ev)
            ev_counts.append(ev_count)
            evidence_types.append(evidence_type)



    split_evidence_df = pd.DataFrame.from_dict({'ROW_ID': np.array(row_ids),
                                                'NOTE_ID': np.array(note_ids),
                                                'CHUNK_ID': np.array(chunk_ids),
                                                'DIAGNOSIS': np.array(diags),
                                                'EVIDENCE': np.array(evidences),
                                                'EVIDENCE_TYPE': np.array(evidence_types),
                                                'INDEX': np.array(ev_counts),
                                                'IS_RELEVANT': np.array(is_relevant)})

    split_evidence_df.to_csv(args.output_path, sep='\t', index=False)




