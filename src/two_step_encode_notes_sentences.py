import re
import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from utils import write_cache, pad_and_mask
import spacy

def chunk_sent_encode(df, tokenizer, stemplate, binary_etemplate, reason_etemplate, model_name, pad_side='right', max_length=750):
    '''
    df: Pandas dataframe that contains:
        ROW_ID: This has to be unique to every row (does not correspond to MIMIC ROW_ID).
        NOTE_ID: Unique to patient note. It is possible the df has multiple rows with the same note and different diagnosis.
        TEXT: Patient note
        DIAGNOSIS: diagnosis. Eg: hypertension
    stemplate: prompt start template - Read the following clinical note of a patient:
    binary_etemplate: binary prompt end template - Is the patient at risk of [DIAGNOSIS]?
    reason_etemplate: binary prompt end template - Why is the patient at risk of [DIAGNOSIS]?
    model_name: model name
    pad_side: padding side (right for FLAN, left for Mistral)
    max_length: max chunk length
    '''

    row_ids, note_ids, label_texts = [], [], []
    binary_encodings, binary_masks, reason_encodings, reason_masks = [], [], [], []
    note_type, texts, chunk_ids, sentences = [], [], [], []

    nlp = spacy.load('en_core_web_sm')


    global_count = 0

    for ix, row in df.iterrows():
        row_text = row['TEXT']
        row_text = re.sub('\_+', '\n', row_text)
        row_text = re.sub('\s+', ' ', row_text)

        sents = list(nlp(row_text).sents)
        sents = [s.text for s in sents]
        sent_ids = [k for k in range(len(sents))]

        sentences.append({'ROW_ID': row['ROW_ID'], 
                        'NOTE_ID': row['NOTE_ID'], 
                        'SENTENCE_IDS': sent_ids,
                        'SENTENCES': sents})
        label_text = row['DIAGNOSIS']

            
        pstart = stemplate
        binary_pend = binary_etemplate.format(label_text)
        reason_pend = reason_etemplate.format(label_text)
                                                  

        # if pad_side == 'right':
        if 'flan' in model_name.lower():
            encoded_pstart = tokenizer(pstart, add_special_tokens=False).input_ids
            encoded_binary_pend = tokenizer(binary_pend).input_ids
            encoded_reason_pend = tokenizer(reason_pend).input_ids
        elif 'mistral' in model_name:
            encoded_pstart = tokenizer(pstart, add_special_tokens=False).input_ids
            encoded_binary_pend = tokenizer(binary_pend, add_special_tokens=False).input_ids
            encoded_reason_pend = tokenizer(reason_pend, add_special_tokens=False).input_ids

            
            
        len_pend = max(len(encoded_binary_pend), len(encoded_reason_pend))
        len_pstart = len(encoded_pstart)


        text_length = max_length - len_pend - len_pstart #to ensure that the chunks are the same during binary and reason prompting.
        i = 0

        sent_accum = []
        text_accum = []
        for s in sents:
            e = tokenizer(s, add_special_tokens=False).input_ids

            if len(sent_accum) + len(e)>text_length:
                binary_enc = encoded_pstart + sent_accum + encoded_binary_pend
                binary_enc, binary_mask = pad_and_mask(binary_enc, max_length, pad_side)

                reason_enc = encoded_pstart + sent_accum + encoded_reason_pend
                reason_enc, reason_mask = pad_and_mask(reason_enc, max_length, pad_side)

                i+=1

                texts.append(' '.join(text_accum))
                binary_encodings.append(binary_enc)
                binary_masks.append(binary_mask)
                reason_encodings.append(reason_enc)
                reason_masks.append(reason_mask)
                row_ids.append(row['ROW_ID'])
                note_ids.append(row['NOTE_ID'])
                label_texts.append(label_text)
                chunk_ids.append(i-1)
                note_type.append(row['NOTE_TYPE'])

                sent_accum = []
                text_accum = []

                    
            if len(e)<text_length: 
                sent_accum =  sent_accum + e
                text_accum.append(s)

            binary_enc = encoded_pstart + sent_accum + encoded_binary_pend
            binary_enc, binary_mask = pad_and_mask(binary_enc, max_length, pad_side)

            reason_enc = encoded_pstart + sent_accum + encoded_reason_pend
            reason_enc, reason_mask = pad_and_mask(reason_enc, max_length, pad_side)

            i+=1

            texts.append(' '.join(text_accum))
            binary_encodings.append(binary_enc)
            binary_masks.append(binary_mask)
            reason_encodings.append(reason_enc)
            reason_masks.append(reason_mask)
            row_ids.append(row['ROW_ID'])
            note_ids.append(row['NOTE_ID'])
            label_texts.append(label_text)
            chunk_ids.append(i-1)
            note_type.append(row['NOTE_TYPE'])                

            global_count +=1

            
    binary_encodings = np.array(binary_encodings)
    binary_masks = np.array(binary_masks)
    reason_encodings = np.array(reason_encodings)
    reason_masks = np.array(reason_masks)

    output = {}
    output['text'] = texts
    output['binary_encoding'] = binary_encodings
    output['binary_mask'] = binary_masks
    output['reason_encoding'] = reason_encodings
    output['reason_mask'] = reason_masks
    output['row_id'] = row_ids
    output['note_id'] = note_ids
    output['label_text'] = label_texts
    output['chunk_id'] = chunk_ids
    output['note_type'] = note_type

    

    return output, sentences



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode Notes")
    parser.add_argument('-input_path', type=str, required=True, help='path to dataframe with notes, corresponding diagnosis, and risk factors')
    parser.add_argument('-encoded_path', type=str, required=True, help='path to prompt encoded data')
    parser.add_argument('-sentences_path', type=str, required=True, help='path to sentences')
    parser.add_argument('-model_name', type=str, required=True, help='model name')
 
    
    args = parser.parse_args()

    summaries = pd.read_csv(args.input_path, sep='\t')


    print('Tokenizer: ', args.model_name, args.sentences_path, args.encoded_path)
    print('Number of records: ', len(summaries))

    if 'mistral' in args.model_name:
        print('Padding left')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map='auto')
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        pad_side = 'left'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map='auto')
        pad_side = 'right'

    
    prompt_start = 'Read the following clinical note of a patient: '

    binary_prompt_end = '\n\nQuestion: Is the patient at risk of {0}? Choice: -Yes -No\n\nAnswer: '
    reason_prompt_end = '\n\nAnswer step by step: based on the note, why is the patient at risk of {0}?\n\nAnswer: '
    if 'mistral' in args.model_name:
        prompt_start = '<s>[INST] Read the following clinical note of a patient: '
        binary_prompt_end = '\n\nQuestion: Is the patient at risk of {0}? Choice: -Yes -No\n\nAnswer: [/INST]'
        reason_prompt_end = '\n\nBased on the note, why is the patient at risk of {0}? Be concise.\n\nAnswer: [/INST]'

    risk_output, sentences = chunk_sent_encode(summaries, tokenizer, prompt_start, binary_prompt_end, reason_prompt_end, args.model_name, pad_side=pad_side)
    size = len(risk_output['text'])
    ev_type_arr = ['risk']*size
    risk_output['evidence_type'] = np.array(ev_type_arr)


    binary_prompt_end = '\n\nQuestion: Does the patient have {0}? Choice: -Yes -No\n\nAnswer: '
    reason_prompt_end = '\n\nQuestion: Extract signs of {0} from the note.\n\nAnswer: '
    if 'mistral' in args.model_name:
        binary_prompt_end = '\n\nQuestion: Does the patient have {0}? Choice: -Yes -No\n\nAnswer: [/INST]'
        reason_prompt_end = '\n\nQuestion: Extract signs of {0} from the note. Be concise.\n\nAnswer: [/INST]'

    sign_output, _ = chunk_sent_encode(summaries, tokenizer, prompt_start, binary_prompt_end, reason_prompt_end, args.model_name, pad_side=pad_side)
    size = len(sign_output['text'])
    ev_type_arr = ['sign']*size
    sign_output['evidence_type'] = np.array(ev_type_arr)

    output = {}
    for k in sign_output.keys():
        output[k] = np.concatenate([risk_output[k], sign_output[k]])
        print(k, output[k].shape)
        
    write_cache(args.encoded_path, output)

    sentences_df = pd.DataFrame.from_dict(sentences)
    sentences_df.to_csv(args.sentences_path, sep='\t', index=False)

    


