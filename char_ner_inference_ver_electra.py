import pandas as pd
import numpy as np
import os, re, csv, sys, json
import torch
import transformers
from transformers import ElectraTokenizer, ElectraForTokenClassification


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.get_device_name(0)

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def ner_tokenizer(sent, max_seq_length):    
    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable 
        pre_syllable = syllable

        input_ids[i] = (tokenizer.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1
    
    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent)+1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent)+1] = 1
    return {"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids}

def train_ner_inference(text) : 
  
    model.eval()
    text = text.replace(' ', '_')

    predictions , true_labels = [], []
    
    tokenized_sent = ner_tokenizer(text, len(text)+2)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)    
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    print('{}\t{}'.format("TOKEN", "TAG"))
    print("===========")
    # for token, tag in zip(tokenizer.decode(tokenized_sent['input_ids']), pred_tags):
    #   print("{:^5}\t{:^5}".format(token, tag))
    for i, tag in enumerate(pred_tags):
        print("{:^5}\t{:^5}".format(tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]), tag))
        
        
    return [(tokenizer.convert_ids_to_tokens(x), y) for x, y in zip(tokenized_sent['input_ids'], pred_tags)]

def ner_return(text):
    ner_tagged_list = train_ner_inference(text)
    
    tmp =''
    for txt, tag in ner_tagged_list:
       if tag != 'O':
           tmp += txt
       else:
           tmp += ' '
    
    if len(tmp.strip())==0:
        print("***** Cannot find the tagged word *****")
        for i in list(json_data.keys()):
            if i in text:
                print(f'***** But we found a baseball term for {i} *****')
                print(json_data[i])
                return json_data[i]
        return text
    else:
        ent = tmp.replace('#','').split()
        result = ' '.join(ent)
        print(result)
        return result


unique_tags = {'DT-B','DT-I','LC-B','LC-I','O','OG-B','OG-I','PS-B','PS-I','QT-B','QT-I','TI-B','TI-I'}
tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
id2tag = {id: tag for tag, id in tag2id.items()}

TOKENIZER_NAME = "monologg/koelectra-small-v3-discriminator"
JSON_NAME = "baseball_term.json"
MODEL_NAME = "checkpoint-13000"

with open(JSON_NAME, encoding='utf-8') as json_file:
    json_data = json.load(json_file)

tokenizer = ElectraTokenizer.from_pretrained(TOKENIZER_NAME)

pad_token_id = tokenizer.pad_token_id # 0
cls_token_id = tokenizer.cls_token_id # 101
sep_token_id = tokenizer.sep_token_id # 102
pad_token_label_id = tag2id['O']    # tag2id['O']
cls_token_label_id = tag2id['O']
sep_token_label_id = tag2id['O']

model = ElectraForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_tags))#13
model.to(device)

result = ner_return('19시즌 롯데 이대호 홈런 얼마나돼?')
result2 = ner_return('세이브가 뭐야')
result3 = ner_return('뭐야')
