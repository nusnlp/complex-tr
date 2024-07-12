from transformers import AutoTokenizer, AutoModel
import json
import sys
import math
import torch
import pandas as pd
import numpy as np




split = sys.argv[2]
data = [json.loads(line) for line in open('temp_reason_v5/{}_processed_gcs.json'.format(split))]
batch_size = int(sys.argv[1])
gpu = int(sys.argv[3])
print(batch_size)
device = torch.device('cuda:{}'.format(gpu))
print(device)
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(device) 
model.eval()
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def batch_encode_sentences(tokenizer, model, device, batch_size, example):
    query = example['question']
    sentences = [query]
    for passage in example['passages']:
        sentences.append(' '.join([passage['title'], passage['text']]))
    dataloader = torch.utils.data.DataLoader(sentences, batch_size=batch_size)
    all_embeddings = [] 
    for i, batch in enumerate(dataloader):
        #print(i)
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        #print(embeddings.size())
        all_embeddings.append(embeddings.detach().cpu())
        del embeddings, outputs, inputs
    
    return torch.cat(all_embeddings, dim=0 )



def get_passage_ranking(split, tokenizer, model, device, batch_size, data):
    
    for e_id, example in enumerate(data):
        df = pd.DataFrame()
        print(e_id)
        sent_embeddings = batch_encode_sentences(tokenizer, model, device, batch_size, example)
        query = sent_embeddings[0]
        passage_embs = sent_embeddings[1:]
        similarity_scores =  (query @ passage_embs.T).numpy()
        df['scores'] = similarity_scores
        df['rank'] = df['scores'].rank(ascending=False)
        #sorted_indices = similarity_scores.argsort()[::-1][:n]
        
        for p_idx, passage in enumerate(example['passages']):
            passage['bge_rank'] = df['rank'].iloc[p_idx]
        del passage_embs, similarity_scores, query, sent_embeddings
    with open('temp_reason_v5/{}_processed_gcs.json'.format(split), 'w') as fout:
        for line in data:
            fout.write(json.dumps(line) + '\n')
            
    return None

get_passage_ranking(split, tokenizer, model, device, batch_size, data)
print('Done')
