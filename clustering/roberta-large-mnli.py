import os
import pickle as pkl
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import settings
from func.metric import *

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli", cache_dir = None)
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli", cache_dir = None).to(device)

def CleanseScore(resultDict): 

    sequences = []

    for _, sample in tqdm(enumerate(resultDict), total=len(resultDict)):
        question = sample['question'] 
        generated_texts = sample['generations'] # 10 generations
        last_embeddings = sample['last_embeddings'] # hidden states of 10 generations
        cosine_total = sample['cosine_total']
        if last_embeddings is None: # None cannot be subscriptable
            cleanse_score = 0.0  
        else: 
            clusters = [[{'answer': generated_texts[0], 'embeddings': last_embeddings[0, :]}]]

            for i in range(1, len(generated_texts)):

                already_add = False

                for c in clusters:
                    
                    qa_1 = question + ' ' + generated_texts[i]
                    qa_2 = question + ' ' + c[0]['answer']

                    encoded_input = tokenizer(qa_1, qa_2, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512).to(device) 
                    prediction = model(**encoded_input).logits
                    predicted_label = torch.argmax(prediction, dim = 1)

                    encoded_reverse_input = tokenizer(qa_2, qa_1, return_tensors = "pt", padding="max_length", truncation = True, max_length = 512).to(device)
                    reverse_prediction = model(**encoded_reverse_input).logits
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    if predicted_label.item() == 2 and reverse_predicted_label.item() == 2: 
                        c.append({'answer': generated_texts[i], 'embeddings': last_embeddings[i, :]})
                        already_add = True
                        break
                
                if not already_add:
                    clusters.append([{'answer': generated_texts[i], 'embeddings': last_embeddings[i]}])

            print("# of clusters:", len(clusters))

            total_similarity = cosine_total
            intra_similarity = 0.0
            for c in clusters:
                num_cluster = len(c)
                if num_cluster < 2: 
                    continue 
                for i in range(num_cluster):
                    for j in range(i+1, num_cluster): 
                        intra_similarity += cosine_similarity_v2(c[i]['embeddings'], c[j]['embeddings'])
            print(f"total similarity: {total_similarity}")
            print(f"intra similarity: {intra_similarity}")
            cleanse_score = intra_similarity / total_similarity
            print("cleanse_score:", cleanse_score)

        cosine_score, _ = compute_CosineSimilarity_v2(last_embeddings)
        print(f"cosine_score:", cosine_score)

        curr_seq = dict(
            num_of_cluster = len(clusters),
            cleanse_score = cleanse_score
        )
        sequences.append(curr_seq)
    return sequences

if __name__ == "__main__":

    open_dir = 'mistral-7b_coqa_for_clustering'
    file_name = f"/mnt/aix7101/minsuh-output/{open_dir}/0.pkl"
    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    dir = 'mistral-7b_coqa_roberta'
    cache_dir = os.path.join(settings.GENERATION_FOLDER, dir) 
    os.makedirs(cache_dir, exist_ok=True) 
    
    sequences = CleanseScore(resultDict)
    pd.to_pickle(sequences, os.path.join(cache_dir, '0.pkl'))