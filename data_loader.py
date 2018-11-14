from gensim.models import word2vec
import nltk
import logging 


import json
from tqdm import tqdm
import pandas as pd
import ast
import csv
import re 
import numpy as np
import os
# data_path = './yelp_dataset/'
# file_name = 'yelp_academic_dataset_review.json'
# data_export = './yelp_dataset/data_split/'

def data_convert_csv(data_path, file_name):
    print('converting json file into csv file')
    with open(data_path + file_name) as json_file:      
        data = json_file.readlines()  # list of string  
    # store the data in csv file 
    with open(data_path+ 'yelp_academic_dataset_review.csv', 'w') as f:
        # write the heading 
        review_dict = ast.literal_eval(data[0])
        heading = review_dict.keys()
        writer = csv.writer(f)
        writer.writerow(list(heading))

        for index in tqdm(range(len(data))):
            review_dict = json.loads(data[index])
            writer = csv.writer(f)
            writer.writerow(review_dict.values())
    print('converting complete !!!')

def data_sample_csv(data_path, file_name):
    print('converting json file into csv file')
    with open(data_path + file_name) as json_file:      
        data = json_file.readlines()  # list of string 
    print("finished json file load") 
    # store the data in csv file 
    with open(data_path+ 'sample_yelp_academic_dataset_review_1000.csv', 'w') as f:
        # write the heading 
        review_dict = ast.literal_eval(data[0])
        heading = review_dict.keys()
        writer = csv.writer(f)
        writer.writerow(list(heading))
        for index in tqdm(range(len(data[:1000]))):
            review_dict = json.loads(data[index])
            writer = csv.writer(f)
            writer.writerow(review_dict.values())
    print('converting complete !!!')

def split_feature_label(data_path,data_export):
    df = pd.read_csv(data_path + 'sample_yelp_academic_dataset_review_1000.csv',low_memory=False)
    # print('finished load data')

    num_samples = len(df['stars'])

    with open(data_export + 'train/sample_train.train','w') as f:
        for i in range(num_samples):
            label = str(df['stars'][i])
            raw_text = df['text'][i]
            '''
            raw_text = [re.split(r'[.,!?;()#$%&*+,-./:<=>@[\]^_`{|}~\d"]', x) for x in raw_text.lower().split(' ')]
            for t in raw_text:
                for item in t:
                    if item != '':
                        text.append(item)
            text = ' '.join(item)
            '''
            raw_text.replace('\n',' ').replace('\t', ' ').replace('\n\n', ' ').replace('\t\t', ' ')
            f.write(label + '\t\t\t'  + raw_text + '\n')

    
def loadData(file_path):
    label_list = []
    text_list = []
    cnt = 0
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            #if cnt > 1000:
            #    break
            #cnt += 1
            line = line.strip().split('\t\t\t')
            try:
                label = int(line[0])
                label_list.append(label)
                text = line[1].split(' ')
                text_list.append(text)
            except ValueError:
                last_text = text_list.pop(-1)
                last_text += line[0].split(' ') # for here using line[0] since the unlabeled text has no label
                text_list.append(last_text)
    return label_list, text_list

def preprocessing(label_list, text_list):
    #TODO: normalization on the training data
    pass

def trainWordEmbedding(data_path, label_list, text_list, embed_dim=100, window_size = 20):
    if os.path.isfile(data_path+"word2vec.model"):
        print('word2vec existing')
        model = word2vec.Word2Vec.load(data_path+"word2vec.model")
    else:
        logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO) 
        model = word2vec.Word2Vec(text_list, size=embed_dim, window=5, min_count=0, workers=4)
        model.save(data_path+"word2vec.model")

    print('start embedding')
    train_feature = []
    train_label = []
    for i in tqdm(range(len(text_list))):
        sentence = text_list[i]
        label = label_list[i]
        sentence_vector = []

        for item in sentence:
            vector = model.wv[item]
            sentence_vector.append(vector)

        for s in range(len(sentence_vector)-19):
            label_emb = [0] * 5
            train_sample = np.array((sentence_vector[s:s+20])).reshape(window_size,embed_dim,1)
            label_emb[label-1] = 1
            train_label.append(label_emb)
            train_feature.append(train_sample)
        # print(train_feature.shape)

    return train_label, train_feature
        

