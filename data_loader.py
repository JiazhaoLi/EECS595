from gensim.models import Word2Vec
import nltk
import logging 
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from tqdm import tqdm
import pandas as pd
import ast
import csv
import re 
import numpy as np
import os
import pdb

data_path = './yelp_dataset/'
file_name = 'yelp_academic_dataset_review.json'
data_export = './yelp_dataset/data_split/'

def loadData(file_path):
    print('preprocessing data by combine the mulit-line comments...')
    label_list = []
    text_list = []
    with open(file_path, 'r') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split('\t\t\t')
            try:
                label = int(line[0])
                label_list.append(label)
                text = line[1]
                text_list.append(text)
            except (IndexError, ValueError):
                last_text = text_list.pop(-1)
                last_text += line[0] # for here using line[0] since the unlabeled text has no label
                text_list.append(last_text)
    return label_list, text_list

def reviewEmbedding(model_path, model_word2vec_output, model_postag_output, label_list, text_list, window_size=20, embed_dim=100):
    # checking word to vecotr embedding
    if not os.path.isfile(model_path + model_word2vec_output):
        trainWord2VecEmbedding(model_path, model_word2vec_output, text_list, embed_dim)
    # checking POS tag embedding
    if not os.path.isfile(model_path + model_postag_output):
        trainWordPOSTag(model_path, model_postag_output, text_list) 
    model = Word2Vec.load(model_path + model_word2vec_output)
    with open(model_path + model_postag_output, 'rb') as f:
        postag_list = pickle.load(f)

    print('start review embedding')
    train_feature = []
    train_label = []
    for sentences in text_list:
        label = label_list.pop(0)
        sentences = sent_tokenize(sentences)
        sentence_tmp = []
        sentence_vector = []
        for sen in sentences:
            sentence_tmp += word_tokenize(sen)
        pos_vector = [postag_list.index(x[1]) for x in  nltk.pos_tag(sentence_tmp)]
        # getting each word embedding vector in a review sample
        for i in range(len(sentence_tmp)):
            vector = model.wv[sentence_tmp[i]]
            sentence_vector.append(vector.tolist() + [pos_vector[i], i])
        # getting each word postag in a review sample
        for s in range(len(sentence_vector)-19):
            label_emb = [0] * 5
            train_sample = np.array((sentence_vector[s:s+window_size])).reshape(window_size,len(sentence_vector[1]),1) 
            label_emb[label-1] = 1
            train_label.append(label_emb)
            train_feature.append(train_sample)
    return train_label, train_feature

def trainWord2VecEmbedding(mdoel_path, model_temp_path, model_output, text_list, embed_dim=100):
    batch_size = 1000000
    model = Word2Vec(size=embed_dim, min_count=1, workers=4, sample=1e-5, window=4)
    #initialized batch 
    batch_text_list = text_list[0:batch_size]
    sentence_list = []
    for sentences in tqdm(batch_text_list):
        sentences = sent_tokenize(sentences)
        for sen in sentences:
            sentence_list.append(word_tokenize(sen))
    print('word2vec training samples have been loaded !!!')
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    model.build_vocab(sentence_list)
    model.train(sentence_list,total_examples=batch_size,epochs=5)
    model.save(model_temp_path + model_output)
    for i in range(1, len(text_list)//batch_size - 1):  # owing to the data size, we train word2vec model batch by batch
        print('this is the epoch %d' %i)
        start = i * batch_size
        end = start + batch_size
        batch_text_list = text_list[start:end]
        sentence_list = []
        for sentences in tqdm(batch_text_list):
            sentences = sent_tokenize(sentences)
            for sen in sentences:
                sentence_list.append(word_tokenize(sen))
        print('word2vec training samples have been loaded !!!')
        logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
        model.build_vocab(sentence_list, update=True)
        model.train(sentence_list,total_examples=batch_size,epochs=5)
        #model = Word2Vec(sentence_list, size=embed_dim, window=5, min_count=0, workers=4) # will be tuned to imporve the embedding performance
    # the last part sine len(text_list) cannot be divided by batch_size
    batch_text_list = text_list[end:]
    sentence_list = []
    for sentences in tqdm(batch_text_list):
        sentences = sent_tokenize(sentences)
        for sen in sentences:
            sentence_list.append(word_tokenize(sen))
    print('word2vec training samples have been loaded !!!')
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    model.build_vocab(sentence_list)
    model.train(sentence_list,total_examples=len(batch_text_list),epochs=5)

    model.save(mdoel_path + model_output)
    print('word to vector embedding complete')

def trainWordPOSTag(model_path, model_output, text_list):
    sentence_tag_list = []
    sentence_tag_collecter = []
    for sentences in text_list:
        sentences = sent_tokenize(sentences)
        sentence_tmp = []
        for sen in sentences:
            sen = word_tokenize(sen)
            sentence_tmp += nltk.pos_tag(sen)
        sentence_tag = [x[1] for x in sentence_tmp]
        sentence_tag_collecter += sentence_tag
        sentence_tag_list.append(sentence_tag)
    tag_idx_converter = set(sentence_tag_collecter)
    pickle.dump(list(tag_idx_converter), open(model_path + model_output, 'wb'))
    print('POS tag embedding complete')


