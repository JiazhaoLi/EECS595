from gensim.models import word2vec
import nltk
import logging 
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
import os


data_path = './yelp_dataset/'
file_name = 'yelp_academic_dataset_review.json'
data_export = './yelp_dataset/data_split/'

def loadData(file_path):
    print('preprocessing data by combine the mulit-line comments...')
    label_list = []
    text_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.strip().split('\t\t\t')
            if len(line) == 1 and len(line[0]) <= 2:
                continue
            try:
                label = int(line[0])
                label_list.append(label)
                text = line[1]
                text_list.append(text)
            except (IndexError, ValueError):
                last_text = text_list.pop(-1)
                last_text += line[0] # for here using line[0] since the unlabeled text has no label
                text_list.append(last_text)
    print('loading data complete !!!')
    return label_list, text_list

def reviewEmbedding(model_path, model_word2vec_output, model_postag_output, label_list, text_list, window_size=30, embed_dim=100, valid_size=(30, 200), step_size=3):
    # checking word to vecotr embedding
    print('embedding review ... ')
    if not os.path.isfile(model_path + model_word2vec_output):
        trainWord2VecEmbedding(model_path, model_word2vec_output, text_list, embed_dim)
    else:
        print('word2vec model already exist !!!')
    # checking POS tag embedding
    if not os.path.isfile(model_path + model_postag_output):
        trainWordPOSTag(model_path, model_postag_output, text_list)
    else:
        print('psotag model already exist !!!')
    print('loading word2vec model ... ')
    model = word2vec.Word2Vec.load(model_path + model_word2vec_output)
    print('word2vec model load complete !!!')
    print('loading postag mdoel ... ')
    postag_dict = {}
    cnt = 0
    with open(model_path + model_postag_output, 'rb') as f:
        postag_list = pickle.load(f)
        for item in postag_list:
            if item not in postag_dict:
                postag_dict[item] = cnt
                cnt += 1
    print('load postag model complete !!!')
    print('start review embedding')
    train_feature = []
    train_label = []
    sample_cnt = 0
    while len(text_list) > 0:
        if sample_cnt == 100:
            break
        sentences = text_list.pop()
        label = label_list.pop()
        sentences = sent_tokenize(sentences)
        sentence_tmp = []
        sentence_vector = []
        pos_vector = []
        for sen in sentences:
            sentence_tmp += word_tokenize(sen)
        # getting each word embedding vector in a review sample
        if valid_size[0] <= len(sentence_tmp) <= valid_size[1]:
            sample_cnt += 1
            pos_vector = [postag_dict[x[1]] for x in nltk.pos_tag(sentence_tmp)]
            for i in range(len(sentence_tmp)):
                vector = model.wv[sentence_tmp[i]]
                sentence_vector.append(vector.tolist() + [pos_vector[i], i])
            # getting each word postag in a review sample
            for s in range(0, len(sentence_vector)-window_size-1, step_size):
                label_emb = [0] * 3
                train_sample = np.array((sentence_vector[s: s+window_size])).reshape(window_size, len(sentence_vector[0]), 1)
                if label <= 3:
                    label_emb[0] = 1
                elif label == 4:
                    label_emb[1] = 1
                else:
                    label_emb[2] = 1
                train_label.append(label_emb)
                train_feature.append(train_sample)

    print('complete file writting !!!')
    return train_label, train_feature, label_list, text_list


def trainWord2VecEmbedding(model_path, model_output, text_list, embed_dim=100):
    print('i am in word2vec embedding ...')
    sentence_list = []
    for sentences in text_list:
        sentences = sent_tokenize(sentences)
        for sen in sentences:
            sentence_list.append(word_tokenize(sen))
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentence_list, size=embed_dim, window=5, min_count=0, workers=4) # will be tuned to imporve the embedding performance
    model.save(model_path + model_output)
    print('word to vector embedding complete')

def trainWordPOSTag(model_path, model_output, text_list):
    print('embedding postag ... ')
    sentence_tag_list = []
    sentence_tag_collecter = []
    sentence_tag_dict = {}
    for sentences in tqdm(text_list):
        #print('sentence_level !!!')
        sentences = sent_tokenize(sentences)
        sentence_tmp = []
        for sen in sentences:
            #print('word level !!!')
            sen = word_tokenize(sen)
            sentence_tmp += nltk.pos_tag(sen)
            #print(sentence_tmp)
        sentence_tag = [x[1] for x in sentence_tmp]
        sentence_tag_collecter += sentence_tag
        sentence_tag_list.append(sentence_tag)

    sentence_tag_collecter = set(sentence_tag_collecter)
    sentence_tag_collecter = list(sentence_tag_collecter)

    pickle.dump(sentence_tag_collecter, open(model_path + model_output, 'wb'))
    print('POS tag embedding complete')


