from gensim.models import word2vec
import nltk
import logging 

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

def trainWordEmbedding(label_list, text_list, window_size = 20):
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO) 
    model = word2vec.Word2Vec(text_list, size=100, window=5, min_count=0, workers=4)
    train_feature = []
    train_label = []
    for i in range(len(text_list)):
        sentence = text_list[i]
        label = label_list[i]
        sentence_vector = []

        for item in sentence:
            vector = model.wv[item]
            sentence_vector.append(vector)

        for s in range(len(sentence_vector)-19):
            train_sample = sentence_vector[s:s+20]
            train_label.append(label)
            train_feature.append(train_sample)
    return train_label, train_feature
        

