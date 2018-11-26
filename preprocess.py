import json
from tqdm import tqdm
import pandas as pd
import ast
import csv
import re 
import numpy as np
import os
import pdb

def data_convert_csv(data_path, input_file_name, output_file_name):
    if os.path.isfile(data_path + output_file_name):
        print(output_file_name + ' exists')
    else:
        print('converting json file into csv file')
        with open(data_path + input_file_name, encoding="utf8") as json_file:
            data = json_file.readlines()  # list of string  
        # store the data in csv file 
        with open(data_path+ output_file_name, 'w', encoding='utf-8') as f:
            # write the heading 
            review_dict = ast.literal_eval(data[0])
            heading = review_dict.keys()
            writer = csv.writer(f)
            writer.writerow(list(heading))
            for index in tqdm(range(len(data))):
                #if index > 300000:
                #    break
                review_dict = json.loads(data[index])
                writer = csv.writer(f)
                writer.writerow(review_dict.values())
        print('converting complete !!!')

def data_sample_csv(data_path, input_file_name, sample_file_name):
    if os.path.isfile(data_path+ sample_file_name):
        print(sample_file_name + ' exists')
    else:
        print('converting json file into csv file')
        with open(data_path + input_file_name, encoding="utf8") as json_file:
            data = json_file.readlines()  # list of string 
        print("finished json file load") 
        # store the data in csv file 
        with open(data_path+ sample_file_name, 'w', encoding="utf8") as f:
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

def split_feature_label(data_path, sample_file_name, data_export, test_size):
    # if os.path.isfile(data_export + train_data_file_name) and os.path.isfile(data_export + test_data_file_name):
    #     print(train_data_file_name + ' exists !!!')
    #     print(test_data_file_name + ' exists !!!')
    if os.path.isfile(data_export + 'all_data.data'):
        print('raw data is exist !!!')
    else:
        df = pd.read_csv(data_path + sample_file_name, low_memory=True) 
        print('finished load data')
        num_samples = len(df['stars'])
        with open(data_export + 'all_data.data', 'w', encoding='utf8') as f:
            for i in tqdm(range(num_samples)):
                label = str(df['stars'][i])
                raw_text = df['text'][i]
                try:
                    f.write(label + '\t\t\t' + raw_text + '\n')
                except TypeError:
                    continue
        print('file wirtten has completed !!!')

    label_list, text_list = loadData(data_export + 'all_data.data')
    print('----------------')
    print(len(label_list))
    print(len(text_list))
    num_samples = len(label_list)
    idx_list = [k for k in range(num_samples)]
    np.random.shuffle(idx_list)
    cut_idx = int(num_samples * (1 - test_size))
    train_text_list = []
    train_label_list = []
    test_text_list = []
    test_label_list = []
    for i in idx_list[:cut_idx]:
        train_text_list.append(text_list[i])
        train_label_list.append(label_list[i])
    for j in idx_list[cut_idx:]:
        test_text_list.append(text_list[j])
        test_label_list.append(label_list[j])

    return train_text_list, train_label_list, test_text_list, test_label_list

def loadData(file_path):
    print('preprocessing data by combine the mulit-line comments...')
    label_list = []
    text_list = []
    with open(file_path, 'r',encoding='utf-8') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split('\t\t\t')
            if len(line) == 1 and len(line[0]) <= 2:
                continue
            try:
                label = int(line[0])
                label_list.append(label)
                text = line[1]
                text_list.append(text)
            except (IndexError, ValueError):
                last_text = text_list.pop()
                last_text += line[0] # for here using line[0] since the unlabeled text has no label
                text_list.append(last_text)
            if len(text_list) != len(label_list):
                label_list.pop()
        return label_list, text_list