data_path = '../data/'
file_name = 'yelp_academic_dataset_review.json'
data_export = '../data/data_split/'

import json
from tqdm import tqdm
import pandas as pd
import ast
import csv 
import numpy as np

def data_convert_csv():
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

def data_sample_csv():
    with open(data_path + file_name) as json_file:      
        data = json_file.readlines()  # list of string 
    print("finished json file load") 
    # store the data in csv file 
    with open(data_path+ 'sample_yelp_academic_dataset_review.csv', 'w') as f:
        # write the heading 
        review_dict = ast.literal_eval(data[0])
        heading = review_dict.keys()
        writer = csv.writer(f)
        writer.writerow(list(heading))
        for index in tqdm(range(len(data[:100000]))):
            review_dict = json.loads(data[index])
            writer = csv.writer(f)
            writer.writerow(review_dict.values())
    
def split_feature_label():
    df = pd.read_csv(data_path + 'sample_yelp_academic_dataset_review.csv',low_memory=False)
    print('finished load data')

    print(len(df['stars']))

    with open(data_export + 'train/sample_train.csv','w') as f:
        for i in df['text']:
            f.write(i + '\n')
    
    with open(data_export + 'test/sample_test.csv','w') as f:
        for i in df['stars']:
            f.write(str(i))
            f.write('\n')

    
    