import json
from tqdm import tqdm
import pandas as pd
import ast
import csv
import re 
import numpy as np

data_path = './yelp_dataset/'
file_name = 'yelp_academic_dataset_review.json'
data_export = './yelp_dataset/data_split/'

def data_convert_csv():
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

def data_sample_csv():
    print('converting json file into csv file')
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
    print('converting complete !!!')

def split_feature_label():
    df = pd.read_csv(data_path + 'sample_yelp_academic_dataset_review.csv',low_memory=False)
    print('finished load data')

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

    
    