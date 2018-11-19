from preprocess import *
from data_loader import *
from ACNNModel import *
import pdb
import os 


data_path = './yelp_dataset/'
input_file_name = 'yelp_academic_dataset_review.json'
output_file_name = 'yelp_academic_dataset_review.csv'
sample_file_name = 'sample_yelp_academic_dataset_review_1000.csv'
data_export = './yelp_dataset/data_split/'
train_data_file_name = 'train/train.train'

model_path = './model/'
model_word2vec_output='embedding/word2vec.model'
model_postag_output = 'embedding/POStag.pkl'

data_convert_csv(data_path, input_file_name, output_file_name)
#data_sample_csv(data_path, input_file_name, sample_file_name)
print('I can convert CSV now!!!')
split_feature_label(data_path, output_file_name, data_export, train_data_file_name)

label_list, text_list = loadData(data_export + train_data_file_name)
print('I can return text_list')
# for here, textlist need to do some preprocessing
train_label, train_feature = reviewEmbedding(model_path, model_word2vec_output, model_postag_output, label_list, text_list, window_size=20, embed_dim=100)
print(np.shape(train_label))
print(np.shape(train_feature))

buildModel(train_feature, train_label)