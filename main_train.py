from preprocess import *
from data_loader import *
from ACNNModel import *
import pdb
import os 


data_path = './yelp_dataset/'
input_file_name = 'yelp_academic_dataset_review.json'
output_file_name = 'yelp_academic_dataset_review.csv'
#sample_file_name = 'sample_yelp_academic_dataset_review_1000.csv'
data_export = './yelp_dataset/data_split/'
train_data_file_name = 'train/train.train'
test_data_file_name = 'test/test.test'


model_path = './model/'
model_word2vec_output='embedding/word2vec.model'
model_postag_output = 'embedding/POStag.pkl'

data_convert_csv(data_path, input_file_name, output_file_name)
print('main: data convert complete !!!')
train_text_list, train_label_list, test_text_list, test_label_list = split_feature_label(data_path, output_file_name, data_export, test_size=0.3)
print('main: get text list')
print('main: train_text_size: ', len(train_label_list), len(train_text_list))
print('main: test_text_size: ', len(test_label_list), len(test_text_list))
'''
# label_list, text_list = loadData(data_export + 'all_data.data')
trainWord2VecEmbedding(model_path, model_word2vec_output, train_text_list + test_text_list, embed_dim=100)
trainWordPOSTag(model_path, model_postag_output, train_text_list + test_text_list)
'''
print('-----------------------------------------------------------------------------')

# for here, textlist need to do some preprocessing
# storage the embedding (training feature and label into sub_files with the number of train_file_len......)
train_file_len = 200
if not os.listdir(data_export + 'train/emb_feature/'):
    for i in range(train_file_len):
        train_label, train_feature, train_label_list, train_text_list = reviewEmbedding(model_path, model_word2vec_output, model_postag_output, train_label_list, train_text_list, window_size=30, embed_dim=100)
        pickle.dump(train_label, open(data_export + 'train/emb_label/train_label_emb_%d' %i + '.train', 'wb'))
        pickle.dump(train_feature, open(data_export + 'train/emb_feature/train_feature_emb_%d' %i + '.train', 'wb'))
        #print('end')
idx = 0
test_file_len = 30
if not os.listdir(data_export + 'test/emb_feature/'):
    for i in range(test_file_len):
        train_label, train_feature, test_label_list, test_text_list = reviewEmbedding(model_path, model_word2vec_output, model_postag_output, test_label_list, test_text_list, window_size=30, embed_dim=100)
        pickle.dump(train_label, open(data_export + 'test/emb_label/test_label_emb_%d' %i + '.test', 'wb'))
        pickle.dump(train_feature, open(data_export + 'test/emb_feature/test_feature_emb_%d' %i + '.test', 'wb'))

train_feature_path = data_export + 'train/emb_feature/'
train_label_path = data_export + 'train/emb_label/'
test_feature_path = data_export + 'test/emb_feature/'
test_label_path = data_export + 'test/emb_label/'
buildModel(train_feature_path, train_label_path, test_feature_path, test_label_path)