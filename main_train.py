
from data_loader import *
from ACNNModel import *
import pdb
import os 


data_path = './yelp_dataset/'
file_name = 'yelp_academic_dataset_review.json'
data_export = './yelp_dataset/data_split/'

if os.path.isfile(data_path+ 'yelp_academic_dataset_review.csv'):
    print('yelp_academic_data_review.csv    exist:')
else:
    data_convert_csv(data_path, file_name)

if os.path.isfile(data_path+ 'sample_yelp_academic_dataset_review_1000.csv'):
    print('sample_yelp_academic_dataset_review_1000.csv    exist:')
else:
    data_sample_csv(data_path, file_name)

if os.path.isfile(data_export + 'train/sample_train.train'):
    print('train/sample_train_1000.train   exist:')
else:
    split_feature_label(data_path, data_export)


label_list, text_list = loadData('./yelp_dataset/data_split/train/sample_train.train')
# for here, textlist need to do some preprocessing


train_label, train_feature = trainWordEmbedding(data_path, label_list, text_list, window_size = 20)
print(np.shape(train_label))
print(np.shape(train_feature))

buildModel(train_feature, train_label)