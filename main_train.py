from preprocess import *
from data_loader import *
from ACNNModel import *
import pdb

#data_convert_csv()
#data_sample_csv()
#split_feature_label()
label_list, text_list  = loadData('./yelp_dataset/data_split/train/sample_train.train')
train_label, train_feature = trainWordEmbedding(label_list, text_list, window_size = 20)
buildModel(train_feature, train_label)