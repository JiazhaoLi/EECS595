import tensorflow as tf 
import numpy as np
import pickle
import os

def buildModel(train_feature_path, train_label_path, test_feature_path, test_label_path):
    num_filter = 100
    learning_rate = 0.002
    embedding_dim = 102
    seq_len = 30 # same as window_size in reviewembedding of data_loader.py
    #input data...
    batch_size = 256
    train_feature_tensor = tf.placeholder(tf.float32, shape=[batch_size, seq_len, embedding_dim, 1], name='feature')
    train_label_tensor = tf.placeholder(tf.float32, shape=[batch_size, 3], name='label')
    #test_feature_tensor = tf.placeholder(tf.float32, shape=[batch_size, seq_len, embedding_dim, 1], name='test_feature')
    #test_label_tensor = tf.placeholder(tf.float32, shape=[batch_size, 3], name='test_label')
    weights = _intialize_weights(batch_size, num_filter, seq_len, embedding_dim)
    #build model...
    filter_size = [1, 3, 5]
    
    #initializer and regularizer will be defined later
    unigram = tf.layers.conv2d(inputs=train_feature_tensor, filters=num_filter, kernel_size=(filter_size[0], embedding_dim), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
    bigram = tf.layers.conv2d(inputs=train_feature_tensor, filters=num_filter, kernel_size=(filter_size[1], embedding_dim), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
    trigram = tf.layers.conv2d(inputs=train_feature_tensor, filters=num_filter, kernel_size=(filter_size[2], embedding_dim), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
   
    unigram_f = tf.reshape(unigram[:,:,num_filter//2,:],shape=[-1,seq_len,num_filter])
    bigram_f = tf.reshape(bigram[:,:,num_filter//2,:],shape=[-1,seq_len,num_filter])
    trigram_f = tf.reshape(trigram[:,:,num_filter//2,:],shape=[-1,seq_len,num_filter])

    unigram_vec = unigram_f[:, 0, :]
    bigram_vec = bigram_f[:, 0, :]
    trigram_vec = trigram_f[:, 0, :]
    
    a_1 = tf.add(tf.matmul(unigram_vec, weights['attention_W']), weights['attention_b'])
    a_2 = tf.add(tf.matmul(bigram_vec,weights['attention_W']), weights['attention_b'])
    a_3 = tf.add(tf.matmul(trigram_vec,weights['attention_W']), weights['attention_b'])
    
    #softmax for attention score
    attention_exp = [tf.exp(a_1), tf.exp(a_2), tf.exp(a_3)]
    attention_sum = tf.reduce_sum(attention_exp)
    attention_score = tf.div(attention_exp, attention_sum) # not sure tf operation support list type
    attention_feature = tf.multiply(attention_score[0], unigram_vec) + tf.multiply(attention_score[1], bigram_vec) + tf.multiply(attention_score[2],trigram_vec)
    concat_layer = attention_feature
    
    for i in range(1, seq_len):
        unigram_vec = unigram_f[:, i, :]
        bigram_vec = bigram_f[:, i, :]
        trigram_vec = trigram_f[:, i, :]
        a_1 = tf.add(tf.matmul(unigram_vec, weights['attention_W']), weights['attention_b'])
        a_2 = tf.add(tf.matmul(bigram_vec,weights['attention_W'] ), weights['attention_b'])
        a_3 = tf.add(tf.matmul(trigram_vec,weights['attention_W'] ), weights['attention_b'])
    
        attention_exp = [tf.exp(a_1), tf.exp(a_2), tf.exp(a_3)]
        attention_sum = tf.reduce_sum(attention_exp)
        attention_score = tf.div(attention_exp, attention_sum) # not sure tf operation support list type
        attention_feature = tf.multiply(unigram_vec,attention_score[0]) + tf.multiply(bigram_vec,attention_score[1]) + tf.multiply(trigram_vec,attention_score[2])
        concat_layer = tf.concat([concat_layer, attention_feature], 0)

    flatten_layer = tf.reshape(concat_layer,shape = (seq_len * num_filter,-1))
    prediction = tf.reshape(tf.add(tf.matmul(weights['predict_W'],flatten_layer), weights['predict_b']),shape=(-1,3))
    # prediction = tf.nn.softmax(prediction) # adding prediciton here
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=train_label_tensor))
    loss = tf.nn.l2_loss(tf.math.subtract(prediction,train_label_tensor))/batch_size
    #optimizer function
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # train part
    train_feature_file_list = os.listdir(train_feature_path)
    test_feature_file_list = os.listdir(test_feature_path)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 
    all_epoch = 41
    for e in range(all_epoch):
        total_loss = 0
        train_label_list = []
        prediction_list = []
        all_correct_cnt = 0
        for i in range(len(train_feature_file_list)):
            oneFile_loss = 0
            oneFile_train_label_list = []
            oneFile_prediction_list = []
            correct_cnt = 0
            feature_file_name = train_feature_path + 'train_feature_emb_%d' %i + '.train'
            label_file_name = train_label_path + 'train_label_emb_%d' %i + '.train'
            with open(feature_file_name, 'rb') as f:
                train_feature = pickle.load(f)
            with open(label_file_name, 'rb') as l:
                train_label = pickle.load(l)
            for b in range(len(train_feature)//batch_size - 1):
                start = b * batch_size
                end = start + batch_size
                train_feature_oneBatch = np.array(train_feature[start:end])
                train_label_oneBatch = np.array(train_label[start:end])
                feed_dict = {
                    train_feature_tensor: train_feature_oneBatch,
                    train_label_tensor: train_label_oneBatch
                }
                _, loss_val, train_label_batch, prediction_batch = sess.run([train_op, loss, train_label_tensor, prediction], feed_dict=feed_dict)
                # print(zip(train_label_list, prediction))
                oneFile_loss += loss_val
                oneFile_train_label_list += train_label_batch.tolist()
                oneFile_prediction_list += prediction_batch.tolist()
            average_loss = oneFile_loss/float((len(train_feature)//batch_size))
            print('in file %d, the number of train sample is:' %i, len(oneFile_train_label_list))
            print('in file %d, the train loss is:' %i, average_loss)
            for a in range(len(oneFile_train_label_list)):
                if oneFile_train_label_list[a].index(max(oneFile_train_label_list[a])) == oneFile_prediction_list[a].index(max(oneFile_prediction_list[a])):
                    correct_cnt += 1
            print('in file %d, the train accuracy is:'%i, correct_cnt/ len(oneFile_train_label_list))
            print('-------------------------------------------------------')
            print('-------------------------------------------------------')
            total_loss += average_loss
            train_label_list += oneFile_train_label_list 
            prediction_list += oneFile_prediction_list
            all_correct_cnt += correct_cnt
        print('in epoch %d, the train loss is:' % e, total_loss / len(train_feature_file_list))
        print('in epoch %d, the train accuracy is:'%e, all_correct_cnt/ len(train_label_list))
        print('-------------------------------------------------------')
        print('-------------------------------------------------------')

        if e % 5 == 0:
            total_loss = 0
            train_label_list = []
            prediction_list = []
            all_correct_cnt = 0
            for i in range(len(test_feature_file_list)):
                oneFile_loss = 0
                oneFile_train_label_list = []
                oneFile_prediction_list = []
                correct_cnt = 0
                feature_file_name = test_feature_path + 'test_feature_emb_%d' %i + '.test'
                label_file_name = test_label_path + 'test_label_emb_%d' %i + '.test'
                with open(feature_file_name, 'rb') as f:
                    train_feature = pickle.load(f)
                with open(label_file_name, 'rb') as l:
                    train_label = pickle.load(l)
                for b in range(len(train_feature)//batch_size - 1):
                    start = b * batch_size
                    end = start + batch_size
                    train_feature_oneBatch = np.array(train_feature[start:end])
                    train_label_oneBatch = np.array(train_label[start:end])
                    feed_dict = {
                        train_feature_tensor: train_feature_oneBatch,
                        train_label_tensor: train_label_oneBatch
                    }
                    _, loss_val, train_label_batch, prediction_batch = sess.run([train_op, loss, train_label_tensor, prediction], feed_dict=feed_dict)
                    oneFile_loss += loss_val
                    oneFile_train_label_list += train_label_batch.tolist()
                    oneFile_prediction_list += prediction_batch.tolist()
                average_loss = oneFile_loss/float((len(train_feature)//batch_size))
                print('in file %d, the number of test sample is:' % i, len(oneFile_train_label_list))
                print('in file %d, the test loss is:' %i, average_loss)
                for a in range(len(oneFile_train_label_list)):
                    if oneFile_train_label_list[a].index(max(oneFile_train_label_list[a])) == oneFile_prediction_list[a].index(max(oneFile_prediction_list[a])):
                        correct_cnt += 1
                print('in file %d, the test accuracy is:'%i, correct_cnt/ len(oneFile_train_label_list))
                print('-------------------------------------------------------')
                print('-------------------------------------------------------')
                total_loss += average_loss
                train_label_list += oneFile_train_label_list
                prediction_list += oneFile_prediction_list
                all_correct_cnt += correct_cnt
            print('the test loss is:', total_loss / len(train_feature_file_list))
            print('the test accuracy is:', all_correct_cnt/ len(train_label_list))

def _intialize_weights(batch_size,num_filter,seq_len,embedding_dim):
    weights = {}
    weights['attention_W'] = tf.Variable(np.random.normal(loc=0, scale=np.sqrt(2.0 / num_filter), size=(num_filter, 1)), dtype=np.float32, name='attention_W')
    weights['attention_b'] = tf.Variable(tf.constant(1.0), name='attention_b') 
    weights['predict_W'] = tf.Variable(np.random.normal(loc=0.0, scale=2.0 / (num_filter * seq_len), size=(3, num_filter * seq_len)), dtype=np.float32, name='prediction_name')
    weights['predict_b'] = tf.Variable(tf.constant(0.0), name='prediction_bias')
    return weights

