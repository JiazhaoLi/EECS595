import tensorflow as tf 
import numpy as np

def buildModel(train_feature, train_label):
    
    num_filter = 100
    learning_rate = 0.1
    embedding_dim = 102
    seq_len = 20
    #input data...
    batch_size = 64
    train_feature_tensor = tf.placeholder(tf.float32, shape=[batch_size, 20, 102, 1], name='feature')
    train_label_tensor = tf.placeholder(tf.int32, shape=[batch_size,5], name='label')

    weights = _intialize_weights(batch_size,num_filter,seq_len,embedding_dim)
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
    
    #softmax for attention socre
    attention_exp = [tf.exp(a_1), tf.exp(a_2), tf.exp(a_3)]
    attention_sum = tf.reduce_sum(attention_exp)
    attention_score = tf.div(attention_exp, attention_sum) # not sure tf operation support list type
    attention_feature = tf.multiply(attention_score[0], unigram_vec) + tf.multiply(attention_score[1], bigram_vec) + tf.multiply(attention_score[2],trigram_vec)
    concat_layer = attention_feature
    
    for i in range(1,seq_len):
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
    prediction = tf.reshape(tf.add(tf.matmul(weights['predict_W'],flatten_layer), weights['predict_b']),shape=(-1,5))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=train_label_tensor))
    #optimizer function
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    total_loss =0
    train_epoch = 10
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(train_epoch):
        for i in range(len(train_feature)//batch_size - 1):
            start = i * batch_size
            end = start + batch_size
            train_feature_oneBatch = np.array(train_feature[start:end])
            train_label_oneBatch = np.array(train_label[start:end])

            feed_dict = {
                train_feature_tensor : train_feature_oneBatch,
                train_label_tensor : train_label_oneBatch
            }
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
            total_loss += loss_val
            print('in epoch %d, the loss is:' %e, total_loss/(len(train_feature)//batch_size*batch_size))


def _intialize_weights(batch_size,num_filter,seq_len,embedding_dim):
    weights = {}
    weights['attention_W'] = tf.Variable(np.random.normal(loc=0, scale=np.sqrt(2.0/num_filter), size=(num_filter,1)), dtype=np.float32, name='attention_W')
    weights['attention_b'] = tf.Variable(tf.constant(1.0), name='attention_b') 
    weights['predict_W'] = tf.Variable(np.random.normal(loc=0.0, scale=2.0/(num_filter * seq_len), size=(5,num_filter * seq_len)), dtype=np.float32, name='prediction_name')
    weights['predict_b'] = tf.Variable(tf.constant(0.0), name='prediction_bias')
    return weights

