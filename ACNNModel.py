import tensorflow as tf 
import numpy as np

def buildModel(train_feature, train_label):
    learning_rate = 0.1
    #input data...
    train_feature = tf.placeholder(tf.float32, shape=[None, 20, 100], name='feature')
    train_label = tf.placeholder(tf.int32, shape=[None,1], name='label')
    weights = _intialize_weights()
    #build model...
    filter_size = [1, 3, 5]
    embedding_dim = 100
    #initializer and regularizer will be defined later
    unigram = tf.layers.conv2d(inputs=train_feature, filters=100, kernel_size=(filter_size[0], embedding_dim), padding='same', activation=tf.nn.relu)
    bigram = tf.layers.conv2d(inputs=train_feature, filters=100, kernel_size=(filter_size[1], embedding_dim), padding='same', activation=tf.nn.relu)
    trigram = tf.layers.conv2d(inputs=train_feature, filters=100, kernel_size=(filter_size[2], embedding_dim), padding='same', activation=tf.nn.relu)
    #attentive layer...
    for i in range(tf.shape(train_feature)[1]):
        unigram_vec = unigram[:,i,:]
        bigram_vec = bigram[:,i,:]
        trigram_vec = trigram[:,i,:]
        a_1 = tf.add_n(tf.multiply(unigram_vec,weights['attention_W']), weights['attention_b'])
        a_2 = tf.add_n(tf.multiply(bigram_vec,weights['attention_W']), weights['attention_b'])
        a_3 = tf.add_n(tf.multiply(trigram_vec,weights['attention_W']), weights['attention_b'])
        #softmax for attention socre
        attention_exp = [tf.exp(a_1), tf.exp(a_2), tf.exp(a_3)]
        attention_sum = tf.reduce_sum(attention_exp)
        attention_score = tf.div(attention_exp,attention_sum) # not sure tf operation support list type
        attention_feature = tf.multiply(attention_score[0],unigram_vec) + tf.multiply(attention_score[1],bigram_vec) + tf.multiply(attention_score[2],trigram_vec)
        tf.assign(weights['attention_feature'][:,i,:], attention_feature)
    flatten_layer = tf.reshape(weights['attention_feature'], [-1])
    prediction = tf.add_n(tf.reduce_sum(tf.multiply(flatten_layer, weights['predict_W'])), weights['prediciton_b'])
    #loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=train_label))
    #optimizer function
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #trainModel(train_feature,train_label):
    total_loss =0
    train_epoch = 10
    batch_size = 25
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(train_epoch):
        for i in range(len(train_feature)//batch_size - 1):
            start = i * 256
            end = start + 256
            train_feature_oneBatch = np.array(train_feature[start:end])
            train_label_oneBatch = np.array(train_label[start:end])
            feed_dict = {
                train_feature : train_feature_oneBatch,
                train_label : train_label_oneBatch
            }
            _, loss = sess.run([train_op, loss], feed_dict=feed_dict)
            total_loss += loss
        print('in epoch %d, the loss is:' %e, total_loss/(len(train_feature)//batch_size*batch_size))

def _intialize_weights():
    weights = {}
    sample_len = 20
    embedding_dim = 100
    weights['attention_W'] = tf.Variable(np.random.normal(loc=0, scale=np.sqrt(2.0/embedding_dim), size=(embedding_dim,1)), dtype=np.float32, name='attention_W')
    weights['attention_b'] = tf.Variable(tf.constant(1.0), name='attention_b') 
    weights['attention_feature'] = tf.zeros(shape=[None, sample_len, embedding_dim])
    weights['predict_W'] = tf.Variable(np.random.normal(loc=0.0, scale=2.0/(embedding_dim * sample_len)), dtype=np.float32, name='prediction_name')
    weights['predict_b'] = tf.Variable(tf.constant(0.0), name='prediction_bias')
    return weights

