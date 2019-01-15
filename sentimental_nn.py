import tensorflow as tf
from sentimental import create_feature_sets_and_labels
import numpy as np
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

nhl1 = 1500
nhl2 = 1500
nhl3 = 1500

n_classes = 2
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):


    hidden_layer_1 = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), nhl1], stddev=0.1)),
    'biases': tf.Variable(tf.constant(0.1,shape=[nhl1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.truncated_normal([nhl1, nhl2], stddev=0.1)),
    'biases': tf.Variable(tf.constant(0.1,shape=[nhl2]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.truncated_normal([nhl2, nhl3], stddev=0.1)),
    'biases': tf.Variable(tf.constant(0.1,shape=[nhl3]))}
    output_layer = {'weights': tf.Variable(tf.truncated_normal([nhl3, n_classes], stddev=0.1)),
    'biases': tf.Variable(tf.constant(0.1,shape=[n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']

    return output

#Part -2 : Training a Neural Network
def training_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epoch = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            epoch_loss = 0
            i=0
            while i<len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss+=c
                i+=batch_size
            print('Epoch',epoch,'completed out of',n_epoch,'loss: ',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

training_neural_network(x)
