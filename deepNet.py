import tensorflow as tf
#Part -1 : Computational Graph and building a Neural Network
'''
input >> weight >> hidden layer 1(activation function) >> weight >>
hidden l2(activation function) >> weights >> output layer

compare output to intended output >> cost function(cross entropy)
optimisation function(optimizer) >> minimize cost(AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
.
.
.
9 = [0,0,0,0,0,0,0,0,0,1]
'''

nhl1 = 500
nhl2 = 500
nhl3 = 500

n_classes = 10
batch_size = 100

#height x width
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):


    hidden_layer_1 = {'weights': tf.Variable(tf.truncated_normal([784, nhl1], stddev=0.1)),
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
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss+=c
            print('Epoch',epoch,'completed out of',n_epoch,'loss: ',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

training_neural_network(x)

