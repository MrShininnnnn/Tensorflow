DATA_FILE = "Data/"

import numpy as np
import tensorflow as tf
from mnist import MNIST
from sklearn import preprocessing

print("Data preprocessing...")
mndata = MNIST(DATA_FILE)
train_sample, train_label = mndata.load_training()
test_sample, test_label = mndata.load_testing()
size_train = len(train_sample)
size_test = len(test_sample)
len_sample = len(train_sample[0])
train_sample = preprocessing.normalize(train_sample, norm = 'l2')
test_sample = preprocessing.normalize(test_sample, norm = 'l2')

print("OneHotEncoding...")

label_range = [[i] for i in np.arange(10)]
enc = preprocessing.OneHotEncoder()
enc.fit(label_range) 
oh_train_label, oh_test_label = [], []

for i in range(size_train):
	oh_train_label.append(enc.transform(train_label[i]).toarray()[0])
for i in range(size_test):
	oh_test_label.append(enc.transform(test_label[i]).toarray()[0])

print("Training...")

x = tf.placeholder(tf.float32, [None, len_sample])
W = tf.Variable(tf.zeros([len_sample, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
	random_range = np.random.choice(size_train, 100)
	batch_xs = [train_sample[i] for i in random_range]
	batch_ys = [oh_train_label[i] for i in random_range]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("The score: ")
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x: test_sample, y_: oh_test_label}) * 100, "%")

