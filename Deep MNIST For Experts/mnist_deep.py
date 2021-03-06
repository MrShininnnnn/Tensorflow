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

print("Graph loading...")

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Training...")

def progbar(curr, total, full_progbar):
	frac = curr/(total-100)
	filled_progbar = round(frac*full_progbar)
	print("\r", "#"*filled_progbar + "_"*(full_progbar - filled_progbar), "[{:>7.2%}]".format(frac), end="")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		random_range = np.random.choice(size_train, 50)
		batch_xs = [train_sample[i] for i in random_range]
		batch_ys = [oh_train_label[i] for i in random_range]
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
			progbar(i, 20000, 100)
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	test_acc = accuracy.eval(feed_dict={x: test_sample, y_: oh_test_label, keep_prob: 1.0})
	print("\ntest accuracy:{:.2%}".format(test_acc))