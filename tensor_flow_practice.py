# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def first_tf ():
    a = tf.placeholder("float")
    b = tf.placeholder("float")

    y = tf.multiply(a,b)

    sess  = tf.Session()

    print (sess.run(y, feed_dict={a:3, b:3}))


def preprare_data(in_w, in_b, in_ptnum, in_dtcat='a'):
    a = in_ptnum
    vecs = []

    for i in range(a):
        if in_dtcat == 'a':
            x1 = np.random.normal(0.0, 0.55)
            y1 = in_w * x1 + in_b + np.random.normal(0.0,0.03)
            vecs.append([x1, y1])
        else:
            if np.random.random() > 0.5:
                vecs.append([np.random.normal(0.0,0.9), np.random.normal(0.0,0.9)])
            else:
                vecs.append([np.random.normal(3.0,0.5), np.random.normal(1.0,0.5)])

#    plt.plot([x[0] for x in vecs], [y[1] for y in vecs], "ro")
#    plt.show()

#    df = pd.DataFrame({'x': [v[0] for v in vecs], 'y': [v[1] for v in vecs]})
#    sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
#    plt.show()

    return vecs

def linear_regress(in_XnYs, in_times):
    x_d = [x[0] for x in in_XnYs]
    y_d = [y[1] for y in in_XnYs]
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_d + b
    loss = tf.reduce_mean(tf.square(y-y_d))
    optmzr = tf.train.GradientDescentOptimizer(0.5)
    train = optmzr.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # iterate <in_times> times (with SGD)
    for step in range(in_times):
        sess.run(train)
        print ("step {0}: W - {1} , b - {2}, loss - {3}".format(step, sess.run(W), sess.run(b), sess.run(loss)))
        #print ("step {0}: loss - {1}".format(step, sess.run(loss)))

    plt.plot(x_d, y_d, 'ro')
    plt.plot(x_d, sess.run(W) * x_d + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
    plt.show()

def k_means(in_XnYs, in_k, in_times):

    vectors = tf.constant(in_XnYs)
    k = in_k
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))
    expanded_vectors = tf.expand_dims(vectors,0)
    expanded_centroids = tf.expand_dims(centroids,1)

    assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)),2),0)

    mean_reds =  [tf.reduce_mean (tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments,c)), [1,-1] ) ), reduction_indices=[1] ) for c in range(k) ]
    means = tf.concat (mean_reds, 0)
    print("{0} -> {1}".format(mean_reds,means))

    update_centroids = tf.assign(centroids, means)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)

#   텐서 내부의 값을 찍을 때는 sess.run(var)를 print하자
#    for c in [0,1,2,3]:
#        ee = tf.where(tf.equal(assignments,c))
#        print(sess.run(ee))

    print ('p')
    for step in range(in_times):
        _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

    data = {"x":[], "y":[], "cluster": []}

    print(len(assignment_values))
    for i in range(len(assignment_values)):
        data["x"].append(in_XnYs[i][0])
        data["y"].append(in_XnYs[i][1])
        data["cluster"].append(assignment_values[i])

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()

def mnist_simple_nn(in_lR, in_iN):
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    tf.convert_to_tensor(mnist.train.images).get_shape()

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    x = tf.placeholder("float", [None, 784]) # connected to training data from MNIST
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y_t = tf.placeholder("float", [None, 10]) # connected to labels from MNIST

    cross_entropy = -tf.reduce_sum(y_t * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(in_lR).minimize(cross_entropy)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for i in range(in_iN):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_t:batch_ys})

    correct_precision = tf.equal(tf.argmax(y, 1), tf.argmax(y_t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_precision, "float"))

    rst = sess.run(accuracy, feed_dict={x:mnist.test.images, y_t: mnist.test.labels})
    print (rst)

def wgt_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant (0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pooling_2x2 (x):
    return tf.nn.max_pool (x, ksize=[1,2,2,1], strides=[1,2,2,1], padding ="SAME")

def mnist_cnn(in_Ap, in_iN):

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder("float", shape = [None, 784]) # holds input image data  (data size is undefined)
    y_t = tf.placeholder("float", shape= [None, 10]) # holds answer set (data size is undefined)

    x_img = tf.reshape(x, [-1,28,28,1])
    print ("x_image={0}".format(x_img))

    W_conv1 = wgt_variable([5,5,1,32]) # 32 is current filter num
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
    h_pool1 = max_pooling_2x2(h_conv1)

    W_conv2 = wgt_variable([5,5,32,64]) # 32 is previous filter num, as current channel num. 64 is current filter num
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pooling_2x2(h_conv2)

    W_fc1 = wgt_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

    keep_prob = tf.placeholder("float") # connect to keeping probability for drop-out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = wgt_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_t * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(in_Ap).minimize(cross_entropy)
    correct_precision = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_precision, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range (in_iN):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_t:batch[1], keep_prob:1.0})
            print ("step {0}, training accuracy: {1}%".format(i, train_accuracy))
        sess.run(train_step, feed_dict={x:batch[0], y_t:batch[1], keep_prob:0.5})

    print("test accuracy: {0}".format(sess.run(accuracy,
         feed_dict={x:mnist.test.images, y_t:mnist.test.labels, keep_prob:1.0})))

def system_test():
    import tensorflow as tf
    # 아래 한 줄은 GPU에서만 동작함
    #with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2,3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='b')
    c = tf.matmul(a,b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print (sess.run(c))



def main():
    #first_tf()
    #xy = preprare_data(0.1, 0.3, 2000, 'b')
    #linear_regress(xy, 100)
    #k_means(xy, 4, 100)
    mnist_simple_nn(0.01, 1000)
    #mnist_cnn(1e-4, 1000)
    #system_test()

if __name__ == "__main__":
    main()