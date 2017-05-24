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

    print ('p')
    for step in range(in_times):
        print('step')
        _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

    data = {"x":[], "y":[], "cluster": []}

    print(len(assignment_values))
    for i in range(len(assignment_values)):
        print(i)
        data["x"].append(in_XnYs[i][0])
        data["y"].append(in_XnYs[i][1])
        data["cluster"].append(assignment_values[i])

    print(data)
    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
#    plt.plot(data["x"], data["y"], 'ro')
    plt.show()
    print('s')

def main():
    #first_tf()
    xy = preprare_data(0.1, 0.3, 2000, 'b')
    #linear_regress(xy, 100)
    k_means(xy, 4, 100)

if __name__ == "__main__":
    main()