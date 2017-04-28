# -*- coding: utf-8 -*-

def cat_say(text):
    """말하는 고양이 그림을 만든다"""
    textlen = str(text).__len__()
    print ('        {0}'.format(' ' * textlen))
    print ('       <{0}>'.format(text))
    print ('        {0}'.format('-' * textlen))
    print ('        /')
    print ('        //')

import tensorflow as tf

def main():
    # in python2, use raw_input () instead of input ()
    #text = input ('what can cat say?')
    #cat_say(text)

    a = tf.placeholder("float")
    b = tf.placeholder("float")

    y = tf.multiply(a,b)

    sess = tf.Session()

    print(sess.run(y, feed_dict={a:3, b:3}))

if __name__ == '__main__':
    main()
