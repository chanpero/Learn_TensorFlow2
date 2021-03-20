import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # a = tf.constant([1, 2, 3, 1, 1])
    # b = tf.constant([0, 1, 3, 4, 5])
    # c = tf.where(tf.greater(a, b), a , b)
    # print(c)

    # rdm = np.random.RandomState(seed=1)
    # a = rdm.rand()
    # b = rdm.rand(2, 3)
    # print(a, b)
    # a = np.random.RandomState().rand()
    # print(a)

    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    # c = np.vstack((a, b))
    # print(c)

    # x, y = np.mgrid[1:3:1, 2:4:0.5]
    # print(x)
    # print(y)
    # grid = np.c_[x.ravel(), y.ravel()]
    # print(x.ravel())
    # print(grid)

    y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
    y_pro = tf.nn.softmax(y)
    loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)
    loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)
    print('categorical_crossentropy:\n', loss_ce1)
    print('softmax_cross_entropy_with_logits:\n', loss_ce2)