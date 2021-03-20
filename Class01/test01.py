import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # a = tf.constant([[1, 2, 3]], dtype=tf.int64)
    # print(a)
    # print(a.dtype)
    # print(a.shape)
    #
    # a = np.arange(0, 5)
    # print(a)
    # print(tf.convert_to_tensor(a, dtype=tf.int64))

    # a = tf.zeros([3, 4])
    # b = tf.ones(3)
    # c = tf.fill([2, 2], 9)
    # print(a, b, c)
    #
    # d = tf.random.normal([2, 2], mean=0.5, stddev=1)
    # print(d)
    # e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
    # print(e)
    #
    # f = tf.random.uniform([2, 2], minval=-1, maxval=1.01)
    # print(f)

    # a = tf.constant([1, 2, 3], dtype=tf.int64)
    # b = tf.cast(a, dtype=tf.int32)
    # print(a)
    # print(b)
    # print(tf.reduce_min(a))
    # print(tf.reduce_max(a))
    #
    # a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    # print(tf.reduce_mean(a, axis=0))
    # print(tf.reduce_sum(a, axis=1))

    # features = tf.constant([12, 23, 10, 18])
    # labels = tf.constant([0, 1, 1, 0])
    # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # print(dataset)
    # for element in dataset:
    #     print(element)

    # with tf.GradientTape() as tape:
    #     w = tf.Variable(tf.constant(3.0))
    #     loss = tf.add(w, tf.pow(w, 2))
    # grad = tape.gradient(loss, w)
    # print(grad)

    # classes = 3
    # labels = tf.constant([1, 0, 2])
    # output = tf.one_hot(labels, depth=classes)
    # print(output)

    # y = tf.constant([1.01, 2.01, -0.66])
    # y_pro = tf.nn.softmax(y)
    # print('After softmax: ', y_pro)

    # w = tf.Variable(4)
    # w.assign_sub(2)
    # print(w)

    test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(tf.argmax(test, axis=0))
    print(tf.argmax(test, axis=1))

