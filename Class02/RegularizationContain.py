import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# 读入数据/标签 生成x_train y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.001, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01
EPOCH = 400

# Train
for epoch in range(EPOCH):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            loss = tf.reduce_mean(tf.square(y_train - y))

            # MSE 均方差损失函数
            loss_regularization = []
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            loss_reg_sum = tf.reduce_sum(loss_regularization)

            loss_mse = loss + 0.03 * loss_reg_sum

        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch % 20 == 0:
        print('Epoch: ', epoch, ', Loss: ' , float(loss))

# Test
print('\n**************Predict************')
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

probs = []
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

x1 = x_data[:, 0]
x2 = x_data[:, 1]
probs = np.array(probs).reshape(xx.shape)
# squeeze去掉纬度是1的纬度,相当于去掉[['red'],[''blue]],内层括号变为['red','blue']
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标xx yy和对应的值probs放入contour函数，给probs值为0.5的所有点上色  plt点show后显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
