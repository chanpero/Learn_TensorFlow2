import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plt.imshow(x_train[5], cmap='gray')
# plt.show()

# print('x_train[0]: \n', x_train[0])
# print('y_train[0]: \n', y_train[0])
#
# print("x_train.shape:\n", x_train.shape)
# print("y_train.shape:\n", y_train.shape)
# print("x_test.shape:\n", x_test.shape)
# print("y_test.shape:\n", y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

model.summary()
