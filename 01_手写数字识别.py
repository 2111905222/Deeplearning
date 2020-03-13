from keras.datasets import mnist
from keras import models
from keras import layers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images & train_labels: ', train_images.shape, len(train_labels))
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
prediction = network.predict(test_images)
print('prediction[0]:&shape & test_label', prediction[0], prediction[0].shape, test_labels[0])
# 同样输出10个数字的概率
# prediction[0]:&shape & test_label [5.5214233e-09 6.4628702e-10 6.6159851e-07 1.4738458e-05 1.9960192e-13
#  4.0310167e-07 4.7862078e-14 9.9998403e-01 8.9099025e-08 1.4387281e-07] (10,) [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
