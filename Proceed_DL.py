from keras.datasets import imdb
import numpy as np
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)
# 拿到了10000个单词，但是有25000个？？什么东西
print('train_datas.shape,train_datas.ndim ', train_datas.shape, train_datas.ndim)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 正确的写法是 zeros((2,1024))，
    # python的二维数据表示要用二层括号来进行表示。三维要用三层括号
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
x_train = vectorize_sequences(train_datas)
x_test = vectorize_sequences(test_datas)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(x_test.shape)

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
x_val = x_train[:10000]  # 将
partial_x_train = x_train[10000:]

y_val = y_train[:10000]  # 前10000的当做检验集
partial_y_train = y_train[10000:]  # 10000-25000的留下当做训练集
print(partial_x_train.shape,partial_y_train.shape)

history = model.fit(partial_x_train, partial_y_train, epochs=3, batch_size=512,
                    validation_data=(x_val, y_val))
import matplotlib.pyplot as plt
history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)  # 设定1-20 ，x轴
plt.plot(epochs, loss_values, 'bo', label='Training loss')  # x轴，y轴，bo为点
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # b为线
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs, acc, 'bo', label='Traning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
