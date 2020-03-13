from keras.datasets import imdb
import numpy as np
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)
# 拿到了10000个单词，类似字典的东西，其实总数25000
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_d = models.Sequential()
model_d.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model_d.add(layers.Dropout(0.5))
model_d.add(layers.Dense(16, activation='relu'))
model_d.add(layers.Dropout(0.5))
model_d.add(layers.Dense(1, activation='sigmoid'))
model_d.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
x_val = x_train[:10000]  # 将
partial_x_train = x_train[10000:]

y_val = y_train[:10000]  # 前10000的当做检验集
partial_y_train = y_train[10000:]  # 10000-25000的留下当做训练集
print(partial_x_train.shape,partial_y_train.shape)

history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512,
                    validation_data=(x_val, y_val))
history_d = model_d.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
history_dict = history.history
history_d_dict = history_d.history
loss_values_d = history_d_dict['loss']
val_loss_values_d = history_d_dict['val_loss']


loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
acc_d = history_d_dict['accuracy']
val_acc_d = history_d_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)  # 设定1-20 ，x轴
plt.plot(epochs, loss_values, 'bo', label='Training loss')  # x轴，y轴，bo为点
plt.plot(epochs, val_loss_values, 'b-', label='Validation loss')  # b为线
plt.plot(epochs, loss_values_d, 'ro', label='Training_d loss')  # x轴，y轴，bo为点
plt.plot(epochs, val_loss_values_d, 'r-', label='Validation_d loss')  # b为线
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs, acc, 'bo', label='Traning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, acc_d, 'r+', label='Traning_d acc')
plt.plot(epochs, val_acc_d, 'r-', label='Validation_d acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
prediction = model.predict(x_test)
print('prediction[0]:', prediction[0], prediction[0].shape)
# 输出一个概率，相当于二分类问题
# 输出：prediction[0]: [0.24827671] (1,)，这就是一个好评与否的概率
