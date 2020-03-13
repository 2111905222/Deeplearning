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


def original_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def Smaller_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


x_val = x_train[:10000]  # 将
partial_x_train = x_train[10000:]

y_val = y_train[:10000]  # 前10000的当做检验集
partial_y_train = y_train[10000:]  # 10000-25000的留下当做训练集
print(partial_x_train.shape, partial_y_train.shape)
model = original_model()
model1 = Smaller_model()
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))
history1 = model1.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20,
                      validation_data=(x_val, y_val))

# 绘制图像
import matplotlib.pyplot as plt


def plot_data(history_dict, history_dict1, shape='b',
              label_name='Validation loss', label_name1='Validation loss'):
    if not(isinstance(shape, str)):
        shape = 'b'
    if not (isinstance(label_name, str)):
        label_name = 'Validation loss'
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    loss_values1 = history_dict1['loss']
    val_loss_values1 = history_dict1['val_loss']
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)  # 设定1-20 ，x轴
    plt.plot(epochs, val_loss_values, shape, label=label_name)  # x轴，y轴，bo为点
    plt.plot(epochs, val_loss_values1, 'b', label=label_name1)  # b为线
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()
    # plt.clf()
    # plt.plot(epochs, acc, 'bo', label='Traning acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()


history_dict1 = history1.history
history_dict = history.history
print(history_dict.keys())
plot_data(history_dict, history_dict1, shape='bo', label_name='original', label_name1='smaller')
prediction = model.predict(x_test)
prediction1 = model1.predict(x_test)
print('prediction[0]:', prediction[0], prediction[0].shape)
print('prediction1[0]:', prediction1[0], prediction1[0].shape)
# 输出一个概率，相当于二分类问题
# 输出：prediction[0]: [0.24827671] (1,)，这就是一个好评与否的概率
