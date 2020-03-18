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


def create_model(layer_num):
    model = models.Sequential()
    model.add(layers.Dense(layer_num, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(layer_num, activation='relu'))
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
original_model = create_model(16)
Smaller_model = create_model(4)
bigger_model = create_model(512)
original_history = original_model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                                      validation_data=(x_val, y_val))
smaller_history = Smaller_model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20,
                                  validation_data=(x_val, y_val))
bigger_history = bigger_model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20,
                                  validation_data=(x_val, y_val))
# 绘制图像
import matplotlib.pyplot as plt


def plot_data(history_dict,  shape, **label_name):
    if not (isinstance(shape, str)):
        shape = 'b'
    loss_values = []
    val_loss_values = []
    acc = []
    val_acc = []
    for i in range(len(history_dict)):
        history = history_dict[i]
        print(type(history))
        loss_values.append(history['loss'])
        val_loss_values.append(history['val_loss'])
        acc.append(history['accuracy'])
        val_acc.append(history['val_accuracy'])
    epochs = range(1, len(loss_values[0]) + 1)  # 设定1-20 ，x轴
    color = ['bo-', 'ro-', 'yo-']
    label_names = list(label_name.values())
    for i in range(len(loss_values)):
        plt.plot(epochs, loss_values[i], color[i], label=label_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    for i in range(len(val_loss_values)):
        plt.plot(epochs, val_loss_values[i], color[i], label=label_names[i])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()
    # plt.plot(epochs, acc, 'bo', label='Traning acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()


history_dict = [original_history.history, smaller_history.history, bigger_history.history]
plot_data(history_dict, shape='bo', label_name='original', label_name1='smaller', label_name2='bigger')
# prediction = model.predict(x_test)
# prediction1 = model1.predict(x_test)
# print('prediction[0]:', prediction[0], prediction[0].shape)
# print('prediction1[0]:', prediction1[0], prediction1[0].shape)
# 输出一个概率，相当于二分类问题
# 输出：prediction[0]: [0.24827671] (1,)，这就是一个好评与否的概率
