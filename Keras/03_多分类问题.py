from keras.datasets import reuters
(train_datas, train_labels), (test_datas, test_labels) = reuters.load_data(num_words=10000)
# 与IMDB一样是前10000个常用单词
print('train_datas.shape & train_datas.ndim &  train_labels.len: ', train_datas.shape, train_datas.ndim, len(train_labels))
print('train_datas[1], train_datas[1].shape, type(train_datas)', train_datas[1],  type(train_datas))
# 每个样本都是一个单词索引组成的ndarray, 输出train_datas[1].shape时出现错误'list' object has no attribute 'shape'

import  numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 正确的写法是 zeros((2,1024))，
    # python的二维数据表示要用二层括号来进行表示。三维要用三层括号
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
x_train = vectorize_sequences(train_datas)
x_test = vectorize_sequences(test_datas)
print('数据向量化与one-hot化后：x_train.shape, x_test.shape', x_train.shape, x_test.shape )
# 将标签向量化
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
# =one_hot_test_labels = to_one_hot(test_labels)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # 输出关注点
# 添加正则化
from keras import regularizers
model_re = models.Sequential()
model_re.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model_re.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model_re.add(layers.Dense(46, activation='softmax'))
model_re.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  #最后的参数需要[]
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val))
history_re = model_re.fit(partial_x_train, partial_y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val))
# 输出图像
import matplotlib.pyplot as plt
history_dict = history.history
history_re_dict = history_re.history

loss_value_re = history_re_dict['loss']
val_loss_value_re = history_re_dict['val_loss']

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)  # 设定1-20 ，x轴
plt.plot(epochs, loss_values, 'b*', label='original_Training loss')  # x轴，y轴，bo为点
plt.plot(epochs, val_loss_values, 'b+', label='original_Validation loss')  # b为线
plt.plot(epochs, loss_value_re, 'ro-', label='re_loss_value')
plt.plot(epochs, val_loss_value_re, 'r-', label='re_validation_loss_value')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
predictions = model.predict(x_test)
print('predictions[0].shape, predictions[0]', predictions[0].shape, predictions[0])
# 输出46个标签的分类概率
# 注意，这里分类的标签有46个那么多，所以中间的隐藏层要多一点，如果只取4个的话会导致信息不足从而大大影响精度
# predictions[0].shape, predictions[0] (46,) [4.48676610e-06 1.14068964e-04 1.36330063e-05 9.41669345e-01
#  5.03556393e-02 2.27316577e-05 1.61731459e-05 1.33219737e-05
#  3.61729437e-03 9.47952117e-07 1.76528865e-05 2.76752689e-04
#  6.39222117e-06 3.01166801e-05 2.40595200e-05 1.21386356e-05
#  3.64544278e-04 1.50961554e-04 3.68432156e-05 2.17982088e-04
#  9.01179912e-04 9.46540604e-06 3.41155851e-06 1.26864659e-
#  4.27874284e-06 5.02825969e-05 2.96143526e-07 2.48278798e-06
#  5.34254650e-04 4.53481953e-05 1.45051963e-04 2.61251698e-04
#  4.74590997e-05 6.16024454e-06 3.55395809e-04 8.73993031e-06
#  2.17685854e-04 5.72450190e-05 3.52448160e-06 1.84454650e-04
#  3.68323242e-07 3.03287561e-05 6.56310749e-06 1.32952437e-05
#  6.23199730e-06 1.31614952e-05]
