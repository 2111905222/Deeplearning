from keras.datasets import boston_housing

(train_datas, train_targets), (test_datas, test_targets) = boston_housing.load_data()

# 数据预处理,标准化处理,即数值本身减去平均数再除以标准差,因为房子和房子之间价格差距过大
mean = train_datas.mean(axis=0)
train_datas -= mean  # 减去平均数
std = train_datas.std(axis=0)  # 标准差
train_datas /= std
# 注意，无论是测试数据还是训练数据都要从这里进行统一标准化！！
from keras import layers
from keras import models


def build_model():  # 问题需要将同一个模型多次实例化，所以用同一个函数来构建模型
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_datas.shape[1],)
                           )
              )  # 这里不要漏了一个逗号
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


import numpy as np

k = 4  # 准备4个分区
num_val_samples = len(train_datas) // k
# num_epochs = 100
# all_scores = []
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('process fold', i)  # 0 1 2 3
    val_data = train_datas[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_datas[:i * num_val_samples],
         train_datas[(i + 1) * num_val_samples:]],
        axis=0)  # 当axis为0时增加将第二个参数的数据以行形式增加到第一个参数处
    partial_train_target = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()  # 构建新模型，已经编译了的，但是是全新，未经训练的
    history = model.fit(partial_train_data, partial_train_target,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)  # verbose=0，静默模式，就是不输出数据
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)  # 在验证数据上评估模型
    # all_scores.append(val_mae)
    print(history.history.keys())
    mae_history = history.history['val_mean_absolute_error']
    print('mae_history:', mae_history)  # 是一个200个数字的列表
    all_mae_histories.append(mae_history)
# print(all_scores, np.mean(all_scores))
# [2.275611639022827, 2.544431686401367, 2.628783941268921, 2.5923256874084473] 2.5102882385253906
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]  # 500次迭代都求出每次的迭代的平均值
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation')
plt.show()
test_mse_score, test_mae_score = model.evaluate(test_datas, test_targets)
print('test_mse_score, test_mae_score:', test_mse_score, test_mae_score)