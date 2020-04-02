import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random
from tensorflow import optimizers


# t1 = tf.convert_to_tensor(x1， dtype=tf.float32)
# # print(t1.shape, type(t1))  # 效果和cast一样
# x1 = tf.reshape(x1, [-1, 1])
# x2 = tf.reshape(x2, [-1, 1])
# y = tf.reshape(y, [-1, 1])
# print(x1.shape, x2.shape, y.shape, type(x1), type(x2), type(y))
# train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 这个dataset方便弄batch和repeat
# print('......', train_db)


def main():
    x1 = np.array(
        [137.9, 104.5, 100.0, 124.32, 79.20, 99.0, 124.0, 114.0, 106.69, 138.05, 53.75, 46.91, 48.00, 63.02, 81.26,
         86.21])
    x2 = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
    y = np.array(
        [145.0, 110.0, 93.0, 116.0, 65.32, 104.0, 118.0, 91.0, 62.0, 133.0, 51.0, 45.0, 78.5, 69.65, 75.69, 95.30])

    x1 = tf.cast(x1, dtype=tf.float32)
    x2 = tf.cast(x2, dtype=tf.float32)
    x = tf.stack([x1, x2], axis=0)
    x = (x - tf.reduce_mean(x)) / np.std(x)
    y = tf.cast(y, dtype=tf.float32)
    print(x.shape, y.shape, type(x), type(y))
    lr = 0.0001
    accs, losses = [], []
    # w1, w2, b = tf.Variable(tf.random.normal([1, 1])), \
    #             tf.Variable(tf.random.normal([1, 1])), \
    #             tf.Variable(tf.zeros([1]))  # random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
    w = tf.Variable(np.random.random([1, 2]), trainable=True, dtype=tf.float32)
    b = tf.Variable(random.random(), trainable=True, dtype=tf.float32)
    print(w, b)
    # print(w, b)
    # for step, (x1, x2, y) in enumerate(train_db):
    # w1 = tf.reshape(w1, [1, 1])
    # w2 = tf.reshape(w2, [1, 1])
    # b = tf.reshape(b, [1, 1])
    # print(x1.shape, x2.shape, y.shape, type(x1), type(x2), type(y))
    # print(w1.shape, w2.shape, b.shape, type(w1), type(w2), type(b))
    num = 10000
    optimizer = optimizers.SGD(lr)
    for i in range(num):
        with tf.GradientTape() as tape:
            out = tf.matmul(w, x) + b

            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)
            losses.append(loss)
            grads = tape.gradient(loss, [w, b])

            optimizer.apply_gradients(zip(grads, [w, b]))
        # if i % 100 == 0:
        #
        #     print('loss', loss)

    # w1.assign_sub(lr * grads[0])
    # w2.assign_sub(lr * grads[1])
    # b.assign_sub(lr * grads[2])  # 简单点说就是 -=

    # for p, g in zip([w1, w2, b], grads):
    #     p.assign_sub(lr * g)

    # print(step, 'loss:', float(loss))
    plt.figure(1)
    epoches = list(range(0, num))
    plt.plot(epoches, losses, color='C0', marker='s')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()

    fig = plt.figure(2)
    ax = Axes3D(fig)
    ax.scatter3D(x[0, :], x[1, :], y, edgecolors='b')
    mesh1, mesh2 = tf.meshgrid(x[0, :], x[1, :])
    y1 = w[0, 0] * mesh1 + w[0, 1] * mesh2 + b
    ax.plot_surface(mesh1, mesh2, y1, color='r')
    plt.show()
    # plt.savefig('train.svg')


if __name__ == '__main__':
    main()
