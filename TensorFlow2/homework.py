import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['STSong']  # 解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus'] = False  # 解决复述坐标显示问题

x1 = np.array(
    [137.9, 104.5, 100.0, 124.32, 79.20, 99.0, 124.0, 114.0, 106.69, 138.05, 53.75, 46.91,
     48.00, 63.02, 81.26, 86.21])
x2 = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
y = np.array([145.0, 110.0, 93.0, 116.0, 65.32, 104.0, 118.0, 91.0, 62.0, 133.0, 51.0, 45.0,
              78.5, 69.65, 75.69, 95.30])

data = np.stack((x1, x2, y), axis=1)
data_original = np.stack((x1, x2, y), axis=1)


print('data:', data)


def preprocess_data(data_init):
    data_init -= np.mean(data_init, axis=0)
    data_init /= np.std(data_init, axis=0)
    return data_init
    # miu = []
    # S = []
    # V = []
    # P = []
    # miu.append(np.mean(data[:, 0]))
    # miu.append(np.mean(data[:, 1]))
    # miu.append(np.mean(data[:, 2]))
    # print(miu)
    # # 方式一
    # S.append(np.std(data[:, 0]))
    # S.append(np.std(data[:, 1]))
    # S.append(np.std(data[:, 2]))
    # # 方式二
    # for i in range(3):
    #     V.append(np.std(data[:, i]))
    # # 方式三
    # f = lambda data: np.std(data, axis=0)
    # P[:3] = f(data)
    # for i in range(0, len(data[:, 0])):
    #     data[i, 0:3] = (data[i, 0:3] - miu[0:3]) / S[0:3]
    # print('normalization', data)




def mse(b, w1, w2, data):
    totalError = 0
    for i in range(0, len(data)):
        x1 = data[i, 0]
        x2 = data[i, 1]
        y = data[i, 2]
        totalError = totalError + (y - (w1 * x1 + w2 * x2 + b)) ** 2
    return totalError / float(len(data) / 2)


def step_gradient(b_current, w1_current, w2_current, data, lr):  # 对mse求x1 x2偏导数
    b_gradient = 0
    w1_gradient = 0
    w2_gradient = 0
    M = len(data)

    for i in range(0, len(data)):
        x1 = data[i, 0]
        x2 = data[i, 1]
        y = data[i, 2]
        b_gradient += (2 / M) * ((w1_current * x1 + w2_current * x2 + b_current) - y)
        w1_gradient += (2 / M) * w1_current * ((w1_current * x1 + w2_current * x2 + b_current) - y)
        w2_gradient += (2 / M) * w2_current * ((w1_current * x1 + w2_current * x2 + b_current) - y)

    new_b = b_current - (lr * b_gradient)
    new_w1 = w1_current - (lr * w1_gradient)
    new_w2 = w2_current - (lr * w2_gradient)
    return [new_b, new_w1, new_w2]


def gradient_descent(data, start_b, start_w1, start_w2, lr, num):
    data = preprocess_data(data)
    b = start_b
    w1 = start_w1
    w2 = start_w2
    losses = []
    for step in range(num):
        b, w1, w2 = step_gradient(b, w1, w2, data, lr)
        loss = mse(b, w1, w2, data)
        losses.append(loss)
        if step % 100 == 0:
            print(f'iterations:{step}, loss{loss}, w1,w2:{w1, w2}, b{b}')
    epoches = range(1, num + 1)
    plt.plot(epoches, losses, 'bo')
    plt.xlabel('训练次数')
    plt.ylabel('损失值')
    plt.title('房价预测')
    plt.show()
    return [b, w1, w2]


if __name__ == '__main__':
    lr = 0.0001
    np.random.seed(0)
    init_b = np.random.rand(1)
    np.random.seed(0)
    init_w1 = np.random.rand(1)
    np.random.seed(0)
    init_w2 = np.random.rand(1)
    num = 10000
    b, w1, w2 = gradient_descent(data, init_b, init_w1, init_w2, lr, num)
    fig = plt.figure(1)
    ax1 = Axes3D(fig)
    ax1.scatter3D(data[:, 0], data[:, 1], data[:, 2], edgecolors='y')
    x1, x2 = np.meshgrid(data[:, 0], data[:, 1])
    y1 = w1 * x1 + w2 * x2 + b
    ax1.plot_surface(x1, x2, y1, color='r')
    ax1.set_xlabel('面积')
    ax1.set_ylabel('房间数')
    ax1.set_zlabel('价格')
    ax1.set_title('房价预测系统')
    plt.show()

    # ax1.plot3D(x1, x2, y1, 'r')

    fig2 = plt.figure(2)
    ax = Axes3D(fig2)
    ax.scatter3D(data_original[:, 0], data_original[:, 1], data_original[:, 2], edgecolors='y')
    original_x1, original_x2 = np.meshgrid(data_original[:, 0], data_original[:, 1])
    y = w1 * original_x1 + w2 * original_x2 + b
    ax.plot_surface(original_x1, original_x2, y, color='r')
    ax.set_xlabel('面积')
    ax.set_ylabel('房间数')
    ax.set_zlabel('价格')
    ax.set_title('房价预测系统')
    plt.show()
