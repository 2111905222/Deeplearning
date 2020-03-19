import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# # 旧版本的tensorflow格式
from numpy import float32

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.version)
# 创建计算图
a = tf.placeholder(tf.float32, name='variable_a')
b = tf.placeholder(tf.float32, name='variable_b')
c = tf.add(a, b, name='variable_c')
# 运行计算图
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
c_numpy = sess.run(c, feed_dict={a:2.0, b:4.0})
print('a+b=', c_numpy)
'''
# -------------------------------------------------------------------------
# tensorflow2.0中的计算声明
a1 = tf.constant(2.)
b1 = tf.constant(4.)
print('a+b', a1 + b1)

# -------------------------------------------------------------------------
# cpu与gpu在运算能力上的比较
n = 10 ** 7
# 创建在cpu上运行的两个矩阵
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1, n])
    cpu_b = tf.random.normal([n, 1])
    print(cpu_a.device, cpu_b.device)
# 创建在gpu上运行的两个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1, n])
    gpu_b = tf.random.normal([n, 1])
    print(gpu_a.device, gpu_b.device)


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c


def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c


# 第一次计算需要热身，避免将初始化阶段算进来
timeit.timeit(cpu_run, number=10)
timeit.timeit(gpu_run, number=10)
# 正式计算10次，取平均时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time * 10000, gpu_time * 10000)  # 在n较小时几乎没有什么变化,但是n较大时变化就很明显了

# -------------------------------------------------------------------------
# 自动梯度求导
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:  # 构建梯度环境
    tape.watch([w])  # 将w加入梯度跟踪列表
    # 构建计算过程
    y = a * w ** 2 + b * w + c
# 求导
[dy_dw] = tape.gradient(y, [w])
print('导数为：', dy_dw)

# --------------------------------------------------------------
# 线性模型实战
# 第一步：采样数据
data = []  # 保存样本集的列表
for i in range(100):
    x = np.random.uniform(-10., 10.)  # 随机采样100个输入点
    eps = np.random.normal(0., 0.1)  # 采样高斯噪声，均值为0，方差为0.1的平方的高斯分布随机取噪声
    y = 1.477 * x + 0.089 + eps  # 得到模型的输出,加入噪声是为了非线性
    data.append([x, y])  # 保存样本点
data = np.array(data)  # 转换为2D Numpy数组,就是单纯地变成了数组
print('data:', data)


# 第二步：计算误差
def mse(b, w, points):
    # 根据当前的w，b参数计算均方差损失
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]  # 获得i号点的输入x
        y = points[i, 1]  # 获得i号点的输出y
        # 计算差的平方，并累加
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))  # 得到均方差


def step_gradient(b_current, w_current, points, lr):
    # 计算误差函数在所有点上的倒数，并更新w，b
    b_gradient = 0
    w_gradient = 0
    m = float(len(points))
    total = len(points)  # 注意 浮点数不能变成整数，即不能total = m
    for i in range(0, total):
        x = points[i, 0]
        y = points[i, 1]
        # 参考误差函数对b的导数： grad_b = 2(wx+b-y)
        b_gradient += (2 / m) * ((w_current * x + b_current) - y)
        # 参考误差函数对b的导数： grad_w = 2x(wx+b-y)
        w_gradient += (2 / m) * x * ((w_current * x + b_current) - y)  # 每个数据的误差都进行迭代+上
        # 根据梯度下降算法更新w，b lr为学习率
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


def gradient_descent(points, starting_b, starting_w, lr, num_iteraions):
    # 循环更新w，b多次
    b = starting_b
    w = starting_w
    losses = []
    print('points:', points)
    print('points:', np.array(points))
    for step in range(num_iteraions):  # 迭代1000次
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        losses.append(loss)
        if step % 100 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w], losses  # 返回最后一次的w，b


def main():
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    epoches = range(0, num_iterations)
    [b, w], losses = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')
    plt.plot(epoches, losses, '-', label='mse')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('mse')
    plt.show()


if __name__ == '__main__':
    main()
